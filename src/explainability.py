import torch
import pandas as pd
import numpy as np
import copy
from pathlib import Path
from src.config import *
from src.utils import setup_logger

logger = setup_logger("explainability")

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


logger = setup_logger("v27_explainability")

def calculate_permutation_importance(model, val_loader, criterion, device, feature_names):
    """
    Fallback Permutation Importance if SHAP fails or is too slow.
    Works by randomly shuffling one feature column at a time and measuring drop in F1.
    """
    model.eval()
    
    # 1. Get baseline F1
    all_preds = []
    all_labels = []
    baseline_seq, baseline_num, baseline_cat = [], [], []
    with torch.no_grad():
        for seq_x, num_x, cat_x, labels, _ in val_loader:
            seq_x = seq_x.to(device) if seq_x is not None else None
            num_x = num_x.to(device) if num_x is not None else None
            cat_x = cat_x.to(device) if cat_x is not None else None
            
            if seq_x is not None: baseline_seq.append(seq_x.cpu())
            if num_x is not None: baseline_num.append(num_x.cpu())
            if cat_x is not None: baseline_cat.append(cat_x.cpu())
            
            logits, _ = model(seq_x, num_x, cat_x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    baseline_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    if len(baseline_num) == 0 and len(baseline_cat) == 0:
        return pd.DataFrame()
        
    full_num = torch.cat(baseline_num, dim=0) if len(baseline_num) > 0 else None
    full_cat = torch.cat(baseline_cat, dim=0) if len(baseline_cat) > 0 else None
    full_seq = torch.cat(baseline_seq, dim=0) if len(baseline_seq) > 0 else None
    
    importances = {}
    
    # helper
    def evaluate_shuffled(num_c, cat_c, seq_c):
        p = []
        with torch.no_grad():
            batch_size = 32
            n_samples = num_c.shape[0] if num_c is not None else cat_c.shape[0]
            for i in range(0, n_samples, batch_size):
                s = seq_c[i:i+batch_size].to(device) if seq_c is not None else None
                n = num_c[i:i+batch_size].to(device) if num_c is not None else None
                c = cat_c[i:i+batch_size].to(device) if cat_c is not None else None
                logits, _ = model(s, n, c)
                p.extend(torch.argmax(logits, dim=1).cpu().numpy())
        return f1_score(all_labels, p, average='macro', zero_division=0)

    idx_feat = 0
    # Process Numerical
    if full_num is not None:
        for i in range(full_num.shape[1]):
            shuffled_num = full_num.clone()
            shuffled_num[:, i] = shuffled_num[torch.randperm(full_num.shape[0]), i]
            shuf_f1 = evaluate_shuffled(shuffled_num, full_cat, full_seq)
            importances[feature_names[idx_feat]] = baseline_f1 - shuf_f1
            idx_feat += 1
            
    # Process Categorical
    if full_cat is not None:
        for i in range(full_cat.shape[1]):
            shuffled_cat = full_cat.clone()
            shuffled_cat[:, i] = shuffled_cat[torch.randperm(full_cat.shape[0]), i]
            shuf_f1 = evaluate_shuffled(full_num, shuffled_cat, full_seq)
            importances[feature_names[idx_feat]] = baseline_f1 - shuf_f1
            idx_feat += 1
            
    df_imp = pd.DataFrame(list(importances.items()), columns=["Feature", "Importance"])
    df_imp = df_imp.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    return df_imp

def explain_model(model, val_loader, criterion, device, feature_names, out_path):
    """
    Attempts SHAP, falls back to Permutation Importance.
    """
    logger.info("Starting Explainability Analysis...")
    
    # We will just use permutation importance for robust evaluation across all architectures
    # Since SHAP requires a specific flat tensor interface which breaks on our multi-input
    # complex architectures (DeepFM, FT-Transformer with both categorical/numerical).
    
    logger.info("Using Permutation Importance (Fallback from SHAP due to multi-input model complexity).")
    df_imp = calculate_permutation_importance(model, val_loader, criterion, device, feature_names)
    
    if not df_imp.empty:
        df_imp.to_csv(out_path, index=False)
        logger.info(f"Saved feature importance to {out_path}")
    
    return df_imp


import torch
import copy
import pandas as pd
import numpy as np


logger = setup_logger("v27_counterfactual")

class GreedyCounterfactualSearcher:
    """
    Fallback Counterfactual Search.
    Finds minimal changes to actionable features to improve the predicted class.
    """
    def __init__(self, model, preprocessor, selector, spec, device):
        self.model = model
        self.preprocessor = preprocessor
        self.selector = selector
        self.spec = spec
        self.device = device
        
        self.model.eval()
        
        # Define actionable features and directions
        if spec.kind == "student":
            # (feature_name, direction: 1 for increase, -1 for decrease)
            self.actionables = {
                "studytime": {"dir": 1, "type": "num", "min": 0, "max": 1}, # scaled 0-1
                "absences": {"dir": -1, "type": "num", "min": 0, "max": 1},
                "goout": {"dir": -1, "type": "num", "min": 0, "max": 1},
                "Dalc": {"dir": -1, "type": "num", "min": 0, "max": 1},
                "Walc": {"dir": -1, "type": "num", "min": 0, "max": 1},
                "internet": {"dir": 1, "type": "cat"}
            }
            self.num_step = 0.25 # Since original values were 1-4, scaled to 0-1, step is ~0.33
        else:
            self.actionables = {
                "raisedhands": {"dir": 1, "type": "num", "min": 0, "max": 1},
                "VisITedResources": {"dir": 1, "type": "num", "min": 0, "max": 1},
                "AnnouncementsView": {"dir": 1, "type": "num", "min": 0, "max": 1},
                "Discussion": {"dir": 1, "type": "num", "min": 0, "max": 1},
                "StudentAbsenceDays": {"dir": -1, "type": "cat"} # Under-7 vs Above-7
            }
            self.num_step = 0.1 # 10 on a scale of 100

    def get_prediction(self, num_x, cat_x, seq_x=None):
        with torch.no_grad():
            if seq_x is not None: seq_x = seq_x.to(self.device)
            num_x = num_x.to(self.device) if num_x is not None else None
            cat_x = cat_x.to(self.device) if cat_x is not None else None
            logits, _ = self.model(seq_x, num_x, cat_x)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred_class].item()
            return pred_class, conf, probs[0].cpu().numpy()

    def search_for_sample(self, seq_tensor, num_tensor, cat_tensor, original_class, desired_classes, max_steps=3):
        """
        Greedy search to find counterfactual.
        Returns: success, changed_dict, new_pred_class, new_conf
        """
        if original_class in desired_classes:
            return False, {}, original_class, 1.0 # Already at target
            
        current_num = num_tensor.clone() if num_tensor is not None else None
        current_cat = cat_tensor.clone() if cat_tensor is not None else None
        
        changed_features = {}
        
        for step in range(max_steps):
            best_improvement = 0
            best_feature = None
            best_num = current_num.clone() if current_num is not None else None
            best_cat = current_cat.clone() if current_cat is not None else None
            best_pred = original_class
            best_conf = 0
            best_val = None
            
            # Try changing each actionable feature
            for feat, rules in self.actionables.items():
                if feat not in self.selector.selected_features:
                    continue
                    
                temp_num = current_num.clone() if current_num is not None else None
                temp_cat = current_cat.clone() if current_cat is not None else None
                new_val = None
                
                if rules["type"] == "num":
                    idx = [c for c in self.preprocessor.numerical_cols if c in self.selector.selected_features].index(feat)
                    curr_val = temp_num[0, idx].item()
                    new_val = curr_val + (rules["dir"] * self.num_step)
                    new_val = max(rules["min"], min(rules["max"], new_val))
                    if abs(new_val - curr_val) < 1e-4: continue # Can't change further
                    temp_num[0, idx] = new_val
                elif rules["type"] == "cat":
                    idx = [c for c in self.preprocessor.categorical_cols if c in self.selector.selected_features].index(feat)
                    curr_val = temp_cat[0, idx].item()
                    # Just a simple toggle for binary cats or pick the "better" one if known
                    # e.g., internet: yes(1) vs no(0). We want 1.
                    if feat == "internet": new_val = 1
                    elif feat == "StudentAbsenceDays": new_val = 0 # Assuming 0 is Under-7
                    else: new_val = 1 - curr_val # flip binary
                    
                    if curr_val == new_val: continue
                    temp_cat[0, idx] = new_val
                    
                pred_class, conf, probs = self.get_prediction(temp_num, temp_cat, seq_tensor)
                
                # We want to increase prob of desired class
                prob_desired = sum([probs[c] for c in desired_classes])
                
                if prob_desired > best_improvement:
                    best_improvement = prob_desired
                    best_feature = feat
                    best_num = temp_num
                    best_cat = temp_cat
                    best_pred = pred_class
                    best_conf = conf
                    best_val = new_val
                    
            if best_feature is not None:
                current_num = best_num
                current_cat = best_cat
                changed_features[best_feature] = best_val
                
                if best_pred in desired_classes:
                    return True, changed_features, best_pred, best_conf
                    
        return False, changed_features, original_class, 0.0

def generate_counterfactuals(model, dataloader, preprocessor, selector, spec, device):
    searcher = GreedyCounterfactualSearcher(model, preprocessor, selector, spec, device)
    
    results = []
    
    for idx, (seq_x, num_x, cat_x, labels, orig_indices) in enumerate(dataloader):
        if seq_x is not None: seq_x = seq_x.to(device)
        if num_x is not None: num_x = num_x.to(device)
        if cat_x is not None: cat_x = cat_x.to(device)
        
        for i in range(len(labels)):
            s_x = seq_x[i:i+1] if seq_x is not None else None
            n_x = num_x[i:i+1] if num_x is not None else None
            c_x = cat_x[i:i+1] if cat_x is not None else None
            
            orig_class, orig_conf, _ = searcher.get_prediction(n_x, c_x, s_x)
            
            if orig_class == 2: # Already High
                continue
                
            desired = [1, 2] if orig_class == 0 else [2]
            
            success, changes, new_pred, new_conf = searcher.search_for_sample(s_x, n_x, c_x, orig_class, desired)
            
            if success:
                res = {
                    "sample_idx": orig_indices[i].item(),
                    "original_class": orig_class,
                    "original_conf": orig_conf,
                    "desired_classes": desired,
                    "new_class": new_pred,
                    "new_conf": new_conf,
                    "changed_features": str(changes),
                    "num_changes": len(changes)
                }
                results.append(res)
                
    df_cf = pd.DataFrame(results)
    return df_cf


import pandas as pd
from .counterfactual import generate_counterfactuals

def create_recommendation_text(row):
    if pd.isna(row.get('new_class')) or row['original_class'] == 2:
        return "Sinh viên đang có thành tích tốt, tiếp tục duy trì phương pháp học hiện tại."
        
    orig_class_name = "Low" if row['original_class'] == 0 else "Medium"
    new_class_name = "Medium" if row['new_class'] == 1 else "High"
    
    orig_conf = row.get('original_conf', 0.0)
    new_conf = row.get('new_conf', 0.0)
    
    text = f"Sinh viên đang được dự đoán thuộc nhóm {orig_class_name} với độ tin cậy {orig_conf:.0%}. "
    
    changes = row.get('changed_features', "{}")
    if changes and changes != "{}":
        text += f"Mô hình tìm thấy phương án cải thiện: điều chỉnh các yếu tố {changes}. "
        text += f"Sau thay đổi này, dự đoán sẽ chuyển sang {new_class_name} với xác suất {new_conf:.0%}. "
        text += "Khuyến nghị: sinh viên nên thực hiện các thay đổi trên và liên hệ cố vấn học tập để theo dõi tiến độ."
    else:
        text += "Mô hình chưa tìm thấy phương án tối ưu dựa trên các biến có thể thay đổi. Đề nghị sinh viên gặp trực tiếp cố vấn học tập để phân tích sâu hơn."
        
    return text

def generate_recommendation_report(df_predictions, df_counterfactuals):
    # Merge predictions with counterfactuals
    if not df_counterfactuals.empty:
        df_merged = pd.merge(df_predictions, df_counterfactuals, left_index=True, right_on="sample_idx", how="left")
    else:
        df_merged = df_predictions.copy()
        
    df_merged['Recommendation'] = df_merged.apply(create_recommendation_text, axis=1)
    
    return df_merged


import json

def calculate_recommendation_metrics(df_cf, total_candidates):
    if len(df_cf) == 0 or total_candidates == 0:
        return {"validity": 0.0, "sparsity": 0.0, "proximity": 0.0}
        
    # Validity: ratio of successful counterfactuals / total candidates that needed them
    validity = len(df_cf) / total_candidates
    
    # Sparsity: average number of features changed
    sparsity = df_cf["num_changes"].mean()
    
    # Proximity: In this fallback, distance is roughly num_changes since we took normalized steps.
    # We can just use the inverse of sparsity as a proxy, or average changes.
    proximity = df_cf["num_changes"].mean()
    
    return {
        "validity": validity,
        "sparsity": sparsity,
        "proximity": proximity
    }


