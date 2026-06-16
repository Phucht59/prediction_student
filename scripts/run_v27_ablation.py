import argparse
import csv
import json
import os
import sys
import types
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASETS, MODELS_DIR, DEFAULT_SEED, ensure_dirs
from src.data_pipeline import (
    load_splits,
    apply_feature_engineering,
    get_sequence_columns,
    DataPreprocessor,
    FeatureSelector,
    StudentDataset,
)
from src.train_v27_pipeline import create_model_v27, train_model_v27, validate_epoch_v27
from src.losses_v27 import ClassBalancedFocalLoss, OrdinalLoss, JointHybridLoss
from src.utils import set_seed, setup_logger

logger = setup_logger("run_v27_ablation")


# Custom preprocessor forcing standard SMOTE instead of SMOTENC
class SMOTEPreprocessor(DataPreprocessor):
    def apply_oversampling(self, df: pd.DataFrame):
        from imblearn.over_sampling import SMOTE
        if self.oversample_method == "none":
            return df
        df = df.copy()
        X = df.drop(columns=[self.target_col])
        y_encoded = df[self.target_col]
        
        class_counts = pd.Series(y_encoded).value_counts()
        majority_count = class_counts.max()
        effective_k_neighbors = min(
            self.resampling_k_neighbors,
            max(1, int(class_counts.min()) - 1),
        )
        strategy = {}
        for cls, count in class_counts.items():
            if count == majority_count:
                strategy[cls] = count
            else:
                target = int(majority_count * self.smote_ratio)
                strategy[cls] = max(count, target)
        
        # Force standard SMOTE
        sampler = SMOTE(
            sampling_strategy=strategy,
            random_state=42,
            k_neighbors=effective_k_neighbors,
        )
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y_encoded)
            X = pd.DataFrame(X_resampled, columns=X.columns)
            # Round categorical columns so they are valid for embedding lookups
            remaining_cat_cols = [col for col in self.categorical_cols if col in X.columns]
            for col in remaining_cat_cols:
                X[col] = X[col].round().astype(int)
            y_encoded = y_resampled
        except Exception as e:
            logger.warning(f"Standard SMOTE failed. Error: {e}")
        
        df_out = X.copy()
        df_out[self.target_col] = y_encoded
        return df_out


# Custom Modules for Ablation
class ConcatenationFusion(nn.Module):
    def __init__(self, seq_dim, ctx_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(seq_dim + ctx_dim, out_dim)

    def forward(self, seq_vec, ctx_vec):
        concat = torch.cat([seq_vec, ctx_vec], dim=1)
        return self.proj(concat)


class MeanPooling1D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = torch.mean(sequence, dim=1)
        batch_size, seq_len, _ = sequence.shape
        weights = torch.ones(batch_size, seq_len, 1, device=sequence.device) / seq_len
        return pooled, weights


# Patched forward methods
def context_only_forward(self, seq_x, num_x, cat_x):
    context = self._prepare_context(
        num_x=num_x,
        cat_x=cat_x,
        batch_size=seq_x.shape[0],
        device=seq_x.device,
    )
    context_vector = self.context_mlp(context)
    
    # Mask sequence_vector to zeros
    sequence_output_dim = self.sequence_bilstm.hidden_size * 2
    sequence_vector = torch.zeros(seq_x.shape[0], sequence_output_dim, device=seq_x.device)
    
    fused = self.fusion(sequence_vector, context_vector)
    
    class_logits = self.class_head(fused)
    ordinal_logits = self.ordinal_head(fused)
    reg_logits = self.reg_head(fused).squeeze(-1)
    
    return class_logits, ordinal_logits, reg_logits


def sequence_only_forward(self, seq_x, num_x, cat_x):
    sequence = seq_x.float().transpose(1, 2)
    sequence = self.sequence_cnn(sequence).transpose(1, 2)
    sequence, _ = self.sequence_bilstm(sequence)
    sequence_vector, _ = self.sequence_pool(sequence)
    
    # Mask context_vector to zeros
    context_hidden_dim = self.context_mlp[0].out_features
    context_vector = torch.zeros(seq_x.shape[0], context_hidden_dim, device=seq_x.device)
    
    fused = self.fusion(sequence_vector, context_vector)
    
    class_logits = self.class_head(fused)
    ordinal_logits = self.ordinal_head(fused)
    reg_logits = self.reg_head(fused).squeeze(-1)
    
    return class_logits, ordinal_logits, reg_logits


def evaluate_variant(variant_name: str, best_params: dict, train_pool: pd.DataFrame, spec):
    logger.info(f"--- Evaluating variant: {variant_name} ---")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=DEFAULT_SEED)
    labels = train_pool[spec.target_col].astype(int).to_numpy()
    
    fold_f1s = []
    fold_recalls_low = []
    fold_accuracies = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(best_params.get("batch_size", 32))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_pool, labels)):
        set_seed(DEFAULT_SEED + fold)
        
        train_fold = train_pool.iloc[train_idx].copy()
        val_fold = train_pool.iloc[val_idx].copy()
        
        train_fold = apply_feature_engineering(train_fold, spec.kind)
        val_fold = apply_feature_engineering(val_fold, spec.kind)
        
        # 1. Handle Preprocessor configuration based on variant
        oversample_method = best_params.get("oversample_method", "none")
        if oversample_method == "none" or oversample_method == "":
            # For ablation purposes, let's default the Base model to use SMOTENC to show the impact of SMOTE/no oversampling
            oversample_method = "smotenc"
            
        if variant_name == "No oversampling":
            oversample_method = "none"
            
        if variant_name == "Standard SMOTE":
            preprocessor = SMOTEPreprocessor(
                target_col=spec.target_col,
                oversample_method="smote",
                smote_ratio=best_params.get("smote_ratio", 1.0),
                resampling_k_neighbors=5,
            )
        else:
            preprocessor = DataPreprocessor(
                target_col=spec.target_col,
                oversample_method=oversample_method,
                smote_ratio=best_params.get("smote_ratio", 1.0),
                resampling_k_neighbors=5,
            )
            
        train_prep = preprocessor.fit_transform(train_fold, apply_oversampling=False)
        val_prep = preprocessor.transform(val_fold)
        
        # 2. Feature Selector
        sequence_columns = get_sequence_columns(spec.kind)
        selector = FeatureSelector(
            target_col=spec.target_col,
            use_feature_selection=True,
            required_features=sequence_columns,
        )
        train_selected = selector.fit_transform(
            train_prep,
            preprocessor.numerical_cols,
            preprocessor.categorical_cols,
        )
        val_selected = selector.transform(val_prep)
        
        train_resampled = preprocessor.apply_oversampling(train_selected)
        
        train_ds = StudentDataset(
            train_resampled,
            spec.kind,
            spec.target_col,
            preprocessor.numerical_cols,
            preprocessor.categorical_cols,
        )
        val_ds = StudentDataset(
            val_selected,
            spec.kind,
            spec.target_col,
            preprocessor.numerical_cols,
            preprocessor.categorical_cols,
        )
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        num_numerical = len(train_ds.num_cols)
        cat_cardinalities = [
            len(preprocessor.label_encoders[col].classes_)
            for col in train_ds.cat_cols
        ]
        
        # Create model
        model = create_model_v27(spec.kind, best_params, num_numerical, cat_cardinalities).to(device)
        
        # Apply structural modifications based on variant
        if variant_name == "Context-only":
            model.forward = types.MethodType(context_only_forward, model)
        elif variant_name == "Sequence-only":
            model.forward = types.MethodType(sequence_only_forward, model)
        elif variant_name == "Concatenation fusion":
            sequence_output_dim = model.sequence_bilstm.hidden_size * 2
            context_hidden_dim = model.context_mlp[0].out_features
            fusion_hidden_dim = model.fusion.proj_seq.out_features
            model.fusion = ConcatenationFusion(sequence_output_dim, context_hidden_dim, fusion_hidden_dim).to(device)
        elif variant_name == "No Attention Pooling":
            model.sequence_pool = MeanPooling1D().to(device)
            
        # 3. Setup Loss configuration based on variant
        original_train_labels = train_fold[spec.target_col].astype(int).to_numpy()
        class_counts = np.bincount(original_train_labels, minlength=3)
        
        if variant_name == "No Class-Balanced Focal Loss":
            class_loss_fn = nn.CrossEntropyLoss()
        else:
            class_loss_fn = ClassBalancedFocalLoss(class_counts=class_counts, beta=0.99, gamma=2.0)
            
        ordinal_loss_fn = OrdinalLoss()
        regression_loss_fn = nn.MSELoss()
        
        w_ord = 0.0 if variant_name == "No Ordinal Auxiliary Head" else 1.0
        w_reg = 0.0 if (variant_name == "No Regression Auxiliary Head" or spec.kind == "xapi") else 1.0
        
        criterion = JointHybridLoss(
            class_loss_fn=class_loss_fn,
            ordinal_loss_fn=ordinal_loss_fn,
            regression_loss_fn=regression_loss_fn,
            w_class=1.0,
            w_ord=w_ord,
            w_reg=w_reg
        )
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=best_params.get("learning_rate", 0.001),
            weight_decay=best_params.get("weight_decay", 0.0001)
        )
        
        class TrainingConfigV27:
            max_epochs = 40
            patience = 10
            scheduler_patience = 4
            scheduler_factor = 0.5
            
        config = TrainingConfigV27()
        
        model, history, best_score = train_model_v27(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            config,
            device
        )
        
        # Evaluate on validation fold
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                seq_x, num_x, cat_x, labels, _, _ = batch
                # Make sure we use the patched forward if monkey-patched
                outputs = model(seq_x.to(device), num_x.to(device), cat_x.to(device))
                preds = torch.argmax(outputs[0], dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        f1_m = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        
        recalls = recall_score(all_labels, all_preds, average=None, zero_division=0)
        rec_low = float(recalls[0]) if len(recalls) > 0 else 0.0
        
        fold_f1s.append(f1_m)
        fold_accuracies.append(acc)
        fold_recalls_low.append(rec_low)
        
    avg_f1 = np.mean(fold_f1s)
    avg_rec_low = np.mean(fold_recalls_low)
    avg_acc = np.mean(fold_accuracies)
    
    logger.info(f"Variant '{variant_name}' average results - F1: {avg_f1:.4f}, Recall-Low: {avg_rec_low:.4f}, Accuracy: {avg_acc:.4f}")
    return avg_f1, avg_rec_low, avg_acc


def main():
    ensure_dirs()
    
    dataset_name = "student-mat"
    spec = DATASETS[dataset_name]
    
    # Load best params
    best_params_path = MODELS_DIR / f"{dataset_name}_3class_best_params.json"
    if not best_params_path.exists():
        logger.error(f"Best parameters file not found at {best_params_path}. Please check.")
        sys.exit(1)
        
    with open(best_params_path, "r", encoding="utf-8") as f:
        best_params = json.load(f)
        
    # Load splits
    train_pool, _ = load_splits(dataset_name, "3class")
    logger.info(f"Loaded student-mat train pool size: {len(train_pool)}")
    
    variants = [
        "Full V27 Model (Base)",
        "Context-only",
        "Sequence-only",
        "Concatenation fusion",
        "No Attention Pooling",
        "No Ordinal Auxiliary Head",
        "No Regression Auxiliary Head",
        "No oversampling",
        "Standard SMOTE",
        "No Class-Balanced Focal Loss"
    ]
    
    results = []
    
    for var in variants:
        f1_m, rec_l, acc = evaluate_variant(var, best_params, train_pool, spec)
        results.append({
            "variant": var,
            "f1_macro": float(f1_m),
            "recall_low": float(rec_l),
            "accuracy": float(acc)
        })
        
    # Save results to CSV
    output_dir = Path("outputs/v27")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "ablation_results.csv"
    
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "f1_macro", "recall_low", "accuracy"])
        writer.writeheader()
        for res in results:
            writer.writerow(res)
            
    logger.info(f"Ablation study completed. Results saved to {out_file}")


if __name__ == "__main__":
    main()
