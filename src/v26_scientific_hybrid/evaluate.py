import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, average_precision_score

def compute_metrics(y_true_cls, y_pred_cls, y_true_ord=None, y_pred_ord=None, y_probs=None):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true_cls, y_pred_cls)
    metrics['precision_macro'] = precision_score(y_true_cls, y_pred_cls, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true_cls, y_pred_cls, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true_cls, y_pred_cls, average='macro', zero_division=0)
    
    if y_true_ord is not None and y_pred_ord is not None:
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true_ord, y_pred_ord))
        metrics['r2'] = r2_score(y_true_ord, y_pred_ord)
        
    if y_probs is not None:
        n_classes = y_probs.shape[1]
        try:
            metrics['pr_auc'] = average_precision_score(
                np.eye(n_classes)[y_true_cls], y_probs, average='macro'
            )
        except Exception:
            pass
            
    return metrics

def evaluate_model(model, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    model.eval()
    
    all_preds, all_targets_cls, all_preds_ord, all_targets_ord, all_probs = [], [], [], [], []
    with torch.no_grad():
        for seq_x, num_x, cat_x, y_cls, y_ord in dataloader:
            seq_x, num_x, cat_x = seq_x.to(device), num_x.to(device), cat_x.to(device)
            logits, ord_score = model(seq_x, num_x, cat_x)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets_cls.extend(y_cls.numpy())
            all_preds_ord.extend(ord_score.cpu().numpy())
            all_targets_ord.extend(y_ord.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    metrics = compute_metrics(all_targets_cls, all_preds, all_targets_ord, all_preds_ord, np.array(all_probs))
    return metrics, np.array(all_targets_cls), np.array(all_preds), np.array(all_probs)
