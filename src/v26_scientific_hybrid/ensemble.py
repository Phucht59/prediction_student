import numpy as np
import torch
from .evaluate import compute_metrics

def evaluate_ensemble(models, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    all_models_probs = []
    all_targets = []
    
    for model_idx, model in enumerate(models):
        model.to(device)
        model.eval()
        all_probs = []
        with torch.no_grad():
            for seq_x, num_x, cat_x, y_cls, _ in dataloader:
                seq_x, num_x, cat_x = seq_x.to(device), num_x.to(device), cat_x.to(device)
                logits, _ = model(seq_x, num_x, cat_x)
                probs = torch.softmax(logits, dim=1)
                all_probs.extend(probs.cpu().numpy())
                if model_idx == 0:
                    all_targets.extend(y_cls.numpy())
        all_models_probs.append(all_probs)
        
    avg_probs = np.mean(all_models_probs, axis=0)
    ensemble_preds = np.argmax(avg_probs, axis=1)
    metrics = compute_metrics(all_targets, ensemble_preds, y_probs=avg_probs)
    return metrics, np.array(all_targets), ensemble_preds, avg_probs
