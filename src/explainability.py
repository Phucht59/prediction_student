"""Optional permutation importance for the sequence-only research model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from src.utils import setup_logger

logger = setup_logger("explainability")


def calculate_permutation_importance(model, val_loader, device, numerical_feature_names, categorical_feature_names):
    """Measure Macro-F1 change for active context features only.

    The frozen CNN-BiLSTM has no context branch, so this returns an empty table
    for the final architecture rather than creating a separate recommender.
    """
    if not numerical_feature_names and not categorical_feature_names:
        return pd.DataFrame(columns=["Feature", "Importance"])
    model.eval()
    sequences, numerical, categorical, labels = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            seq_x, num_x, cat_x, batch_labels, _ = batch[:5]
            sequences.append(seq_x.cpu())
            numerical.append(num_x.cpu())
            categorical.append(cat_x.cpu())
            labels.extend(batch_labels.numpy())
    full_seq = torch.cat(sequences, dim=0)
    full_num = torch.cat(numerical, dim=0)
    full_cat = torch.cat(categorical, dim=0)
    if full_num.shape[1] == 0 and full_cat.shape[1] == 0:
        return pd.DataFrame(columns=["Feature", "Importance"])

    def score(num_values, cat_values):
        predictions = []
        with torch.no_grad():
            for start in range(0, len(labels), 32):
                logits = model(full_seq[start:start + 32].to(device), num_values[start:start + 32].to(device), cat_values[start:start + 32].to(device))
                predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        return f1_score(labels, predictions, average="macro", zero_division=0)

    baseline = score(full_num, full_cat)
    values = []
    for column, name in enumerate(numerical_feature_names):
        shuffled = full_num.clone()
        shuffled[:, column] = shuffled[torch.randperm(len(shuffled)), column]
        values.append((name, baseline - score(shuffled, full_cat)))
    for column, name in enumerate(categorical_feature_names):
        shuffled = full_cat.clone()
        shuffled[:, column] = shuffled[torch.randperm(len(shuffled)), column]
        values.append((name, baseline - score(full_num, shuffled)))
    return pd.DataFrame(values, columns=["Feature", "Importance"]).sort_values("Importance", ascending=False, ignore_index=True)


def explain_model(model, val_loader, device, numerical_feature_names, categorical_feature_names, out_path):
    importance = calculate_permutation_importance(model, val_loader, device, numerical_feature_names, categorical_feature_names)
    importance.to_csv(out_path, index=False)
    return importance
