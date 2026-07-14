"""Canonical metric API for Protocol V2 artifacts.

The immutable class order is Low, Medium, High = 0, 1, 2.  Per-class
precision/recall/F1 are reported as zero when a class has no predicted or true
samples; support is always the true-label count.  This explicit behaviour
keeps fold-level output deterministic when a small fold omits a class.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support
from src.evaluation.protocol import validate_probability_matrix

METRIC_VERSION = "benchmark_metrics_v2_1_1"
CLASS_ORDER = (0, 1, 2)

def _validated_labels(values, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw.astype(float)).all():
        raise ValueError(f"{name} must be a finite one-dimensional numeric label vector.")
    if not np.equal(raw.astype(float), raw.astype(int)).all():
        raise ValueError(f"{name} must contain integer class labels.")
    labels = raw.astype(int)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional label vector.")
    if np.any(~np.isin(labels, CLASS_ORDER)):
        raise ValueError(f"{name} must contain only Low/Medium/High encoded as 0/1/2.")
    return labels

def top_label_ece(y_true, probabilities, *, n_bins: int = 10) -> float:
    if not isinstance(n_bins, (int, np.integer)) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")
    y = _validated_labels(y_true, name="y_true"); p = np.asarray(probabilities, dtype=float)
    validate_probability_matrix(p)
    if len(y) != len(p):
        raise ValueError("y_true and probabilities must contain the same number of rows.")
    confidence = p.max(axis=1); predicted = p.argmax(axis=1); total = len(y)
    if total == 0: return 0.0
    ece = 0.0
    for index in range(n_bins):
        lower, upper = index / n_bins, (index + 1) / n_bins
        mask = (confidence >= lower) & ((confidence <= upper) if index == n_bins - 1 else (confidence < upper))
        if mask.any(): ece += (mask.mean() * abs((predicted[mask] == y[mask]).mean() - confidence[mask].mean()))
    return float(ece)

def classification_metrics(y_true, y_pred, probabilities):
    y=_validated_labels(y_true,name="y_true"); q=_validated_labels(y_pred,name="predicted_labels"); p=np.asarray(probabilities,dtype=float)
    if len(y) != len(q) or len(y) != len(p):
        raise ValueError("y_true, predicted_labels and probabilities must have equal length.")
    validate_probability_matrix(p,q)
    precision, recall, f1, support=precision_recall_fscore_support(y,q,labels=CLASS_ORDER,zero_division=0)
    onehot=np.eye(3)[y]
    return {"accuracy":float(accuracy_score(y,q)),"macro_f1":float(f1_score(y,q,average="macro",zero_division=0)),"weighted_f1":float(f1_score(y,q,average="weighted",zero_division=0)),"balanced_accuracy":float(balanced_accuracy_score(y,q)),"quadratic_weighted_kappa":float(cohen_kappa_score(y,q,weights="quadratic")),"ordinal_mae":float(np.abs(y-q).mean()),"brier_score":float(np.mean(np.sum((p-onehot)**2,axis=1))),"pr_auc_macro":float(average_precision_score(onehot,p,average="macro")),"ece_top_label_equal_width_10":top_label_ece(y,p),"confusion_matrix":confusion_matrix(y,q,labels=CLASS_ORDER).tolist(),"per_class":{str(i):{"precision":float(precision[i]),"recall":float(recall[i]),"f1":float(f1[i]),"support":int(support[i])} for i in CLASS_ORDER}}
