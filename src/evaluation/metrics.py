"""Single metric API for Protocol V2 artifacts (Low, Medium, High = 0,1,2)."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support
from src.evaluation.protocol import validate_probability_matrix

METRIC_VERSION = "benchmark_metrics_v2_1"

def top_label_ece(y_true, probabilities, *, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int); p = np.asarray(probabilities, dtype=float)
    validate_probability_matrix(p)
    confidence = p.max(axis=1); predicted = p.argmax(axis=1); total = len(y)
    if total == 0: return 0.0
    ece = 0.0
    for index in range(n_bins):
        lower, upper = index / n_bins, (index + 1) / n_bins
        mask = (confidence >= lower) & ((confidence <= upper) if index == n_bins - 1 else (confidence < upper))
        if mask.any(): ece += (mask.mean() * abs((predicted[mask] == y[mask]).mean() - confidence[mask].mean()))
    return float(ece)

def classification_metrics(y_true, y_pred, probabilities):
    y=np.asarray(y_true,dtype=int); q=np.asarray(y_pred,dtype=int); p=np.asarray(probabilities,dtype=float)
    validate_probability_matrix(p,q)
    precision, recall, f1, support=precision_recall_fscore_support(y,q,labels=[0,1,2],zero_division=0)
    onehot=np.eye(3)[y]
    return {"accuracy":float(accuracy_score(y,q)),"macro_f1":float(f1_score(y,q,average="macro",zero_division=0)),"weighted_f1":float(f1_score(y,q,average="weighted",zero_division=0)),"balanced_accuracy":float(balanced_accuracy_score(y,q)),"quadratic_weighted_kappa":float(cohen_kappa_score(y,q,weights="quadratic")),"ordinal_mae":float(np.abs(y-q).mean()),"brier_score":float(np.mean(np.sum((p-onehot)**2,axis=1))),"pr_auc_macro":float(average_precision_score(onehot,p,average="macro")),"ece_top_label_equal_width_10":top_label_ece(y,p),"confusion_matrix":confusion_matrix(y,q,labels=[0,1,2]).tolist(),"per_class":{str(i):{"precision":float(precision[i]),"recall":float(recall[i]),"f1":float(f1[i]),"support":int(support[i])} for i in range(3)}}
