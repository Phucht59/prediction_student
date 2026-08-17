"""Inner-development binary metrics."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def binary_classification_metrics(target, score, *, threshold: float = 0.5) -> dict[str, float | int]:
    """Threshold-aware binary diagnostics; F1 values use the positive risk class."""
    target=np.asarray(target, dtype=int);score=np.asarray(score, dtype=float);prediction=(score>=threshold).astype(int)
    if len(np.unique(target)) != 2 or not np.isfinite(score).all():
        raise ValueError("Evaluation requires finite scores and both classes")
    tn,fp,fn,tp=confusion_matrix(target,prediction,labels=[0,1]).ravel()
    return {"pr_auc":float(average_precision_score(target,score)),"roc_auc":float(roc_auc_score(target,score)),"threshold":float(threshold),
            "risk_precision":float(precision_score(target,prediction,pos_label=1,zero_division=0)),"risk_recall":float(recall_score(target,prediction,pos_label=1,zero_division=0)),
            "risk_f1":float(f1_score(target,prediction,pos_label=1,zero_division=0)),"macro_f1":float(f1_score(target,prediction,average="macro",zero_division=0)),
            "micro_f1":float(f1_score(target,prediction,average="micro",zero_division=0)),"accuracy":float(accuracy_score(target,prediction)),
            "balanced_accuracy":float(balanced_accuracy_score(target,prediction)),"specificity":float(tn/(tn+fp)) if tn+fp else 0.0,
            "tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn)}


def binary_metrics(target, score) -> dict[str, float]:
    values=binary_classification_metrics(target,score)
    return {
        "pooled_inner_oof_pr_auc": values["pr_auc"], "pooled_inner_oof_roc_auc": values["roc_auc"],
        "risk_precision": values["risk_precision"], "risk_recall": values["risk_recall"], "risk_f1": values["risk_f1"], "balanced_accuracy": values["balanced_accuracy"],
    }
