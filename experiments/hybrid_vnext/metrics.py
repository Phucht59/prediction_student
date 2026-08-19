"""Inner-only binary metrics including ECE. Threshold from STOP only."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


THRESHOLD_GRID = tuple(round(value / 100, 2) for value in range(5, 96))


def expected_calibration_error(target, score, n_bins: int = 15) -> float:
    target = np.asarray(target, dtype=float)
    score = np.asarray(score, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        right = bins[i + 1] if i < n_bins - 1 else bins[i + 1] + 1e-12
        mask = (score >= bins[i]) & (score < right)
        if mask.any():
            ece += float(mask.mean() * abs(score[mask].mean() - target[mask].mean()))
    return float(ece)


def binary_metrics(target, score, *, threshold: float = 0.5) -> dict[str, float]:
    target = np.asarray(target, dtype=int)
    score = np.asarray(score, dtype=float)
    if target.size == 0:
        raise ValueError("empty target")
    if not np.isfinite(np.asarray(score)).all():
        raise ValueError("nonfinite scores")
    if len(np.unique(target)) < 2:
        raise ValueError("single-class partition")
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(target, score)),
        "roc_auc": float(roc_auc_score(target, score)),
        "risk_precision": float(precision_score(target, pred, pos_label=1, zero_division=0)),
        "risk_recall": float(recall_score(target, pred, pos_label=1, zero_division=0)),
        "risk_f1": float(f1_score(target, pred, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(target, pred)),
        "accuracy": float(accuracy_score(target, pred)),
        "ece": expected_calibration_error(target, score),
        "brier": float(np.mean((score - target) ** 2)),
        "nll": float(-np.mean(target * np.log(np.clip(score, 1e-6, 1)) + (1 - target) * np.log(np.clip(1 - score, 1e-6, 1)))),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def select_stop_threshold(target, score) -> float:
    ranked = []
    for threshold in THRESHOLD_GRID:
        metrics = binary_metrics(target, score, threshold=threshold)
        ranked.append((metrics["risk_f1"], metrics["risk_recall"], -abs(threshold - 0.5), threshold))
    return float(max(ranked)[-1])
