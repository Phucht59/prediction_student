"""Metrics for the imbalance experiment. Same family as production prediction."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

THRESHOLD_GRID = tuple(round(v / 100, 2) for v in range(5, 96))


def select_stop_threshold(target: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(target, dtype=int)
    p = np.asarray(score, dtype=float)
    ranked = []
    for threshold in THRESHOLD_GRID:
        pred = (p >= threshold).astype(int)
        ranked.append(
            (
                float(f1_score(y, pred, pos_label=1, zero_division=0)),
                float(recall_score(y, pred, pos_label=1, zero_division=0)),
                -abs(threshold - 0.5),
                threshold,
            )
        )
    return float(max(ranked)[-1])


def metrics(target: np.ndarray, score: np.ndarray, *, threshold: float) -> dict[str, float]:
    y = np.asarray(target, dtype=int)
    p = np.asarray(score, dtype=float)
    pred = (p >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "minority_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "threshold": float(threshold),
    }


__all__ = ["metrics", "select_stop_threshold"]
