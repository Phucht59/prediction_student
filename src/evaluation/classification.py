"""Classification calculations over immutable predictions."""

from __future__ import annotations

import numpy as np


def metrics_from_confusion(matrix: list[list[int]]) -> dict[str, float]:
    cm = np.asarray(matrix, dtype=float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    diagonal = np.diag(cm)
    precision = np.divide(
        diagonal, predicted, out=np.zeros_like(diagonal), where=predicted != 0
    )
    recall = np.divide(
        diagonal, support, out=np.zeros_like(diagonal), where=support != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(diagonal),
        where=(precision + recall) != 0,
    )
    total = float(cm.sum())
    return {
        "accuracy": float(diagonal.sum() / total),
        "balanced_accuracy": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
    }
