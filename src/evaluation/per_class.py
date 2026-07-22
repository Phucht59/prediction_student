"""Per-class calculations over a confusion matrix."""

from __future__ import annotations

import numpy as np


def per_class_from_confusion(
    matrix: list[list[int]], labels: list[str]
) -> list[dict[str, object]]:
    cm = np.asarray(matrix, dtype=float)
    result = []
    for index, label in enumerate(labels):
        precision = (
            float(cm[index, index] / cm[:, index].sum()) if cm[:, index].sum() else 0.0
        )
        recall = float(cm[index, index] / cm[index].sum()) if cm[index].sum() else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        result.append(
            {
                "class": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(cm[index].sum()),
            }
        )
    return result
