"""Deterministic budget metrics from frozen risk probabilities."""

from __future__ import annotations

import math

import numpy as np


def top_k_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    record_ids: np.ndarray,
    fraction: float,
) -> dict[str, float]:
    budget = max(1, math.ceil(len(labels) * fraction))
    order = np.lexsort((record_ids.astype(str), -probabilities))[:budget]
    selected = labels[order].astype(float)
    precision = float(selected.mean())
    recall = float(selected.sum() / labels.sum()) if labels.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    discounts = 1.0 / np.log2(np.arange(2, budget + 2))
    dcg = float((selected * discounts).sum())
    ideal = float(discounts[: min(int(labels.sum()), budget)].sum())
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ndcg": dcg / ideal if ideal else 0.0,
    }
