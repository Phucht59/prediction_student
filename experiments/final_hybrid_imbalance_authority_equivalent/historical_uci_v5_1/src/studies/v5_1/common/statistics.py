from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(f1_score(target, prediction, average="macro", zero_division=0))


def paired_group_bootstrap(
    target: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int = 5000,
    seed: int = 3407,
    metric: Callable[[np.ndarray, np.ndarray], float] = macro_f1,
) -> dict[str, float | int]:
    target = np.asarray(target)
    prediction_a = np.asarray(prediction_a)
    prediction_b = np.asarray(prediction_b)
    groups = np.asarray(groups)
    if not (len(target) == len(prediction_a) == len(prediction_b) == len(groups)):
        raise ValueError("Paired bootstrap inputs must be aligned")
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_rows = [np.flatnonzero(inverse == index) for index in range(len(unique_groups))]
    observed_a = metric(target, prediction_a)
    observed_b = metric(target, prediction_b)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_groups = rng.integers(0, len(unique_groups), size=len(unique_groups))
        indices = np.concatenate([group_rows[index] for index in sampled_groups])
        deltas[replicate] = metric(target[indices], prediction_a[indices]) - metric(
            target[indices], prediction_b[indices]
        )
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    return {
        "records": int(len(target)),
        "groups": int(len(unique_groups)),
        "replicates": int(replicates),
        "metric_a": observed_a,
        "metric_b": observed_b,
        "delta_a_minus_b": observed_a - observed_b,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "probability_delta_positive": float((deltas > 0).mean()),
    }


def practical_verdict(comparison: dict[str, float | int], margin: float = 0.005) -> str:
    delta = float(comparison["delta_a_minus_b"])
    lower = float(comparison["ci95_lower"])
    upper = float(comparison["ci95_upper"])
    if lower > 0 and delta >= margin:
        return "A_SUPERIOR"
    if upper < 0 and delta <= -margin:
        return "B_SUPERIOR"
    return "PRACTICAL_TIE_OR_UNCERTAIN"


__all__ = ["macro_f1", "paired_group_bootstrap", "practical_verdict"]

