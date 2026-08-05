"""Run the conditional final evaluation with precision-only controls."""
from __future__ import annotations

import numpy as np

import evaluate_release as base


def _precision_only(
    scores: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
) -> float:
    masked = np.where(valid, scores, -np.inf)
    top = np.argmax(masked, axis=1)
    row = np.arange(len(top))
    return float((targets[row, top] > 0).mean())


def _random_ranking_control(
    targets: np.ndarray,
    valid: np.ndarray,
    observed: float,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        values[index] = _precision_only(rng.random(valid.shape), targets, valid)
    return {
        "repetitions": int(repetitions),
        "seed": int(seed),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
        "p_value": float(
            (1 + np.count_nonzero(values >= observed)) / (repetitions + 1)
        ),
    }


def _action_identity_control(
    scores: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
    observed: float,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        permutation = rng.permutation(base.ACTION_COUNT)
        values[index] = _precision_only(
            scores[:, permutation],
            targets,
            valid[:, permutation],
        )
    return {
        "repetitions": int(repetitions),
        "seed": int(seed),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
        "p_value": float(
            (1 + np.count_nonzero(values >= observed)) / (repetitions + 1)
        ),
    }


base._random_ranking_control = _random_ranking_control
base._action_identity_control = _action_identity_control


if __name__ == "__main__":
    base.main()
