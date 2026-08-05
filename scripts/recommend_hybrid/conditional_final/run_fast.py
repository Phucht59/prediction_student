"""Run the conditional final evaluation with precision-only controls."""
from __future__ import annotations

import itertools

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
    identity = tuple(range(base.ACTION_COUNT))
    alternatives = np.asarray(
        [
            permutation
            for permutation in itertools.permutations(range(base.ACTION_COUNT))
            if permutation != identity
        ],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    sampled = rng.integers(0, len(alternatives), size=repetitions)
    for index, permutation_index in enumerate(sampled):
        permutation = alternatives[permutation_index]
        values[index] = _precision_only(
            scores[:, permutation],
            targets,
            valid[:, permutation],
        )
    return {
        "repetitions": int(repetitions),
        "seed": int(seed),
        "identity_excluded": True,
        "unique_non_identity_permutations": int(len(alternatives)),
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
