from __future__ import annotations

import numpy as np

from src.recommend_hybrid.causal.imbalance import (
    IMBALANCE_MODES,
    run_frozen_embedding_imbalance_study,
    select_validation_threshold,
)


def _synthetic_split(seed: int, count: int, positive_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    target = np.zeros(count, dtype=np.int8)
    target[: max(2, int(count * positive_fraction))] = 1
    rng.shuffle(target)
    features = rng.normal(size=(count, 8))
    features[:, 0] += 1.2 * target
    features[:, 1] -= 0.5 * target
    return features, target


def test_validation_threshold_is_deterministic() -> None:
    target = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.4, 0.6, 0.9])
    first = select_validation_threshold(target, probability)
    second = select_validation_threshold(target, probability)
    assert first == second


def test_all_imbalance_modes_leave_checkpoint_frozen() -> None:
    train_x, train_y = _synthetic_split(1, 160, 0.15)
    validation_x, validation_y = _synthetic_split(2, 80, 0.20)
    test_x, test_y = _synthetic_split(3, 80, 0.20)
    result = run_frozen_embedding_imbalance_study(
        train_features=train_x,
        train_target=train_y,
        validation_features=validation_x,
        validation_target=validation_y,
        test_features=test_x,
        test_target=test_y,
        random_state=42,
    )
    assert result["modes"] == list(IMBALANCE_MODES)
    rows = result["results"]
    assert [row["mode"] for row in rows] == list(IMBALANCE_MODES)
    assert all(row["resampling_scope"] == "TRAIN_EMBEDDINGS_ONLY" for row in rows)
    assert all(row["canonical_checkpoint_replaced"] is False for row in rows)
    assert rows[0]["fitted_train_count"] == len(train_y)
    assert rows[1]["fitted_train_count"] == len(train_y)
    assert rows[2]["fitted_train_count"] >= len(train_y)
    assert rows[3]["fitted_train_count"] >= len(train_y)
    assert all(row["metrics"]["confusion_matrix"] for row in rows)
