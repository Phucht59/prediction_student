from __future__ import annotations

import numpy as np
import pytest

from src.studies.v5_1.common.statistics import paired_group_bootstrap, practical_verdict


def test_paired_bootstrap_is_deterministic_and_group_preserving() -> None:
    target = np.array([0, 0, 1, 1, 2, 2])
    better = target.copy()
    worse = np.array([0, 1, 1, 0, 2, 1])
    groups = np.array([10, 10, 20, 20, 30, 30])
    first = paired_group_bootstrap(target, better, worse, groups, replicates=200, seed=42)
    second = paired_group_bootstrap(target, better, worse, groups, replicates=200, seed=42)
    assert first == second
    assert first["groups"] == 3
    assert first["delta_a_minus_b"] > 0


def test_bootstrap_rejects_unaligned_inputs() -> None:
    with pytest.raises(ValueError, match="aligned"):
        paired_group_bootstrap(np.array([0, 1]), np.array([0]), np.array([0, 1]), np.array([1, 2]))


def test_practical_verdict_requires_effect_and_interval() -> None:
    assert (
        practical_verdict(
            {"delta_a_minus_b": 0.02, "ci95_lower": 0.01, "ci95_upper": 0.03}
        )
        == "A_SUPERIOR"
    )
    assert (
        practical_verdict(
            {"delta_a_minus_b": 0.002, "ci95_lower": -0.01, "ci95_upper": 0.02}
        )
        == "PRACTICAL_TIE_OR_UNCERTAIN"
    )
