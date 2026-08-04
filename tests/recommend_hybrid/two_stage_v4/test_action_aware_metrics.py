from __future__ import annotations

import numpy as np

from src.recommend_hybrid.two_stage_v4.metrics import (
    ActionAwareThresholds,
    blended_gate_probability,
    evaluate_action_aware,
    make_decisions,
)


def test_action_derived_probability_separates_all_zero_actions() -> None:
    direct = np.zeros(2, dtype=np.float64)
    action_logits = np.asarray(
        [
            [4.0, -4.0, -4.0, -4.0, -4.0],
            [-4.0, -4.0, -4.0, -4.0, -4.0],
        ]
    )
    mask = np.ones((2, 5), dtype=bool)
    _, action_any, joint = blended_gate_probability(
        direct,
        action_logits,
        mask,
        blend=0.0,
    )
    assert action_any[0] > 0.9
    assert action_any[1] < 0.1
    assert np.allclose(action_any, joint)


def test_stage_specific_thresholds_apply_without_removing_stage() -> None:
    direct = np.asarray([2.0, 2.0, 2.0])
    action_logits = np.asarray(
        [
            [3.0, -3.0, -3.0, -3.0, -3.0],
            [3.0, -3.0, -3.0, -3.0, -3.0],
            [3.0, -3.0, -3.0, -3.0, -3.0],
        ]
    )
    mask = np.ones((3, 5), dtype=bool)
    thresholds = ActionAwareThresholds(
        stage_gate_probability=(0.99, 0.50, 0.50),
        direct_action_blend=0.5,
        minimum_action_probability=0.0,
        minimum_action_margin=0.0,
    )
    decision = make_decisions(
        direct,
        action_logits,
        mask,
        ["EARLY_20", "EARLY_35", "MIDDLE_50"],
        thresholds,
    )
    assert decision.issued.tolist() == [False, True, True]


def test_end_to_end_precision_excludes_abstentions() -> None:
    direct = np.asarray([5.0, -5.0])
    action_logits = np.asarray(
        [
            [5.0, -5.0, -5.0, -5.0, -5.0],
            [-5.0, -5.0, -5.0, -5.0, -5.0],
        ]
    )
    mask = np.ones((2, 5), dtype=bool)
    action_target = np.zeros((2, 5), dtype=np.int8)
    action_target[0, 0] = 1
    metrics = evaluate_action_aware(
        direct_gate_logits=direct,
        action_logits=action_logits,
        action_mask=mask,
        group_target=np.asarray([1, 0]),
        action_target=action_target,
        stages=["EARLY_35", "EARLY_35"],
        thresholds=ActionAwareThresholds(
            stage_gate_probability=(0.5, 0.5, 0.5),
            direct_action_blend=0.5,
            minimum_action_probability=0.5,
            minimum_action_margin=0.0,
        ),
    )
    assert metrics["issued_groups"] == 1
    assert metrics["end_to_end_precision_at_1"] == 1.0
    assert metrics["positive_group_coverage"] == 1.0
