from __future__ import annotations

import numpy as np

from src.recommend_hybrid.two_stage_v3.metrics import (
    TwoStageThresholds,
    derive_action_thresholds,
    evaluate_two_stage,
)
from src.recommend_hybrid.two_stage_v3.selection import select_thresholds_staged


def synthetic() -> tuple[np.ndarray, ...]:
    gate_logits = np.array([4.0, 3.0, 2.0, -2.0, -3.0, 1.0], dtype=np.float32)
    action_logits = np.array(
        [
            [0.0, 5.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 5.0, 0.0, 0.0],
            [5.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 5.0, 0.0, 0.0],
            [5.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mask = np.ones((6, 5), dtype=bool)
    group_target = np.array([1, 1, 0, 0, 0, 1], dtype=np.float32)
    action_target = np.zeros((6, 5), dtype=np.float32)
    action_target[0, 1] = 1.0
    action_target[1, 2] = 1.0
    action_target[5, 1] = 1.0
    stages = np.array(
        ["EARLY_20", "EARLY_35", "MIDDLE_50", "EARLY_20", "EARLY_35", "MIDDLE_50"],
        dtype=object,
    )
    return gate_logits, action_logits, mask, group_target, action_target, stages


def test_metric_decomposition_matches_product() -> None:
    gate, action, mask, group_target, action_target, stages = synthetic()
    metrics = evaluate_two_stage(
        gate_logits=gate,
        action_logits=action,
        action_mask=mask,
        group_target=group_target,
        action_target=action_target,
        thresholds=TwoStageThresholds(0.5, 0.0, 0.0),
        stages=stages,
    )
    product = (
        float(metrics["stage_a_precision"])
        * float(metrics["stage_b_conditional_precision_at_1"])
    )
    assert abs(product - float(metrics["end_to_end_precision_at_1"])) < 1.0e-12


def test_action_specific_threshold_can_suppress_unreliable_action() -> None:
    gate, action, mask, group_target, action_target, _ = synthetic()
    thresholds = derive_action_thresholds(
        gate_logits=gate,
        action_logits=action,
        action_mask=mask,
        group_target=group_target,
        action_target=action_target,
        base_thresholds=TwoStageThresholds(0.0, 0.0, 0.0),
        target_precision=0.80,
        minimum_support=1,
    )
    assert thresholds.action_probability_by_id[0] == 1.0
    assert thresholds.action_probability_by_id[1] < 1.0


def test_staged_selection_respects_coverage_floor() -> None:
    gate, action, mask, group_target, action_target, stages = synthetic()
    thresholds, metrics, audit = select_thresholds_staged(
        gate_logits=gate,
        action_logits=action,
        action_mask=mask,
        group_target=group_target,
        action_target=action_target,
        stages=stages,
        minimum_coverage=0.50,
        target_precision=0.80,
        action_probability_grid=[0.0, 0.5],
        margin_grid=[0.0, 0.1],
        action_specific_minimum_support=1,
    )
    assert isinstance(thresholds, TwoStageThresholds)
    assert float(metrics["positive_group_coverage"]) >= 0.50
    assert int(audit["global_candidate_count"]) > 0
