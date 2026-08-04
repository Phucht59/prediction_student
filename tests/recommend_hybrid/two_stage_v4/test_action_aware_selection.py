from __future__ import annotations

import numpy as np

from src.recommend_hybrid.two_stage_v4.selection import (
    select_action_aware_thresholds,
)


def test_action_aware_selection_can_use_candidate_evidence_to_reject_negatives() -> None:
    stages = np.asarray(
        [
            "EARLY_20",
            "EARLY_20",
            "EARLY_35",
            "EARLY_35",
            "MIDDLE_50",
            "MIDDLE_50",
        ],
        dtype=object,
    )
    group_target = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.float32)
    action_target = np.zeros((6, 5), dtype=np.float32)
    action_target[[0, 2, 4], [1, 2, 4]] = 1.0
    action_logits = np.full((6, 5), -6.0, dtype=np.float32)
    action_logits[0, 1] = 6.0
    action_logits[2, 2] = 6.0
    action_logits[4, 4] = 6.0
    direct_logits = np.zeros(6, dtype=np.float32)
    mask = np.ones((6, 5), dtype=bool)

    thresholds, metrics, audit = select_action_aware_thresholds(
        direct_gate_logits=direct_logits,
        action_logits=action_logits,
        action_mask=mask,
        group_target=group_target,
        action_target=action_target,
        stages=stages,
        blend_weights=[0.0, 1.0],
        action_probability_grid=[0.0, 0.5],
        margin_grid=[0.0],
        stage_coverage_floor={
            "EARLY_20": 0.5,
            "EARLY_35": 0.5,
            "MIDDLE_50": 0.5,
        },
        minimum_global_coverage=0.5,
        target_precision=0.8,
        action_specific_minimum_support=1,
        stage_quantile_count=5,
    )
    assert thresholds.direct_action_blend == 0.0
    assert metrics["positive_group_coverage"] >= 0.5
    assert metrics["end_to_end_precision_at_1"] == 1.0
    assert metrics["selection_target_met"] is True
    assert audit["target_candidate_count"] > 0


def test_selection_does_not_claim_target_when_coverage_is_zero() -> None:
    stages = np.asarray(["EARLY_20", "EARLY_35", "MIDDLE_50"], dtype=object)
    group_target = np.ones(3, dtype=np.float32)
    action_target = np.zeros((3, 5), dtype=np.float32)
    action_target[:, 0] = 1.0
    action_logits = np.full((3, 5), -6.0, dtype=np.float32)
    mask = np.ones((3, 5), dtype=bool)
    thresholds, metrics, _ = select_action_aware_thresholds(
        direct_gate_logits=np.full(3, -6.0, dtype=np.float32),
        action_logits=action_logits,
        action_mask=mask,
        group_target=group_target,
        action_target=action_target,
        stages=stages,
        blend_weights=[1.0],
        action_probability_grid=[0.9],
        margin_grid=[0.9],
        stage_coverage_floor={stage: 0.5 for stage in stages},
        minimum_global_coverage=0.5,
        target_precision=0.8,
        action_specific_minimum_support=1,
        stage_quantile_count=5,
    )
    assert metrics["positive_group_coverage"] == 0.0
    assert metrics["selection_target_met"] is False
    assert thresholds.minimum_action_probability == 0.9
