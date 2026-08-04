"""Efficient inner-OOF threshold selection for Two-Stage V3."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .metrics import (
    TwoStageThresholds,
    derive_action_thresholds,
    evaluate_two_stage,
    make_decisions,
)


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _fast_metrics(
    *,
    gate_logits: np.ndarray,
    action_logits: np.ndarray,
    action_mask: np.ndarray,
    group_target: np.ndarray,
    action_target: np.ndarray,
    thresholds: TwoStageThresholds,
    stages: np.ndarray,
) -> dict[str, object]:
    decision = make_decisions(gate_logits, action_logits, action_mask, thresholds)
    positive = np.asarray(group_target, dtype=bool)
    targets = np.asarray(action_target, dtype=np.int8)
    row = np.arange(len(positive))
    correct = decision.issued & (targets[row, decision.top_action] > 0)
    issued_positive = decision.issued & positive
    issued_count = int(decision.issued.sum())
    positive_count = int(positive.sum())
    issued_positive_count = int(issued_positive.sum())
    correct_count = int(correct.sum())
    selected = decision.top_action[decision.issued]
    if issued_count:
        counts = np.bincount(selected, minlength=5)
        diversity = int((counts > 0).sum())
        concentration = float(counts.max() / counts.sum())
    else:
        diversity = 0
        concentration = 1.0
    metrics: dict[str, object] = {
        "issued_groups": issued_count,
        "positive_groups": positive_count,
        "issued_positive_groups": issued_positive_count,
        "correct_issued_actions": correct_count,
        "false_issue_groups": int((decision.issued & ~positive).sum()),
        "stage_a_precision": _ratio(issued_positive_count, issued_count),
        "stage_a_recall": _ratio(issued_positive_count, positive_count),
        "stage_b_conditional_precision_at_1": _ratio(
            correct_count, issued_positive_count
        ),
        "end_to_end_precision_at_1": _ratio(correct_count, issued_count),
        "positive_group_coverage": _ratio(issued_positive_count, positive_count),
        "abstention_rate": 1.0 - _ratio(issued_count, len(positive)),
        "action_diversity": diversity,
        "top_action_concentration": concentration,
    }
    per_stage = []
    for stage in sorted(set(stages.tolist())):
        mask = stages == stage
        per_stage.append(
            {
                "stage": str(stage),
                **_fast_metrics(
                    gate_logits=np.asarray(gate_logits)[mask],
                    action_logits=np.asarray(action_logits)[mask],
                    action_mask=np.asarray(action_mask)[mask],
                    group_target=np.asarray(group_target)[mask],
                    action_target=np.asarray(action_target)[mask],
                    thresholds=thresholds,
                    stages=np.asarray([], dtype=object),
                ),
            }
        )
    metrics["per_stage"] = per_stage
    return metrics


def _worst_supported_stage(metrics: dict[str, object]) -> float:
    rows = metrics.get("per_stage", [])
    values = [
        float(row["end_to_end_precision_at_1"])
        for row in rows
        if int(row["issued_groups"]) >= 50
    ]
    return min(values) if values else 0.0


def _select_from_candidates(
    candidates: list[tuple[TwoStageThresholds, dict[str, object]]],
    *,
    minimum_coverage: float,
    target_precision: float,
) -> tuple[TwoStageThresholds, dict[str, object], str]:
    covered = [
        item
        for item in candidates
        if float(item[1]["positive_group_coverage"]) >= minimum_coverage
    ]
    target = [
        item
        for item in covered
        if float(item[1]["end_to_end_precision_at_1"]) >= target_precision
    ]
    pool = target or covered or candidates

    def key(item: tuple[TwoStageThresholds, dict[str, object]]) -> tuple[float, ...]:
        metrics = item[1]
        worst_stage = _worst_supported_stage(metrics)
        if target:
            return (
                float(metrics["positive_group_coverage"]),
                worst_stage,
                float(metrics["stage_b_conditional_precision_at_1"]),
                -float(metrics["abstention_rate"]),
            )
        return (
            float(metrics["end_to_end_precision_at_1"]),
            worst_stage,
            float(metrics["stage_b_conditional_precision_at_1"]),
            float(metrics["positive_group_coverage"]),
        )

    thresholds, metrics = max(pool, key=key)
    reason = (
        "TARGET_MET_MAXIMIZE_COVERAGE"
        if target
        else (
            "TARGET_NOT_MET_MAXIMIZE_PRECISION_WITH_COVERAGE"
            if covered
            else "COVERAGE_NOT_MET_MAXIMIZE_PRECISION"
        )
    )
    return thresholds, metrics, reason


def select_thresholds_staged(
    *,
    gate_logits: np.ndarray,
    action_logits: np.ndarray,
    action_mask: np.ndarray,
    group_target: np.ndarray,
    action_target: np.ndarray,
    stages: Iterable[str],
    minimum_coverage: float,
    target_precision: float,
    action_probability_grid: Iterable[float],
    margin_grid: Iterable[float],
    action_specific_minimum_support: int,
) -> tuple[TwoStageThresholds, dict[str, object], dict[str, object]]:
    """Search global thresholds, then calibrate per-action abstention once."""

    stage_values = np.asarray(list(stages), dtype=object)
    gate_probability = 1.0 / (
        1.0 + np.exp(-np.clip(np.asarray(gate_logits, dtype=np.float64), -40, 40))
    )
    quantiles = np.linspace(0.0, 1.0, 41)
    gate_grid = np.unique(
        np.concatenate(([0.0, 1.0], np.quantile(gate_probability, quantiles)))
    )
    global_candidates: list[tuple[TwoStageThresholds, dict[str, object]]] = []
    for gate_threshold in gate_grid:
        for action_threshold in action_probability_grid:
            for margin_threshold in margin_grid:
                thresholds = TwoStageThresholds(
                    gate_probability=float(gate_threshold),
                    minimum_action_probability=float(action_threshold),
                    minimum_action_margin=float(margin_threshold),
                )
                metrics = _fast_metrics(
                    gate_logits=gate_logits,
                    action_logits=action_logits,
                    action_mask=action_mask,
                    group_target=group_target,
                    action_target=action_target,
                    thresholds=thresholds,
                    stages=stage_values,
                )
                global_candidates.append((thresholds, metrics))
    global_thresholds, global_fast_metrics, reason = _select_from_candidates(
        global_candidates,
        minimum_coverage=minimum_coverage,
        target_precision=target_precision,
    )
    calibrated_thresholds = derive_action_thresholds(
        gate_logits=gate_logits,
        action_logits=action_logits,
        action_mask=action_mask,
        group_target=group_target,
        action_target=action_target,
        base_thresholds=global_thresholds,
        target_precision=target_precision,
        minimum_support=action_specific_minimum_support,
    )
    calibrated_fast_metrics = _fast_metrics(
        gate_logits=gate_logits,
        action_logits=action_logits,
        action_mask=action_mask,
        group_target=group_target,
        action_target=action_target,
        thresholds=calibrated_thresholds,
        stages=stage_values,
    )
    calibrated_usable = (
        float(calibrated_fast_metrics["positive_group_coverage"]) >= minimum_coverage
        and float(calibrated_fast_metrics["end_to_end_precision_at_1"])
        >= float(global_fast_metrics["end_to_end_precision_at_1"])
    )
    if calibrated_usable:
        selected_thresholds = calibrated_thresholds
        selected_reason = f"{reason}_ACTION_CALIBRATED"
    else:
        selected_thresholds = global_thresholds
        selected_reason = reason
    selected_metrics = evaluate_two_stage(
        gate_logits=gate_logits,
        action_logits=action_logits,
        action_mask=action_mask,
        group_target=group_target,
        action_target=action_target,
        thresholds=selected_thresholds,
        stages=stage_values,
    )
    selected_metrics["selection_target_met"] = bool(
        float(selected_metrics["positive_group_coverage"]) >= minimum_coverage
        and float(selected_metrics["end_to_end_precision_at_1"]) >= target_precision
    )
    audit = {
        "global_candidate_count": len(global_candidates),
        "gate_threshold_count": int(len(gate_grid)),
        "global_selection_reason": reason,
        "action_calibration_used": calibrated_usable,
        "selected_reason": selected_reason,
        "global_thresholds": global_thresholds.to_dict(),
        "calibrated_thresholds": calibrated_thresholds.to_dict(),
        "global_fast_metrics": global_fast_metrics,
        "calibrated_fast_metrics": calibrated_fast_metrics,
    }
    return selected_thresholds, selected_metrics, audit


__all__ = ["select_thresholds_staged"]
