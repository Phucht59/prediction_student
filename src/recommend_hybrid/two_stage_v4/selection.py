"""Inner-OOF constrained selection for action-aware integrated V4."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .metrics import (
    ACTION_COUNT,
    STAGE_ORDER,
    ActionAwareThresholds,
    action_probabilities,
    blended_gate_probability,
    evaluate_action_aware,
    make_decisions,
)


@dataclass(frozen=True)
class _StageChoice:
    threshold: float
    issued: int
    issued_positive: int
    correct: int
    false_issue: int
    coverage: float
    precision: float


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _stage_choices(
    *,
    score: np.ndarray,
    base_eligible: np.ndarray,
    group_target: np.ndarray,
    correct_target: np.ndarray,
    stage_mask: np.ndarray,
    quantile_count: int,
) -> list[_StageChoice]:
    mask = np.asarray(base_eligible, dtype=bool) & np.asarray(stage_mask, dtype=bool)
    positive_count = int((np.asarray(group_target, dtype=bool) & stage_mask).sum())
    values = np.asarray(score, dtype=np.float64)[mask]
    if not len(values):
        return [
            _StageChoice(
                threshold=1.0,
                issued=0,
                issued_positive=0,
                correct=0,
                false_issue=0,
                coverage=0.0,
                precision=0.0,
            )
        ]
    quantiles = np.linspace(0.0, 1.0, max(2, int(quantile_count)))
    thresholds = np.unique(
        np.concatenate(([0.0, 1.0], np.quantile(values, quantiles)))
    )
    result = []
    positive = np.asarray(group_target, dtype=bool)
    correct = np.asarray(correct_target, dtype=bool)
    for threshold in thresholds:
        issued_mask = mask & (score >= threshold)
        issued = int(issued_mask.sum())
        issued_positive = int((issued_mask & positive).sum())
        correct_count = int((issued_mask & correct).sum())
        result.append(
            _StageChoice(
                threshold=float(threshold),
                issued=issued,
                issued_positive=issued_positive,
                correct=correct_count,
                false_issue=int((issued_mask & ~positive).sum()),
                coverage=_ratio(issued_positive, positive_count),
                precision=_ratio(correct_count, issued),
            )
        )
    result.sort(key=lambda item: (item.issued, item.threshold))
    deduplicated: list[_StageChoice] = []
    seen = set()
    for item in result:
        key = (item.issued, item.issued_positive, item.correct)
        if key not in seen:
            deduplicated.append(item)
            seen.add(key)
    return deduplicated


def _choose_initial_stage(
    choices: list[_StageChoice],
    minimum_coverage: float,
) -> int:
    feasible = [
        (index, item)
        for index, item in enumerate(choices)
        if item.coverage >= minimum_coverage
    ]
    pool = feasible or list(enumerate(choices))
    if feasible:
        return max(
            pool,
            key=lambda pair: (
                pair[1].precision,
                pair[1].coverage,
                -pair[1].issued,
                pair[1].threshold,
            ),
        )[0]
    return max(
        pool,
        key=lambda pair: (
            pair[1].coverage,
            pair[1].precision,
            -pair[1].issued,
        ),
    )[0]


def _next_expansion(
    choices: list[_StageChoice],
    current: int,
) -> int | None:
    current_issued = choices[current].issued
    candidates = [
        (index, item)
        for index, item in enumerate(choices)
        if item.issued > current_issued
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda pair: pair[1].issued)[0]


def _derive_stage_thresholds(
    *,
    joint_probability: np.ndarray,
    base_eligible: np.ndarray,
    top_action: np.ndarray,
    group_target: np.ndarray,
    action_target: np.ndarray,
    stages: np.ndarray,
    stage_coverage_floor: Mapping[str, float],
    global_coverage_floor: float,
    quantile_count: int,
) -> tuple[tuple[float, float, float], dict[str, object]]:
    row = np.arange(len(group_target))
    correct_target = np.asarray(action_target)[row, top_action] > 0
    stage_choices: dict[str, list[_StageChoice]] = {}
    selected_index: dict[str, int] = {}
    for stage in STAGE_ORDER:
        choices = _stage_choices(
            score=joint_probability,
            base_eligible=base_eligible,
            group_target=group_target,
            correct_target=correct_target,
            stage_mask=np.asarray(stages, dtype=object) == stage,
            quantile_count=quantile_count,
        )
        stage_choices[stage] = choices
        selected_index[stage] = _choose_initial_stage(
            choices,
            float(stage_coverage_floor.get(stage, 0.0)),
        )

    total_positive = int(np.asarray(group_target, dtype=bool).sum())

    def totals() -> tuple[int, int, int]:
        selected = [
            stage_choices[stage][selected_index[stage]] for stage in STAGE_ORDER
        ]
        return (
            sum(item.issued for item in selected),
            sum(item.issued_positive for item in selected),
            sum(item.correct for item in selected),
        )

    while _ratio(totals()[1], total_positive) < global_coverage_floor:
        proposals = []
        for stage in STAGE_ORDER:
            next_index = _next_expansion(stage_choices[stage], selected_index[stage])
            if next_index is None:
                continue
            current_item = stage_choices[stage][selected_index[stage]]
            next_item = stage_choices[stage][next_index]
            delta_issued = next_item.issued - current_item.issued
            delta_positive = next_item.issued_positive - current_item.issued_positive
            delta_correct = next_item.correct - current_item.correct
            proposals.append(
                (
                    _ratio(delta_correct, delta_issued),
                    _ratio(delta_positive, delta_issued),
                    delta_positive,
                    -delta_issued,
                    stage,
                    next_index,
                )
            )
        if not proposals:
            break
        *_, selected_stage, next_index = max(proposals)
        selected_index[selected_stage] = next_index

    selected_choices = {
        stage: stage_choices[stage][selected_index[stage]] for stage in STAGE_ORDER
    }
    thresholds = tuple(
        float(selected_choices[stage].threshold) for stage in STAGE_ORDER
    )
    issued, issued_positive, correct = totals()
    audit = {
        "stage_choices": {
            stage: {
                "threshold": selected_choices[stage].threshold,
                "issued": selected_choices[stage].issued,
                "issued_positive": selected_choices[stage].issued_positive,
                "correct": selected_choices[stage].correct,
                "coverage": selected_choices[stage].coverage,
                "precision": selected_choices[stage].precision,
                "candidate_count": len(stage_choices[stage]),
            }
            for stage in STAGE_ORDER
        },
        "combined": {
            "issued": issued,
            "issued_positive": issued_positive,
            "correct": correct,
            "positive_group_coverage": _ratio(issued_positive, total_positive),
            "end_to_end_precision_at_1": _ratio(correct, issued),
        },
    }
    return thresholds, audit


def _derive_action_thresholds(
    *,
    direct_gate_logits: np.ndarray,
    action_logits: np.ndarray,
    action_mask: np.ndarray,
    group_target: np.ndarray,
    action_target: np.ndarray,
    stages: np.ndarray,
    base_thresholds: ActionAwareThresholds,
    target_precision: float,
    minimum_support: int,
) -> ActionAwareThresholds:
    base = ActionAwareThresholds(
        stage_gate_probability=base_thresholds.stage_gate_probability,
        direct_action_blend=base_thresholds.direct_action_blend,
        minimum_action_probability=base_thresholds.minimum_action_probability,
        minimum_action_margin=base_thresholds.minimum_action_margin,
    )
    decision = make_decisions(
        direct_gate_logits,
        action_logits,
        action_mask,
        stages,
        base,
    )
    row = np.arange(len(group_target))
    correct = np.asarray(action_target)[row, decision.top_action] > 0
    thresholds = np.zeros(ACTION_COUNT, dtype=np.float64)
    for action_id in range(ACTION_COUNT):
        selected = decision.issued & (decision.top_action == action_id)
        if int(selected.sum()) < minimum_support:
            thresholds[action_id] = 1.0
            continue
        values = decision.top_probability[selected]
        candidates = np.unique(
            np.concatenate(([0.0, 1.0], np.quantile(values, np.linspace(0, 1, 61))))
        )
        valid_candidates = []
        for threshold in candidates:
            kept = selected & (decision.top_probability >= threshold)
            support = int(kept.sum())
            precision = _ratio(int(correct[kept].sum()), support)
            if support >= minimum_support and precision >= target_precision:
                valid_candidates.append((float(threshold), -support))
        thresholds[action_id] = min(valid_candidates)[0] if valid_candidates else 1.0
    return ActionAwareThresholds(
        stage_gate_probability=base_thresholds.stage_gate_probability,
        direct_action_blend=base_thresholds.direct_action_blend,
        minimum_action_probability=base_thresholds.minimum_action_probability,
        minimum_action_margin=base_thresholds.minimum_action_margin,
        action_probability_by_id=tuple(float(value) for value in thresholds),
    )


def _worst_supported_stage(metrics: dict[str, object]) -> float:
    values = [
        float(row["end_to_end_precision_at_1"])
        for row in metrics.get("per_stage", [])
        if int(row["issued_groups"]) >= 50
    ]
    return min(values) if values else 0.0


def select_action_aware_thresholds(
    *,
    direct_gate_logits: np.ndarray,
    action_logits: np.ndarray,
    action_mask: np.ndarray,
    group_target: np.ndarray,
    action_target: np.ndarray,
    stages: Iterable[str],
    blend_weights: Iterable[float],
    action_probability_grid: Iterable[float],
    margin_grid: Iterable[float],
    stage_coverage_floor: Mapping[str, float],
    minimum_global_coverage: float,
    target_precision: float,
    action_specific_minimum_support: int,
    stage_quantile_count: int = 61,
) -> tuple[ActionAwareThresholds, dict[str, object], dict[str, object]]:
    """Select joint gate, stage thresholds, and action abstention on inner OOF."""

    stage_values = np.asarray(list(stages), dtype=object)
    probability = action_probabilities(action_logits, action_mask)
    row = np.arange(len(stage_values))
    top_action = np.argmax(np.where(action_mask, probability, -np.inf), axis=1)
    top_probability = probability[row, top_action]
    without_top = probability.copy()
    without_top[row, top_action] = -np.inf
    second = np.max(without_top, axis=1)
    second = np.where(np.isfinite(second), second, 0.0)
    margin = top_probability - second
    has_candidate = np.asarray(action_mask, dtype=bool).any(axis=1)
    candidates: list[tuple[ActionAwareThresholds, dict[str, object], dict[str, object]]] = []

    for blend in blend_weights:
        _, _, joint = blended_gate_probability(
            direct_gate_logits,
            action_logits,
            action_mask,
            float(blend),
        )
        for action_probability_threshold in action_probability_grid:
            for margin_threshold in margin_grid:
                base_eligible = (
                    has_candidate
                    & (top_probability >= float(action_probability_threshold))
                    & (margin >= float(margin_threshold))
                )
                stage_thresholds, stage_audit = _derive_stage_thresholds(
                    joint_probability=joint,
                    base_eligible=base_eligible,
                    top_action=top_action,
                    group_target=group_target,
                    action_target=action_target,
                    stages=stage_values,
                    stage_coverage_floor=stage_coverage_floor,
                    global_coverage_floor=minimum_global_coverage,
                    quantile_count=stage_quantile_count,
                )
                base = ActionAwareThresholds(
                    stage_gate_probability=stage_thresholds,
                    direct_action_blend=float(blend),
                    minimum_action_probability=float(action_probability_threshold),
                    minimum_action_margin=float(margin_threshold),
                )
                base_metrics = evaluate_action_aware(
                    direct_gate_logits=direct_gate_logits,
                    action_logits=action_logits,
                    action_mask=action_mask,
                    group_target=group_target,
                    action_target=action_target,
                    stages=stage_values,
                    thresholds=base,
                )
                calibrated = _derive_action_thresholds(
                    direct_gate_logits=direct_gate_logits,
                    action_logits=action_logits,
                    action_mask=action_mask,
                    group_target=group_target,
                    action_target=action_target,
                    stages=stage_values,
                    base_thresholds=base,
                    target_precision=target_precision,
                    minimum_support=action_specific_minimum_support,
                )
                calibrated_metrics = evaluate_action_aware(
                    direct_gate_logits=direct_gate_logits,
                    action_logits=action_logits,
                    action_mask=action_mask,
                    group_target=group_target,
                    action_target=action_target,
                    stages=stage_values,
                    thresholds=calibrated,
                )
                calibrated_usable = (
                    float(calibrated_metrics["positive_group_coverage"])
                    >= minimum_global_coverage
                    and float(calibrated_metrics["end_to_end_precision_at_1"])
                    >= float(base_metrics["end_to_end_precision_at_1"])
                )
                selected_thresholds = calibrated if calibrated_usable else base
                selected_metrics = calibrated_metrics if calibrated_usable else base_metrics
                candidates.append(
                    (
                        selected_thresholds,
                        selected_metrics,
                        {
                            "stage_threshold_audit": stage_audit,
                            "action_calibration_used": calibrated_usable,
                            "base_thresholds": base.to_dict(),
                            "calibrated_thresholds": calibrated.to_dict(),
                            "base_metrics": base_metrics,
                            "calibrated_metrics": calibrated_metrics,
                        },
                    )
                )

    covered = [
        item
        for item in candidates
        if float(item[1]["positive_group_coverage"]) >= minimum_global_coverage
    ]
    target = [
        item
        for item in covered
        if float(item[1]["end_to_end_precision_at_1"]) >= target_precision
    ]
    pool = target or covered or candidates

    def key(item):
        metrics = item[1]
        worst_stage = _worst_supported_stage(metrics)
        if target:
            return (
                float(metrics["positive_group_coverage"]),
                worst_stage,
                float(metrics["stage_a_precision"]),
                float(metrics["stage_b_conditional_precision_at_1"]),
                -float(metrics["abstention_rate"]),
            )
        return (
            float(metrics["end_to_end_precision_at_1"]),
            float(metrics["stage_a_precision"]),
            worst_stage,
            float(metrics["stage_b_conditional_precision_at_1"]),
            float(metrics["positive_group_coverage"]),
        )

    selected_thresholds, selected_metrics, selected_audit = max(pool, key=key)
    selected_metrics["selection_target_met"] = bool(target)
    selected_audit.update(
        {
            "candidate_count": len(candidates),
            "covered_candidate_count": len(covered),
            "target_candidate_count": len(target),
            "selected_thresholds": selected_thresholds.to_dict(),
        }
    )
    return selected_thresholds, selected_metrics, selected_audit


__all__ = ["select_action_aware_thresholds"]
