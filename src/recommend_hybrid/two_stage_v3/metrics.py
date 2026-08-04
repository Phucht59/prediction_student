"""Leakage-safe metrics and threshold selection for Two-Stage V3."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .model import ACTION_COUNT, MASKED_LOGIT


@dataclass(frozen=True)
class TwoStageThresholds:
    gate_probability: float
    minimum_action_probability: float
    minimum_action_margin: float
    action_probability_by_id: tuple[float, ...] = (0.0,) * ACTION_COUNT

    def __post_init__(self) -> None:
        if len(self.action_probability_by_id) != ACTION_COUNT:
            raise ValueError("action threshold vector has the wrong length")
        for value in (
            self.gate_probability,
            self.minimum_action_probability,
            self.minimum_action_margin,
            *self.action_probability_by_id,
        ):
            if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
                raise ValueError("thresholds must be finite in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "gate_probability": self.gate_probability,
            "minimum_action_probability": self.minimum_action_probability,
            "minimum_action_margin": self.minimum_action_margin,
            "action_probability_by_id": list(self.action_probability_by_id),
        }


@dataclass(frozen=True)
class TwoStageDecision:
    issued: np.ndarray
    top_action: np.ndarray
    top_probability: np.ndarray
    top_margin: np.ndarray


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    masked = np.where(valid, values, MASKED_LOGIT)
    maximum = np.max(masked, axis=1, keepdims=True)
    exponent = np.where(valid, np.exp(masked - maximum), 0.0)
    denominator = exponent.sum(axis=1, keepdims=True)
    return np.divide(
        exponent,
        denominator,
        out=np.zeros_like(exponent),
        where=denominator > 0,
    )


def make_decisions(
    gate_logits: np.ndarray,
    action_logits: np.ndarray,
    action_mask: np.ndarray,
    thresholds: TwoStageThresholds,
) -> TwoStageDecision:
    gate_probability = _sigmoid(gate_logits)
    action_probability = _softmax(action_logits, action_mask)
    row = np.arange(len(gate_probability))
    top_action = np.argmax(action_probability, axis=1)
    top_probability = action_probability[row, top_action]
    without_top = action_probability.copy()
    without_top[row, top_action] = -np.inf
    second = np.max(without_top, axis=1)
    second = np.where(np.isfinite(second), second, 0.0)
    margin = top_probability - second
    action_threshold = np.asarray(
        thresholds.action_probability_by_id, dtype=np.float64
    )[top_action]
    has_candidate = np.asarray(action_mask, dtype=bool).any(axis=1)
    issued = (
        has_candidate
        & (gate_probability >= thresholds.gate_probability)
        & (top_probability >= thresholds.minimum_action_probability)
        & (top_probability >= action_threshold)
        & (margin >= thresholds.minimum_action_margin)
    )
    return TwoStageDecision(
        issued=issued,
        top_action=top_action,
        top_probability=top_probability,
        top_margin=margin,
    )


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def ranking_metrics(
    action_logits: np.ndarray,
    action_target: np.ndarray,
    action_mask: np.ndarray,
    group_target: np.ndarray,
) -> dict[str, float]:
    logits = np.asarray(action_logits, dtype=np.float64)
    targets = np.asarray(action_target, dtype=np.int8)
    valid = np.asarray(action_mask, dtype=bool)
    positive_group = np.asarray(group_target, dtype=bool)
    rows = np.where(positive_group)[0]
    if not len(rows):
        return {"conditional_precision_at_1_all_positive": 0.0, "ndcg_at_3": 0.0, "mrr": 0.0}
    precision_values: list[float] = []
    ndcg_values: list[float] = []
    reciprocal_values: list[float] = []
    for index in rows:
        candidates = np.where(valid[index])[0]
        if not len(candidates):
            continue
        order = candidates[np.argsort(-logits[index, candidates], kind="stable")]
        relevance = targets[index, order]
        precision_values.append(float(relevance[0] > 0))
        hit = np.where(relevance > 0)[0]
        reciprocal_values.append(float(1.0 / (hit[0] + 1)) if len(hit) else 0.0)
        gains = relevance[:3] / np.log2(np.arange(min(3, len(relevance))) + 2.0)
        dcg = float(gains.sum())
        ideal = np.sort(targets[index, candidates])[::-1][:3]
        idcg = float((ideal / np.log2(np.arange(len(ideal)) + 2.0)).sum())
        ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)
    return {
        "conditional_precision_at_1_all_positive": float(np.mean(precision_values)) if precision_values else 0.0,
        "ndcg_at_3": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "mrr": float(np.mean(reciprocal_values)) if reciprocal_values else 0.0,
    }


def evaluate_two_stage(
    *,
    gate_logits: np.ndarray,
    action_logits: np.ndarray,
    action_mask: np.ndarray,
    group_target: np.ndarray,
    action_target: np.ndarray,
    thresholds: TwoStageThresholds,
    stages: Iterable[str] | None = None,
) -> dict[str, object]:
    group_positive = np.asarray(group_target, dtype=bool)
    action_positive = np.asarray(action_target, dtype=np.int8)
    decision = make_decisions(gate_logits, action_logits, action_mask, thresholds)
    row = np.arange(len(group_positive))
    correct = decision.issued & (action_positive[row, decision.top_action] > 0)
    issued_positive = decision.issued & group_positive
    issued_count = int(decision.issued.sum())
    positive_count = int(group_positive.sum())
    issued_positive_count = int(issued_positive.sum())
    correct_count = int(correct.sum())
    selected = decision.top_action[decision.issued]
    if issued_count:
        counts = np.bincount(selected, minlength=ACTION_COUNT)
        action_diversity = int((counts > 0).sum())
        top_concentration = float(counts.max() / counts.sum())
    else:
        action_diversity = 0
        top_concentration = 1.0
    metrics: dict[str, object] = {
        "issued_groups": issued_count,
        "positive_groups": positive_count,
        "issued_positive_groups": issued_positive_count,
        "correct_issued_actions": correct_count,
        "false_issue_groups": int((decision.issued & ~group_positive).sum()),
        "stage_a_precision": _safe_ratio(issued_positive_count, issued_count),
        "stage_a_recall": _safe_ratio(issued_positive_count, positive_count),
        "stage_b_conditional_precision_at_1": _safe_ratio(correct_count, issued_positive_count),
        "end_to_end_precision_at_1": _safe_ratio(correct_count, issued_count),
        "positive_group_coverage": _safe_ratio(issued_positive_count, positive_count),
        "abstention_rate": 1.0 - _safe_ratio(issued_count, len(group_positive)),
        "action_diversity": action_diversity,
        "top_action_concentration": top_concentration,
        **ranking_metrics(action_logits, action_target, action_mask, group_target),
    }
    if stages is not None:
        stage_array = np.asarray(list(stages), dtype=object)
        per_stage = []
        for stage in sorted(set(stage_array.tolist())):
            mask = stage_array == stage
            stage_result = evaluate_two_stage(
                gate_logits=np.asarray(gate_logits)[mask],
                action_logits=np.asarray(action_logits)[mask],
                action_mask=np.asarray(action_mask)[mask],
                group_target=np.asarray(group_target)[mask],
                action_target=np.asarray(action_target)[mask],
                thresholds=thresholds,
                stages=None,
            )
            per_stage.append({"stage": str(stage), **stage_result})
        metrics["per_stage"] = per_stage
    return metrics


def derive_action_thresholds(
    *,
    gate_logits: np.ndarray,
    action_logits: np.ndarray,
    action_mask: np.ndarray,
    group_target: np.ndarray,
    action_target: np.ndarray,
    base_thresholds: TwoStageThresholds,
    target_precision: float,
    minimum_support: int,
) -> TwoStageThresholds:
    """Derive per-action abstention thresholds from inner OOF predictions only."""

    base = TwoStageThresholds(
        gate_probability=base_thresholds.gate_probability,
        minimum_action_probability=base_thresholds.minimum_action_probability,
        minimum_action_margin=base_thresholds.minimum_action_margin,
    )
    decision = make_decisions(gate_logits, action_logits, action_mask, base)
    row = np.arange(len(decision.issued))
    correct = np.asarray(action_target)[row, decision.top_action] > 0
    thresholds = np.zeros(ACTION_COUNT, dtype=np.float64)
    for action_id in range(ACTION_COUNT):
        selected = decision.issued & (decision.top_action == action_id)
        if int(selected.sum()) < minimum_support:
            thresholds[action_id] = 1.0
            continue
        probabilities = np.unique(decision.top_probability[selected])
        candidates = np.concatenate(([0.0], probabilities, [1.0]))
        best: tuple[float, int] | None = None
        for threshold in candidates:
            kept = selected & (decision.top_probability >= threshold)
            support = int(kept.sum())
            precision = _safe_ratio(int(correct[kept].sum()), support)
            if support >= minimum_support and precision >= target_precision:
                key = (float(threshold), -support)
                if best is None or key < best:
                    best = key
        thresholds[action_id] = best[0] if best is not None else 1.0
    return TwoStageThresholds(
        gate_probability=base_thresholds.gate_probability,
        minimum_action_probability=base_thresholds.minimum_action_probability,
        minimum_action_margin=base_thresholds.minimum_action_margin,
        action_probability_by_id=tuple(float(value) for value in thresholds),
    )


def select_thresholds(
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
) -> tuple[TwoStageThresholds, dict[str, object]]:
    gate_probability = _sigmoid(gate_logits)
    quantiles = np.linspace(0.0, 1.0, 101)
    gate_grid = np.unique(np.concatenate(([0.0, 1.0], np.quantile(gate_probability, quantiles))))
    candidates: list[tuple[TwoStageThresholds, dict[str, object]]] = []
    stage_values = list(stages)
    for gate_threshold in gate_grid:
        for action_threshold in action_probability_grid:
            for margin_threshold in margin_grid:
                base = TwoStageThresholds(
                    gate_probability=float(gate_threshold),
                    minimum_action_probability=float(action_threshold),
                    minimum_action_margin=float(margin_threshold),
                )
                calibrated = derive_action_thresholds(
                    gate_logits=gate_logits,
                    action_logits=action_logits,
                    action_mask=action_mask,
                    group_target=group_target,
                    action_target=action_target,
                    base_thresholds=base,
                    target_precision=target_precision,
                    minimum_support=action_specific_minimum_support,
                )
                metrics = evaluate_two_stage(
                    gate_logits=gate_logits,
                    action_logits=action_logits,
                    action_mask=action_mask,
                    group_target=group_target,
                    action_target=action_target,
                    thresholds=calibrated,
                    stages=stage_values,
                )
                candidates.append((calibrated, metrics))
    covered = [
        item for item in candidates
        if float(item[1]["positive_group_coverage"]) >= minimum_coverage
    ]
    target = [
        item for item in covered
        if float(item[1]["end_to_end_precision_at_1"]) >= target_precision
    ]
    pool = target or covered or candidates

    def key(item: tuple[TwoStageThresholds, dict[str, object]]) -> tuple[float, ...]:
        metrics = item[1]
        per_stage = metrics.get("per_stage", [])
        supported = [
            float(row["end_to_end_precision_at_1"])
            for row in per_stage
            if int(row["issued_groups"]) >= 50
        ]
        worst_stage = min(supported) if supported else 0.0
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

    selected = max(pool, key=key)
    selected[1]["selection_target_met"] = bool(target)
    return selected


__all__ = [
    "TwoStageDecision",
    "TwoStageThresholds",
    "derive_action_thresholds",
    "evaluate_two_stage",
    "make_decisions",
    "ranking_metrics",
    "select_thresholds",
]
