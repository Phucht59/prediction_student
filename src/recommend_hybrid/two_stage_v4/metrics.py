"""Action-aware decision rules and held-out metrics for integrated V4."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .model import ACTION_COUNT, MASKED_LOGIT

STAGE_ORDER = ("EARLY_20", "EARLY_35", "MIDDLE_50")
STAGE_INDEX = {name: index for index, name in enumerate(STAGE_ORDER)}
EPSILON = 1.0e-9


@dataclass(frozen=True)
class ActionAwareThresholds:
    stage_gate_probability: tuple[float, float, float]
    direct_action_blend: float
    minimum_action_probability: float
    minimum_action_margin: float
    action_probability_by_id: tuple[float, ...] = (0.0,) * ACTION_COUNT

    def __post_init__(self) -> None:
        if len(self.stage_gate_probability) != len(STAGE_ORDER):
            raise ValueError("stage threshold vector has the wrong length")
        if len(self.action_probability_by_id) != ACTION_COUNT:
            raise ValueError("action threshold vector has the wrong length")
        for value in (
            *self.stage_gate_probability,
            self.direct_action_blend,
            self.minimum_action_probability,
            self.minimum_action_margin,
            *self.action_probability_by_id,
        ):
            if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
                raise ValueError("threshold values must be finite in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_gate_probability": list(self.stage_gate_probability),
            "direct_action_blend": self.direct_action_blend,
            "minimum_action_probability": self.minimum_action_probability,
            "minimum_action_margin": self.minimum_action_margin,
            "action_probability_by_id": list(self.action_probability_by_id),
        }


@dataclass(frozen=True)
class ActionAwareDecision:
    issued: np.ndarray
    top_action: np.ndarray
    top_probability: np.ndarray
    top_margin: np.ndarray
    direct_gate_probability: np.ndarray
    action_any_probability: np.ndarray
    joint_gate_probability: np.ndarray


def sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def action_probabilities(
    action_logits: np.ndarray,
    action_mask: np.ndarray,
) -> np.ndarray:
    valid = np.asarray(action_mask, dtype=bool)
    probability = sigmoid(np.asarray(action_logits, dtype=np.float64))
    return np.where(valid, probability, 0.0)


def action_any_probability(
    action_logits: np.ndarray,
    action_mask: np.ndarray,
) -> np.ndarray:
    probability = action_probabilities(action_logits, action_mask)
    valid = np.asarray(action_mask, dtype=bool)
    none_probability = np.prod(
        np.where(valid, 1.0 - probability, 1.0),
        axis=1,
    )
    return np.clip(1.0 - none_probability, EPSILON, 1.0 - EPSILON)


def blended_gate_probability(
    direct_gate_logits: np.ndarray,
    action_logits: np.ndarray,
    action_mask: np.ndarray,
    blend: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direct = np.clip(sigmoid(direct_gate_logits), EPSILON, 1.0 - EPSILON)
    action_any = action_any_probability(action_logits, action_mask)
    alpha = float(blend)
    joint = np.exp(alpha * np.log(direct) + (1.0 - alpha) * np.log(action_any))
    return direct, action_any, np.clip(joint, EPSILON, 1.0 - EPSILON)


def _stage_thresholds(stages: Iterable[str], thresholds: ActionAwareThresholds) -> np.ndarray:
    values = []
    for stage in stages:
        index = STAGE_INDEX.get(str(stage))
        if index is None:
            raise ValueError(f"unknown stage {stage}")
        values.append(thresholds.stage_gate_probability[index])
    return np.asarray(values, dtype=np.float64)


def make_decisions(
    direct_gate_logits: np.ndarray,
    action_logits: np.ndarray,
    action_mask: np.ndarray,
    stages: Iterable[str],
    thresholds: ActionAwareThresholds,
) -> ActionAwareDecision:
    valid = np.asarray(action_mask, dtype=bool)
    probability = action_probabilities(action_logits, valid)
    row = np.arange(len(probability))
    top_action = np.argmax(np.where(valid, probability, -np.inf), axis=1)
    top_probability = probability[row, top_action]
    without_top = probability.copy()
    without_top[row, top_action] = -np.inf
    second = np.max(without_top, axis=1)
    second = np.where(np.isfinite(second), second, 0.0)
    margin = top_probability - second
    direct, action_any, joint = blended_gate_probability(
        direct_gate_logits,
        action_logits,
        valid,
        thresholds.direct_action_blend,
    )
    stage_threshold = _stage_thresholds(stages, thresholds)
    action_threshold = np.asarray(
        thresholds.action_probability_by_id,
        dtype=np.float64,
    )[top_action]
    has_candidate = valid.any(axis=1)
    issued = (
        has_candidate
        & (joint >= stage_threshold)
        & (top_probability >= thresholds.minimum_action_probability)
        & (top_probability >= action_threshold)
        & (margin >= thresholds.minimum_action_margin)
    )
    return ActionAwareDecision(
        issued=issued,
        top_action=top_action,
        top_probability=top_probability,
        top_margin=margin,
        direct_gate_probability=direct,
        action_any_probability=action_any,
        joint_gate_probability=joint,
    )


def _ratio(numerator: int | float, denominator: int | float) -> float:
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
    precision_values: list[float] = []
    ndcg_values: list[float] = []
    reciprocal_values: list[float] = []
    for index in np.where(positive_group)[0]:
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
        "conditional_precision_at_1_all_positive": float(np.mean(precision_values))
        if precision_values
        else 0.0,
        "ndcg_at_3": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "mrr": float(np.mean(reciprocal_values)) if reciprocal_values else 0.0,
    }


def evaluate_action_aware(
    *,
    direct_gate_logits: np.ndarray,
    action_logits: np.ndarray,
    action_mask: np.ndarray,
    group_target: np.ndarray,
    action_target: np.ndarray,
    stages: Iterable[str],
    thresholds: ActionAwareThresholds,
    include_breakdown: bool = True,
) -> dict[str, object]:
    stage_values = np.asarray(list(stages), dtype=object)
    positive = np.asarray(group_target, dtype=bool)
    action_positive = np.asarray(action_target, dtype=np.int8)
    decision = make_decisions(
        direct_gate_logits,
        action_logits,
        action_mask,
        stage_values,
        thresholds,
    )
    row = np.arange(len(positive))
    correct = decision.issued & (action_positive[row, decision.top_action] > 0)
    issued_positive = decision.issued & positive
    issued_count = int(decision.issued.sum())
    positive_count = int(positive.sum())
    issued_positive_count = int(issued_positive.sum())
    correct_count = int(correct.sum())
    selected = decision.top_action[decision.issued]
    if issued_count:
        counts = np.bincount(selected, minlength=ACTION_COUNT)
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
            correct_count,
            issued_positive_count,
        ),
        "end_to_end_precision_at_1": _ratio(correct_count, issued_count),
        "positive_group_coverage": _ratio(issued_positive_count, positive_count),
        "abstention_rate": 1.0 - _ratio(issued_count, len(positive)),
        "action_diversity": diversity,
        "top_action_concentration": concentration,
        **ranking_metrics(action_logits, action_target, action_mask, group_target),
    }
    if include_breakdown:
        per_stage = []
        for stage in STAGE_ORDER:
            mask = stage_values == stage
            if not mask.any():
                continue
            per_stage.append(
                {
                    "stage": stage,
                    **evaluate_action_aware(
                        direct_gate_logits=np.asarray(direct_gate_logits)[mask],
                        action_logits=np.asarray(action_logits)[mask],
                        action_mask=np.asarray(action_mask)[mask],
                        group_target=np.asarray(group_target)[mask],
                        action_target=np.asarray(action_target)[mask],
                        stages=stage_values[mask],
                        thresholds=thresholds,
                        include_breakdown=False,
                    ),
                }
            )
        metrics["per_stage"] = per_stage
    return metrics


__all__ = [
    "ACTION_COUNT",
    "ActionAwareDecision",
    "ActionAwareThresholds",
    "STAGE_ORDER",
    "action_any_probability",
    "action_probabilities",
    "blended_gate_probability",
    "evaluate_action_aware",
    "make_decisions",
    "ranking_metrics",
    "sigmoid",
]
