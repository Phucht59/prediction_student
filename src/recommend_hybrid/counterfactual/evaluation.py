"""Evaluation contracts and metrics for counterfactual recommendations."""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable

from src.recommend_hybrid.exceptions import ContractValidationError


@dataclass(frozen=True)
class CounterfactualEvaluationRow:
    student_key: str
    course_key: str
    stage: str
    fold: int
    baseline_risk: float
    decision_threshold: float
    status: str
    top_action_id: str | None
    top_counterfactual_risk: float | None
    top_risk_reduction: float | None
    top_utility_score: float | None
    selected_action_count: int
    selected_workload_minutes: int
    reference_profile_id: str | None
    fallback_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.student_key or not self.course_key or not self.stage:
            raise ContractValidationError("evaluation row identity is required")
        if self.fold < 0:
            raise ContractValidationError("evaluation fold must be non-negative")
        for name, value in (
            ("baseline_risk", self.baseline_risk),
            ("decision_threshold", self.decision_threshold),
        ):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ContractValidationError(f"{name} must be in [0, 1]")
        if self.selected_action_count < 0 or self.selected_workload_minutes < 0:
            raise ContractValidationError(
                "selected action count and workload must be non-negative"
            )
        scored_values = (
            self.top_action_id,
            self.top_counterfactual_risk,
            self.top_risk_reduction,
            self.top_utility_score,
        )
        if any(value is not None for value in scored_values) and not all(
            value is not None for value in scored_values
        ):
            raise ContractValidationError(
                "top action evaluation fields must be supplied together"
            )
        if self.top_counterfactual_risk is not None:
            assert self.top_risk_reduction is not None
            assert self.top_utility_score is not None
            if not 0.0 <= self.top_counterfactual_risk <= 1.0:
                raise ContractValidationError(
                    "top_counterfactual_risk must be in [0, 1]"
                )
            expected = self.baseline_risk - self.top_counterfactual_risk
            if abs(expected - self.top_risk_reduction) > 1e-8:
                raise ContractValidationError(
                    "top_risk_reduction is inconsistent"
                )
            if self.top_utility_score < 0.0 or not isfinite(
                self.top_utility_score
            ):
                raise ContractValidationError(
                    "top_utility_score must be finite and non-negative"
                )

    @property
    def is_scored(self) -> bool:
        return self.top_action_id is not None

    @property
    def threshold_crossed(self) -> bool:
        return bool(
            self.is_scored
            and self.baseline_risk >= self.decision_threshold
            and self.top_counterfactual_risk is not None
            and self.top_counterfactual_risk < self.decision_threshold
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "student_key": self.student_key,
            "course_key": self.course_key,
            "stage": self.stage,
            "fold": self.fold,
            "baseline_risk": self.baseline_risk,
            "decision_threshold": self.decision_threshold,
            "status": self.status,
            "top_action_id": self.top_action_id,
            "top_counterfactual_risk": self.top_counterfactual_risk,
            "top_risk_reduction": self.top_risk_reduction,
            "top_utility_score": self.top_utility_score,
            "selected_action_count": self.selected_action_count,
            "selected_workload_minutes": self.selected_workload_minutes,
            "reference_profile_id": self.reference_profile_id,
            "fallback_reasons": list(self.fallback_reasons),
            "threshold_crossed": self.threshold_crossed,
        }


def aggregate_counterfactual_metrics(
    rows: Iterable[CounterfactualEvaluationRow],
) -> dict[str, Any]:
    records = tuple(rows)
    scored = tuple(row for row in records if row.is_scored)
    reductions = [float(row.top_risk_reduction) for row in scored]
    utilities = [float(row.top_utility_score) for row in scored]
    at_risk = tuple(
        row for row in scored if row.baseline_risk >= row.decision_threshold
    )
    action_frequency = Counter(
        str(row.top_action_id) for row in scored if row.top_action_id is not None
    )
    statuses = Counter(row.status for row in records)
    fallback = tuple(row for row in records if row.fallback_reasons)
    return {
        "record_count": len(records),
        "scored_count": len(scored),
        "scored_coverage": len(scored) / len(records) if records else 0.0,
        "fallback_count": len(fallback),
        "fallback_rate": len(fallback) / len(records) if records else 0.0,
        "mean_top_risk_reduction": (
            statistics.mean(reductions) if reductions else 0.0
        ),
        "median_top_risk_reduction": (
            statistics.median(reductions) if reductions else 0.0
        ),
        "success_at_0_01": (
            sum(value >= 0.01 for value in reductions) / len(reductions)
            if reductions
            else 0.0
        ),
        "success_at_0_05": (
            sum(value >= 0.05 for value in reductions) / len(reductions)
            if reductions
            else 0.0
        ),
        "threshold_crossing_denominator": len(at_risk),
        "threshold_crossing_rate": (
            sum(row.threshold_crossed for row in at_risk) / len(at_risk)
            if at_risk
            else 0.0
        ),
        "mean_top_utility_score": (
            statistics.mean(utilities) if utilities else 0.0
        ),
        "mean_selected_actions": (
            statistics.mean(row.selected_action_count for row in records)
            if records
            else 0.0
        ),
        "mean_selected_workload_minutes": (
            statistics.mean(row.selected_workload_minutes for row in records)
            if records
            else 0.0
        ),
        "unique_top_actions": len(action_frequency),
        "top_action_frequency": dict(sorted(action_frequency.items())),
        "top_action_share": (
            max(action_frequency.values(), default=0) / len(scored)
            if scored
            else 0.0
        ),
        "status_frequency": dict(sorted(statuses.items())),
        "claim_boundary": "MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT",
        "outcome_labels_used_for_ranking": False,
    }


def grouped_counterfactual_metrics(
    rows: Iterable[CounterfactualEvaluationRow],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[CounterfactualEvaluationRow]] = {}
    for row in rows:
        key = f"fold_{row.fold}:{row.stage}"
        groups.setdefault(key, []).append(row)
    return {
        key: aggregate_counterfactual_metrics(value)
        for key, value in sorted(groups.items())
    }


__all__ = [
    "CounterfactualEvaluationRow",
    "aggregate_counterfactual_metrics",
    "grouped_counterfactual_metrics",
]
