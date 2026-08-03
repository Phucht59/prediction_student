"""Observational trajectory checks kept separate from action ranking."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable

from src.recommend_hybrid.exceptions import ContractValidationError

HISTORICAL_CLAIM_BOUNDARY = (
    "OBSERVATIONAL_ASSOCIATION_ONLY_NOT_CAUSAL_EFFECT"
)


@dataclass(frozen=True)
class HistoricalTrajectoryRow:
    student_key: str
    course_key: str
    stage: str
    action_id: str
    behavior_aligned: bool | None
    next_stage_risk: float | None
    favorable_final_outcome: bool

    def __post_init__(self) -> None:
        if not all(
            (self.student_key, self.course_key, self.stage, self.action_id)
        ):
            raise ContractValidationError(
                "historical trajectory identity is required"
            )
        if self.next_stage_risk is not None and (
            not isfinite(self.next_stage_risk)
            or not 0.0 <= self.next_stage_risk <= 1.0
        ):
            raise ContractValidationError(
                "next_stage_risk must be finite in [0, 1]"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "student_key": self.student_key,
            "course_key": self.course_key,
            "stage": self.stage,
            "action_id": self.action_id,
            "behavior_aligned": self.behavior_aligned,
            "next_stage_risk": self.next_stage_risk,
            "favorable_final_outcome": self.favorable_final_outcome,
        }


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_historical_metrics(
    rows: Iterable[HistoricalTrajectoryRow],
) -> dict[str, Any]:
    records = tuple(rows)
    evaluable = tuple(
        row for row in records if row.behavior_aligned is not None
    )
    aligned = tuple(row for row in evaluable if row.behavior_aligned)
    not_aligned = tuple(
        row for row in evaluable if not row.behavior_aligned
    )
    aligned_risk = [
        float(row.next_stage_risk)
        for row in aligned
        if row.next_stage_risk is not None
    ]
    not_aligned_risk = [
        float(row.next_stage_risk)
        for row in not_aligned
        if row.next_stage_risk is not None
    ]
    aligned_outcome = [row.favorable_final_outcome for row in aligned]
    not_aligned_outcome = [
        row.favorable_final_outcome for row in not_aligned
    ]
    aligned_mean_risk = _mean(aligned_risk)
    not_aligned_mean_risk = _mean(not_aligned_risk)
    aligned_outcome_rate = _rate(aligned_outcome)
    not_aligned_outcome_rate = _rate(not_aligned_outcome)
    return {
        "record_count": len(records),
        "behavior_evaluable_count": len(evaluable),
        "behavior_evaluable_rate": (
            len(evaluable) / len(records) if records else 0.0
        ),
        "behavior_aligned_count": len(aligned),
        "behavior_alignment_rate": (
            len(aligned) / len(evaluable) if evaluable else 0.0
        ),
        "aligned_mean_next_stage_risk": aligned_mean_risk,
        "not_aligned_mean_next_stage_risk": not_aligned_mean_risk,
        "observed_next_stage_risk_difference": (
            not_aligned_mean_risk - aligned_mean_risk
            if aligned_mean_risk is not None
            and not_aligned_mean_risk is not None
            else None
        ),
        "aligned_favorable_outcome_rate": aligned_outcome_rate,
        "not_aligned_favorable_outcome_rate": not_aligned_outcome_rate,
        "observed_favorable_outcome_difference": (
            aligned_outcome_rate - not_aligned_outcome_rate
            if aligned_outcome_rate is not None
            and not_aligned_outcome_rate is not None
            else None
        ),
        "claim_boundary": HISTORICAL_CLAIM_BOUNDARY,
        "used_for_action_ranking": False,
    }


__all__ = [
    "HISTORICAL_CLAIM_BOUNDARY",
    "HistoricalTrajectoryRow",
    "aggregate_historical_metrics",
]
