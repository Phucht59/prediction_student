"""Locked recommendation serving contracts. Hybrid PredictionResult is the only model input."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite


class Stage(str, Enum):
    EARLY_20 = "EARLY_20"
    EARLY_35 = "EARLY_35"
    MIDDLE_50 = "MIDDLE_50"
    LATE_75 = "LATE_75"


PREDICTION_TO_STAGE = {
    "20pct": Stage.EARLY_20,
    "35pct": Stage.EARLY_35,
    "50pct": Stage.MIDDLE_50,
    "75pct": Stage.LATE_75,
}
STAGE_FRACTION = {
    Stage.EARLY_20: 0.20,
    Stage.EARLY_35: 0.35,
    Stage.MIDDLE_50: 0.50,
    Stage.LATE_75: 0.75,
}
NON_INTERVENTION = frozenset({"100pct", "FINAL-100", "100"})
PROTOCOL_VERSION = "recommend_hybrid_serving_persist_v1"
K_FRAC_PRIMARY = 0.10
PERSIST_WINDOW_DAYS = 14
UNCERTAINTY_AUTO_MAX = 0.70
ACTIVE_DAY_ENGAGE = 0.20
STREAK_ENGAGE = 7


class PersistLabel(str, Enum):
    ASSESS = "ASSESS"
    ENGAGE = "ENGAGE"
    COUNSEL = "COUNSEL"


class RouteStatus(str, Enum):
    ACTION = "ACTION"
    QUEUE = "QUEUE"
    COUNSEL = "COUNSEL"
    OUT_OF_BUDGET = "OUT_OF_BUDGET"


FEATURE_COLUMNS: tuple[str, ...] = (
    "risk_probability",
    "uncertainty",
    "course_progress",
    "missing_assessment_count",
    "due_soon_count",
    "completion_rate",
    "inactivity_streak",
    "active_day_rate",
    "recent_activity_trend",
    "regularity_score",
    "content_coverage",
    "quiz_activity",
    "assessments_due",
    "remaining_count",
    "time_to_deadline_days",
    "vle_access_available",
    "quiz_available",
    "study_material_available",
)


def map_prediction_state(stage_or_endpoint: str) -> Stage:
    if stage_or_endpoint in NON_INTERVENTION:
        raise ValueError("OULAD 100pct is not an intervention stage")
    if stage_or_endpoint not in PREDICTION_TO_STAGE:
        raise ValueError(f"unknown prediction state: {stage_or_endpoint}")
    return PREDICTION_TO_STAGE[stage_or_endpoint]


@dataclass(frozen=True)
class PathwayItem:
    assessment_id: int
    deadline_day: int
    days_until_due: int


@dataclass(frozen=True)
class RecommendationDecision:
    student_key: str
    course_key: str
    stage: Stage
    route: RouteStatus
    action: PersistLabel
    score: float
    reason_codes: tuple[str, ...]
    pathway: tuple[PathwayItem, ...] = ()
    protocol_version: str = PROTOCOL_VERSION
    in_worklist: bool = False
    rank_in_cohort: int | None = None
    cohort_size: int | None = None

    def __post_init__(self) -> None:
        if not self.reason_codes:
            raise ValueError("reason_codes required")
        if self.route is RouteStatus.OUT_OF_BUDGET and self.action is not PersistLabel.COUNSEL:
            object.__setattr__(self, "action", PersistLabel.COUNSEL)
        if not isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be in [0, 1]")


__all__ = [
    "ACTIVE_DAY_ENGAGE",
    "FEATURE_COLUMNS",
    "K_FRAC_PRIMARY",
    "NON_INTERVENTION",
    "PERSIST_WINDOW_DAYS",
    "PREDICTION_TO_STAGE",
    "PROTOCOL_VERSION",
    "PathwayItem",
    "PersistLabel",
    "RecommendationDecision",
    "RouteStatus",
    "STAGE_FRACTION",
    "STREAK_ENGAGE",
    "Stage",
    "UNCERTAINTY_AUTO_MAX",
    "map_prediction_state",
]
