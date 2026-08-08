"""Immutable contracts for the explainable recommendation V2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from src.recommend_hybrid.contracts import Stage


class CanonicalAction(str, Enum):
    ASSESSMENT_COMPLETION = "ASSESSMENT_COMPLETION"
    RECOVER_ENGAGEMENT = "RECOVER_ENGAGEMENT"
    STUDY_REGULARITY = "STUDY_REGULARITY"
    TARGETED_CONTENT_REVIEW = "TARGETED_CONTENT_REVIEW"
    QUIZ_RETRIEVAL_PRACTICE = "QUIZ_RETRIEVAL_PRACTICE"


class RiskBand(str, Enum):
    LOW = "LOW"
    BORDERLINE = "BORDERLINE"
    HIGH = "HIGH"


class RouteStatus(str, Enum):
    RECOMMEND = "RECOMMEND"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    NO_FEASIBLE_ACTION = "NO_FEASIBLE_ACTION"


@dataclass(frozen=True)
class RiskThresholds:
    low: float
    high: float
    maximum_automatic_uncertainty: float
    maximum_seed_disagreement: float

    def __post_init__(self) -> None:
        values = (
            self.low,
            self.high,
            self.maximum_automatic_uncertainty,
            self.maximum_seed_disagreement,
        )
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("risk thresholds must be finite values in [0, 1]")
        if self.low >= self.high:
            raise ValueError("low risk threshold must be below high risk threshold")


@dataclass(frozen=True)
class RecommendationFeatures:
    student_key: str
    course_key: str
    stage: Stage
    cutoff_day: int
    risk_probability: float
    hybrid_uncertainty: float
    seed_disagreement: float | None
    course_progress: float
    assessment_progress: float | None = None
    assessments_due: int | None = None
    missing_assessment_count: int | None = None
    due_soon_count: int | None = None
    completion_rate: float | None = None
    assessment_window_open: bool | None = None
    time_to_deadline_days: int | None = None
    inactivity_streak: int | None = None
    active_day_rate: float | None = None
    recent_activity_trend: float | None = None
    regularity_score: float | None = None
    content_coverage: float | None = None
    knowledge_gap_evidence: bool | None = None
    quiz_activity: float | None = None
    quiz_available: bool | None = None
    vle_access_available: bool | None = None
    study_material_available: bool | None = None
    label_conflict: float = 0.0
    ood_score: float = 0.0
    available_evidence: frozenset[str] = frozenset()
    contraindications: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.student_key or not self.course_key or self.cutoff_day <= 0:
            raise ValueError("student, course, and positive cutoff are required")
        if not isinstance(self.stage, Stage) or self.stage is Stage.FINAL_EVALUATION:
            raise ValueError("recommendations require a validated intervention stage")
        unit_interval = (
            self.risk_probability,
            self.hybrid_uncertainty,
            self.course_progress,
            self.label_conflict,
            self.ood_score,
        )
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in unit_interval):
            raise ValueError("probability-like fields must be finite values in [0, 1]")
        if self.seed_disagreement is not None and (
            not isfinite(self.seed_disagreement)
            or not 0.0 <= self.seed_disagreement <= 1.0
        ):
            raise ValueError("seed_disagreement must be unavailable or in [0, 1]")
        optional_unit_interval = (
            self.assessment_progress,
            self.active_day_rate,
            self.regularity_score,
            self.content_coverage,
        )
        if any(
            value is not None and (not isfinite(value) or not 0.0 <= value <= 1.0)
            for value in optional_unit_interval
        ):
            raise ValueError("normalized optional features must be in [0, 1]")
        if self.assessments_due is not None and self.assessments_due < 0:
            raise ValueError("assessments_due must be non-negative")
        if (
            self.missing_assessment_count is not None
            and self.missing_assessment_count < 0
        ):
            raise ValueError("missing_assessment_count must be non-negative")
        if self.due_soon_count is not None and self.due_soon_count < 0:
            raise ValueError("due_soon_count must be non-negative")
        if self.completion_rate is not None and (
            not isfinite(self.completion_rate)
            or not 0.0 <= self.completion_rate <= 1.0
        ):
            raise ValueError("completion_rate must be in [0, 1]")
        if self.inactivity_streak is not None and self.inactivity_streak < 0:
            raise ValueError("inactivity_streak must be non-negative")


@dataclass(frozen=True)
class FeasibilityResult:
    action: CanonicalAction
    eligible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.reason_codes:
            raise ValueError("feasibility result requires reason codes")


@dataclass(frozen=True)
class ActionScore:
    action: CanonicalAction
    score: float
    explanation: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("action score must be finite and calibrated to [0, 1]")


@dataclass(frozen=True)
class SafetyThresholds:
    minimum_top1_score: float
    minimum_top1_margin: float
    maximum_hybrid_uncertainty: float
    maximum_seed_disagreement: float | None
    maximum_label_conflict: float
    maximum_ood_score: float

    def __post_init__(self) -> None:
        values = (
            self.minimum_top1_score,
            self.minimum_top1_margin,
            self.maximum_hybrid_uncertainty,
            self.maximum_label_conflict,
            self.maximum_ood_score,
        )
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("safety thresholds must be finite values in [0, 1]")
        if self.maximum_seed_disagreement is not None and (
            not isfinite(self.maximum_seed_disagreement)
            or not 0.0 <= self.maximum_seed_disagreement <= 1.0
        ):
            raise ValueError(
                "maximum_seed_disagreement must be unavailable or in [0, 1]"
            )


@dataclass(frozen=True)
class RecommendationDecision:
    student_key: str
    course_key: str
    stage: Stage
    risk_band: RiskBand
    route: RouteStatus
    ranked_actions: tuple[ActionScore, ...]
    reason_codes: tuple[str, ...]
    protocol_version: str = "recommend_hybrid_explainable_v2"
    runtime_authorized: bool = False

    def __post_init__(self) -> None:
        if not self.reason_codes:
            raise ValueError("recommendation decision requires reason codes")
        if self.runtime_authorized:
            raise ValueError("V2 is offline-only until prospective evidence exists")
        if self.route is RouteStatus.RECOMMEND and not self.ranked_actions:
            raise ValueError("RECOMMEND requires at least one ranked action")
        if self.route in {
            RouteStatus.INSUFFICIENT_EVIDENCE,
            RouteStatus.NO_FEASIBLE_ACTION,
        } and self.ranked_actions:
            raise ValueError(
                "insufficient-evidence and no-feasible-action routes cannot emit actions"
            )


__all__ = [
    "ActionScore",
    "CanonicalAction",
    "FeasibilityResult",
    "RecommendationDecision",
    "RecommendationFeatures",
    "RiskBand",
    "RiskThresholds",
    "RouteStatus",
    "SafetyThresholds",
]
