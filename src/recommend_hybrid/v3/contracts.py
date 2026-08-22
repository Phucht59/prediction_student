"""Recommendation V contracts. Hybrid CNN–BiLSTM PredictionResult is the only prediction input."""

from __future__ import annotations

from dataclasses import dataclass
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
STAGE_TO_PREDICTION = {stage: key for key, stage in PREDICTION_TO_STAGE.items()}
STAGE_FRACTION = {
    Stage.EARLY_20: 0.20,
    Stage.EARLY_35: 0.35,
    Stage.MIDDLE_50: 0.50,
    Stage.LATE_75: 0.75,
}
NON_INTERVENTION_PREDICTION_STATES = frozenset({"100pct", "FINAL-100", "100"})


class CanonicalAction(str, Enum):
    ASSESSMENT_COMPLETION = "ASSESSMENT_COMPLETION"
    RECOVER_ENGAGEMENT = "RECOVER_ENGAGEMENT"
    STUDY_REGULARITY = "STUDY_REGULARITY"
    TARGETED_CONTENT_REVIEW = "TARGETED_CONTENT_REVIEW"
    QUIZ_RETRIEVAL_PRACTICE = "QUIZ_RETRIEVAL_PRACTICE"


class RiskRoute(str, Enum):
    NO_AUTOMATIC = "NO_AUTOMATIC"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    PROCESS = "PROCESS"


class RouteStatus(str, Enum):
    RECOMMEND = "RECOMMEND"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    NO_FEASIBLE_ACTION = "NO_FEASIBLE_ACTION"


@dataclass(frozen=True)
class RiskThresholds:
    """Recommendation margins around the frozen C0 operating threshold t."""

    maximum_automatic_uncertainty: float
    minimum_risk_margin: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.maximum_automatic_uncertainty <= 1.0:
            raise ValueError("maximum_automatic_uncertainty must be in [0, 1]")
        if not isfinite(self.minimum_risk_margin) or self.minimum_risk_margin < 0:
            raise ValueError("minimum_risk_margin must be finite and >= 0")


@dataclass(frozen=True)
class SafetyThresholds:
    minimum_top1_score: float
    minimum_top1_margin: float
    maximum_uncertainty: float

    def __post_init__(self) -> None:
        values = (self.minimum_top1_score, self.minimum_top1_margin, self.maximum_uncertainty)
        if any(not isfinite(v) or not 0.0 <= v <= 1.0 for v in values):
            raise ValueError("safety thresholds must be finite in [0, 1]")


@dataclass(frozen=True)
class RecommendationFeatures:
    student_key: str
    course_key: str
    record_id: str
    stage: Stage
    cutoff_day: int
    risk_probability: float
    predicted_risk: int
    prediction_threshold: float
    uncertainty: float
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
    contraindications: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.student_key or not self.course_key or not self.record_id:
            raise ValueError("identity fields are required")
        if self.cutoff_day <= 0:
            raise ValueError("cutoff_day must be positive")
        if not isinstance(self.stage, Stage):
            raise ValueError("stage must be a V3 intervention Stage")
        if not 0.0 <= self.risk_probability <= 1.0:
            raise ValueError("risk_probability must be in [0, 1]")
        if int(self.predicted_risk) not in {0, 1}:
            raise ValueError("predicted_risk must be binary")
        if not 0.0 < self.prediction_threshold < 1.0:
            raise ValueError("prediction_threshold must be in (0, 1)")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must be in [0, 1]")
        if not 0.0 <= self.course_progress <= 1.0:
            raise ValueError("course_progress must be in [0, 1]")

    @property
    def risk_margin(self) -> float:
        return float(self.risk_probability - self.prediction_threshold)


@dataclass(frozen=True)
class FeasibilityResult:
    action: CanonicalAction
    eligible: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class ActionScore:
    action: CanonicalAction
    score: float
    explanation: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("action score must be in [0, 1]")


@dataclass(frozen=True)
class StructuredLearningPlan:
    action: str
    reason: str
    observed_evidence: tuple[str, ...]
    what_to_do: str
    suggested_duration_days: int
    suggested_frequency: str
    measurable_target: str
    reevaluation_time_days: int
    safety_note: str
    claim_boundary: str = (
        "This is a deterministic support plan, not a causal treatment effect."
    )


@dataclass(frozen=True)
class RecommendationDecision:
    student_key: str
    course_key: str
    stage: Stage
    risk_route: RiskRoute
    route: RouteStatus
    ranked_actions: tuple[ActionScore, ...]
    plan: StructuredLearningPlan | None
    reason_codes: tuple[str, ...]
    protocol_version: str = "recommend_hybrid_v3_c0"

    def __post_init__(self) -> None:
        if not self.reason_codes:
            raise ValueError("recommendation decision requires reason codes")
        if self.route is RouteStatus.RECOMMEND:
            if len(self.ranked_actions) != 1:
                raise ValueError("RECOMMEND must emit exactly Top-1")
            if self.plan is None:
                raise ValueError("RECOMMEND requires a personalized plan")
        if self.route is RouteStatus.HUMAN_REVIEW and not self.ranked_actions:
            raise ValueError("HUMAN_REVIEW must expose ranked suggestions")
        if self.route in {RouteStatus.INSUFFICIENT_EVIDENCE, RouteStatus.NO_FEASIBLE_ACTION}:
            if self.ranked_actions:
                raise ValueError("abstain routes cannot emit actions")
            if self.plan is not None:
                raise ValueError("abstain routes cannot emit a plan")


def map_prediction_state(stage_or_endpoint: str) -> Stage:
    if stage_or_endpoint in NON_INTERVENTION_PREDICTION_STATES:
        raise ValueError("OULAD 100pct is not an intervention stage")
    if stage_or_endpoint not in PREDICTION_TO_STAGE:
        raise ValueError(f"unknown prediction state for V3: {stage_or_endpoint}")
    return PREDICTION_TO_STAGE[stage_or_endpoint]
