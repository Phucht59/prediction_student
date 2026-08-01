"""Immutable, validated contracts for the recommend_hybrid foundation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from math import isfinite
from typing import Any

from .exceptions import ContractValidationError


class Stage(str, Enum):
    EARLY_20 = "EARLY_20"
    EARLY_35 = "EARLY_35"
    MIDDLE_50 = "MIDDLE_50"
    LATE_75 = "LATE_75"
    FINAL_EVALUATION = "FINAL_EVALUATION"


class CandidateStatus(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE_STAGE = "INELIGIBLE_STAGE"
    MISSING_REQUIRED_EVIDENCE = "MISSING_REQUIRED_EVIDENCE"
    PREREQUISITE_NOT_MET = "PREREQUISITE_NOT_MET"
    CONTRAINDICATED = "CONTRAINDICATED"
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"


class ApprovalStatus(str, Enum):
    APPROVE = "APPROVE"
    PARTIAL = "PARTIAL"
    UNSURE = "UNSURE"
    REJECT = "REJECT"


class PlanStatus(str, Enum):
    APPROVED = "APPROVED"
    MODIFIED = "MODIFIED"
    REJECTED = "REJECTED"
    NEEDS_MORE_EVIDENCE = "NEEDS_MORE_EVIDENCE"


def _enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: _enum_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple | list):
        return [_enum_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _enum_value(item) for key, item in value.items()}
    return value


class SerializableContract:
    """Stable JSON-compatible representation for immutable contracts."""

    def to_dict(self) -> dict[str, Any]:
        return _enum_value(self)


@dataclass(frozen=True)
class CheckpointReference(SerializableContract):
    checkpoint_id: str
    path: str
    sha256: str
    fold: int
    seed: int

    def __post_init__(self) -> None:
        if not self.checkpoint_id or not self.path or len(self.sha256) != 64:
            raise ContractValidationError("invalid checkpoint reference")
        if self.fold < 0:
            raise ContractValidationError("fold must be non-negative")


@dataclass(frozen=True)
class PredictionContext(SerializableContract):
    student_key: str
    course_key: str
    stage: Stage
    cutoff_day: int
    predicted_class: int
    class_probabilities: tuple[float, float]
    confidence: float
    uncertainty: float
    seed_disagreement: float
    fold: int
    seeds: tuple[int, ...]
    checkpoint_references: tuple[CheckpointReference, ...]
    architecture_hash: str
    parameter_count: int
    confidence_source: str = "RAW_MAX_CLASS_PROBABILITY"

    def __post_init__(self) -> None:
        if not self.student_key or not self.course_key:
            raise ContractValidationError("student_key and course_key are required")
        if not isinstance(self.stage, Stage):
            raise ContractValidationError("stage must be a canonical Stage")
        if self.cutoff_day <= 0 or self.predicted_class not in (0, 1):
            raise ContractValidationError("invalid cutoff or predicted class")
        if len(self.class_probabilities) != 2:
            raise ContractValidationError("binary probabilities require two values")
        if any(not isfinite(v) or not 0.0 <= v <= 1.0 for v in self.class_probabilities):
            raise ContractValidationError("probabilities must be finite in [0, 1]")
        if abs(sum(self.class_probabilities) - 1.0) > 1e-6:
            raise ContractValidationError("probabilities must sum to one")
        if any(not isfinite(v) or v < 0.0 for v in (self.uncertainty, self.seed_disagreement)):
            raise ContractValidationError("uncertainty values must be finite and non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractValidationError("confidence must be in [0, 1]")
        if self.parameter_count <= 0 or len(self.architecture_hash) != 64:
            raise ContractValidationError("invalid frozen architecture metadata")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ContractValidationError("seeds must be non-empty and unique")


@dataclass(frozen=True)
class StudentRepresentation(SerializableContract):
    student_state_embedding: tuple[float, ...]
    student_state_dimension: int
    tabular_expert_embedding: tuple[float, ...]
    tabular_expert_dimension: int
    model_authority: str
    embedding_source: str
    dtype: str
    device: str

    def __post_init__(self) -> None:
        if self.student_state_dimension != 64 or len(self.student_state_embedding) != 64:
            raise ContractValidationError("student-state embedding must be 64-D")
        if self.tabular_expert_dimension != 32 or len(self.tabular_expert_embedding) != 32:
            raise ContractValidationError("tabular-expert embedding must be 32-D")
        if not self.model_authority or not self.embedding_source:
            raise ContractValidationError("embedding authority and source are required")
        if any(not isfinite(v) for v in (*self.student_state_embedding, *self.tabular_expert_embedding)):
            raise ContractValidationError("embeddings must be finite")


@dataclass(frozen=True)
class FeatureLineage(SerializableContract):
    feature: str
    source_table: str
    source_column: str
    aggregation: str
    observation_start: int | None
    observation_end: int | None
    cutoff_day: int
    missing_status: str

    def __post_init__(self) -> None:
        if not all((self.feature, self.source_table, self.source_column, self.aggregation)):
            raise ContractValidationError("complete feature lineage is required")
        if self.observation_end is not None and self.observation_end >= self.cutoff_day:
            raise ContractValidationError("lineage observation_end must be before cutoff")


@dataclass(frozen=True)
class ObservedLearningState(SerializableContract):
    activity_level: float | None
    inactivity_streak: int | None
    assessment_progress: float | None
    grade_trend: float | None
    course_progress: float
    recent_activity_trend: float | None
    available_evidence: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    feature_lineage: tuple[FeatureLineage, ...]
    cutoff_day: int
    stage: Stage
    total_activity: float | None = None
    recent_activity: float | None = None
    average_activity: float | None = None
    assessments_due: int | None = None
    assessments_completed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, Stage) or self.cutoff_day <= 0:
            raise ContractValidationError("invalid observed-state stage or cutoff")
        if not 0.0 <= self.course_progress <= 1.0:
            raise ContractValidationError("course_progress must be in [0, 1]")
        if set(self.available_evidence) & set(self.missing_evidence):
            raise ContractValidationError("evidence cannot be both available and missing")
        lineage_features = {item.feature for item in self.feature_lineage}
        expected = set(self.available_evidence) | set(self.missing_evidence)
        if not expected.issubset(lineage_features):
            raise ContractValidationError("every evidence field requires lineage")


@dataclass(frozen=True)
class CandidateAction(SerializableContract):
    action_id: str
    category: str
    title: str
    description: str
    weekly_minutes: int
    applicable_stages: tuple[Stage, ...]
    required_evidence: tuple[str, ...]
    prerequisites: tuple[str, ...]
    contraindications: tuple[str, ...]
    requires_human_review: bool
    success_criterion: str
    active: bool = True
    catalog_version: str = "recommend_hybrid_actions_v1"

    def __post_init__(self) -> None:
        if not self.action_id or not self.category or not self.title or not self.description:
            raise ContractValidationError("action identity and descriptions are required")
        if not 0 < self.weekly_minutes <= 180:
            raise ContractValidationError("weekly_minutes must be in [1, 180]")
        if Stage.FINAL_EVALUATION in self.applicable_stages:
            raise ContractValidationError("interventions cannot apply at FINAL_EVALUATION")
        if not self.success_criterion:
            raise ContractValidationError("success criterion is required")


@dataclass(frozen=True)
class CandidateEvaluation(SerializableContract):
    action: CandidateAction
    status: CandidateStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.reason_codes:
            raise ContractValidationError("candidate eligibility needs reason codes")


@dataclass(frozen=True)
class ExpertCase(SerializableContract):
    case_id: str
    prediction_context: PredictionContext
    observed_state: ObservedLearningState
    candidate_actions: tuple[CandidateEvaluation, ...]
    blinding_metadata: tuple[tuple[str, str], ...]
    export_version: str

    def __post_init__(self) -> None:
        if not self.case_id or not self.export_version:
            raise ContractValidationError("case identity and export version are required")


@dataclass(frozen=True)
class ExpertActionRating(SerializableContract):
    case_id: str
    action_id: str
    expert_id: str
    relevance_score: int
    approval_status: ApprovalStatus
    missing_action: bool
    safety_concern: bool
    escalation_required: bool
    reason_support: str
    comment: str

    def __post_init__(self) -> None:
        if self.relevance_score not in (-1, 0, 1, 2, 3):
            raise ContractValidationError("invalid relevance score")
        if self.relevance_score == -1 and not self.safety_concern:
            raise ContractValidationError("unsafe rating must set safety_concern")
        if not all((self.case_id, self.action_id, self.expert_id, self.reason_support)):
            raise ContractValidationError("required expert rating field is empty")


@dataclass(frozen=True)
class ExpertCaseReview(SerializableContract):
    case_id: str
    expert_id: str
    plan_score: int
    overall_status: PlanStatus
    missing_actions: tuple[str, ...]
    safety_concerns: tuple[str, ...]
    review_comment: str

    def __post_init__(self) -> None:
        if not self.case_id or not self.expert_id or not 0 <= self.plan_score <= 3:
            raise ContractValidationError("invalid expert case review")


def contract_dict(contract: SerializableContract) -> dict[str, Any]:
    """Return a deterministic JSON-compatible contract representation."""
    return contract.to_dict()


__all__ = [
    "ApprovalStatus",
    "CandidateAction",
    "CandidateEvaluation",
    "CandidateStatus",
    "CheckpointReference",
    "ExpertActionRating",
    "ExpertCase",
    "ExpertCaseReview",
    "FeatureLineage",
    "ObservedLearningState",
    "PlanStatus",
    "PredictionContext",
    "Stage",
    "StudentRepresentation",
    "contract_dict",
]
