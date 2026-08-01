"""Immutable contracts shared by the UCI and OULAD policy branches."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from math import isfinite
from typing import Any

from src.recommend_hybrid.exceptions import ContractValidationError


class DatasetId(str, Enum):
    STUDENT_MAT = "student_mat"
    STUDENT_POR = "student_por"
    OULAD = "oulad"


class Priority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class EvidenceSeverity(str, Enum):
    MISSING = "MISSING"
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EvidenceAvailability(str, Enum):
    AVAILABLE = "AVAILABLE"
    MISSING = "MISSING"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class EligibilityStatus(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE_STAGE = "INELIGIBLE_STAGE"
    MISSING_REQUIRED_EVIDENCE = "MISSING_REQUIRED_EVIDENCE"
    PREREQUISITE_NOT_MET = "PREREQUISITE_NOT_MET"
    CONTRAINDICATED = "CONTRAINDICATED"
    REQUIRES_HUMAN_CONTACT = "REQUIRES_HUMAN_CONTACT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class AutomationStatus(str, Enum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    ABSTAIN = "ABSTAIN"
    EVALUATION_ONLY = "EVALUATION_ONLY"


class RoutingStatus(str, Enum):
    ROUTED = "ROUTED"
    NO_VALIDATED_PREDICTION_ANCHOR = "NO_VALIDATED_PREDICTION_ANCHOR"
    INSUFFICIENT_STAGE_EVIDENCE = "INSUFFICIENT_STAGE_EVIDENCE"
    EVALUATION_ONLY = "EVALUATION_ONLY"


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


class PolicyContract:
    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclass(frozen=True)
class PolicyPredictionContext(PolicyContract):
    dataset_id: DatasetId
    predicted_class: int
    class_probabilities: tuple[float, ...]
    confidence: float
    uncertainty: float
    seed_disagreement: float
    checkpoint_lineage: tuple[str, ...]
    architecture_authority: str
    representation_lineage: tuple[str, ...] = ()
    embedding_dimensions: tuple[int, ...] = (64, 32)

    def __post_init__(self) -> None:
        if not self.class_probabilities or any(
            not isfinite(value) or not 0 <= value <= 1
            for value in self.class_probabilities
        ):
            raise ContractValidationError("prediction probabilities must be finite in [0,1]")
        if abs(sum(self.class_probabilities) - 1.0) > 1e-6:
            raise ContractValidationError("prediction probabilities must sum to one")
        if self.predicted_class < 0 or self.predicted_class >= len(self.class_probabilities):
            raise ContractValidationError("predicted class is outside probability domain")
        if not 0 <= self.confidence <= 1 or self.uncertainty < 0 or self.seed_disagreement < 0:
            raise ContractValidationError("invalid confidence or uncertainty")
        if not self.checkpoint_lineage or not self.architecture_authority:
            raise ContractValidationError("prediction authority lineage is required")
        if self.embedding_dimensions != (64, 32):
            raise ContractValidationError("locked representation dimensions must be 64 and 32")


@dataclass(frozen=True)
class PredictionAnchor(PolicyContract):
    requested_cutoff: float
    anchor_stage: str | None
    anchor_cutoff: float | None
    prediction_age: float | None
    routing_status: RoutingStatus
    checkpoint_lineage: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.anchor_cutoff is not None and self.anchor_cutoff > self.requested_cutoff:
            raise ContractValidationError("prediction anchor cannot be in the future")
        if self.anchor_cutoff is None and self.prediction_age is not None:
            raise ContractValidationError("prediction age requires an anchor")
        if (
            self.anchor_cutoff is not None
            and self.prediction_age is not None
            and self.prediction_age != self.requested_cutoff - self.anchor_cutoff
        ):
            raise ContractValidationError("prediction age is inconsistent")


@dataclass(frozen=True)
class EvidenceItem(PolicyContract):
    evidence_id: str
    feature_name: str
    observed_value: float | int | str | bool | None
    severity: EvidenceSeverity
    availability: EvidenceAvailability
    source_lineage: str
    observation_end: float | None
    cutoff: float

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.feature_name or not self.source_lineage:
            raise ContractValidationError("evidence identity and source lineage are required")
        if self.observation_end is not None and self.observation_end > self.cutoff:
            raise ContractValidationError("evidence observation exceeds requested cutoff")
        if self.availability is EvidenceAvailability.AVAILABLE and self.observed_value is None:
            raise ContractValidationError("available evidence needs an observed value")
        if self.availability is not EvidenceAvailability.AVAILABLE and self.severity is not EvidenceSeverity.MISSING:
            raise ContractValidationError("unavailable evidence must have MISSING severity")


@dataclass(frozen=True)
class RecommendationRequest(PolicyContract):
    dataset_id: DatasetId
    student_key: str
    course_key: str
    requested_cutoff: float
    available_assessments: tuple[str, ...]
    prediction_context: PolicyPredictionContext
    observed_state: tuple[EvidenceItem, ...]

    def __post_init__(self) -> None:
        if not self.student_key or not self.course_key:
            raise ContractValidationError("student and course keys are required")
        names = [item.feature_name for item in self.observed_state]
        if len(names) != len(set(names)):
            raise ContractValidationError("observed evidence feature names must be unique")
        if any(item.cutoff != self.requested_cutoff for item in self.observed_state):
            raise ContractValidationError("evidence cutoff must match request cutoff")


@dataclass(frozen=True)
class PolicyActionDecision(PolicyContract):
    action_id: str
    eligibility_status: EligibilityStatus
    priority: Priority
    reason_codes: tuple[str, ...]
    supporting_evidence: tuple[EvidenceItem, ...]
    missing_evidence: tuple[str, ...]
    requires_human_contact: bool
    policy_version: str

    def __post_init__(self) -> None:
        eligible = {
            EligibilityStatus.ELIGIBLE,
            EligibilityStatus.REQUIRES_HUMAN_CONTACT,
        }
        if self.eligibility_status in eligible and self.priority is Priority.NOT_APPLICABLE:
            raise ContractValidationError("eligible action needs ordinal priority")
        if self.eligibility_status not in eligible and self.priority is not Priority.NOT_APPLICABLE:
            raise ContractValidationError("ineligible action cannot have a priority")
        if self.supporting_evidence and not self.reason_codes:
            raise ContractValidationError("supporting evidence requires reason codes")
        if self.requires_human_contact != (
            self.eligibility_status is EligibilityStatus.REQUIRES_HUMAN_CONTACT
        ):
            raise ContractValidationError("human-contact status is inconsistent")


@dataclass(frozen=True)
class ActionExplanation(PolicyContract):
    action: str
    observed_evidence: tuple[str, ...]
    prediction_context: str
    reason: str
    limitation: str


@dataclass(frozen=True)
class PolicyRecommendationResult(PolicyContract):
    dataset_id: DatasetId
    student_key: str
    requested_cutoff: float
    prediction_anchor: PredictionAnchor
    automation_status: AutomationStatus
    action_decisions: tuple[PolicyActionDecision, ...]
    explanation: tuple[ActionExplanation, ...]
    abstention_reasons: tuple[str, ...]
    policy_version: str

    def __post_init__(self) -> None:
        explained = {item.action for item in self.explanation}
        eligible = {
            item.action_id
            for item in self.action_decisions
            if item.eligibility_status
            in {EligibilityStatus.ELIGIBLE, EligibilityStatus.REQUIRES_HUMAN_CONTACT}
        }
        if explained != eligible:
            raise ContractValidationError("every eligible action requires one faithful explanation")
        if self.automation_status in {AutomationStatus.ABSTAIN, AutomationStatus.EVALUATION_ONLY} and eligible:
            raise ContractValidationError("abstention/evaluation result cannot expose eligible actions")


__all__ = [
    "ActionExplanation",
    "AutomationStatus",
    "DatasetId",
    "EligibilityStatus",
    "EvidenceAvailability",
    "EvidenceItem",
    "EvidenceSeverity",
    "PolicyActionDecision",
    "PolicyPredictionContext",
    "PolicyRecommendationResult",
    "PredictionAnchor",
    "Priority",
    "RecommendationRequest",
    "RoutingStatus",
]
