"""Immutable contracts for Phase 1 scientific labeling records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from typing import Any

from src.recommend_hybrid.common.policy_contracts import DatasetId
from src.recommend_hybrid.exceptions import ContractValidationError


UCI_STAGES = frozenset({"S0", "S1", "S2"})
OULAD_STAGES = frozenset({"EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75", "FINAL_EVALUATION"})


def _stable_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_stable_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _stable_value(value[key]) for key in sorted(value)}
    return value


class StableContract:
    def to_dict(self) -> dict[str, Any]:
        return _stable_value(asdict(self))


@dataclass(frozen=True)
class PredictionContext(StableContract):
    model_id: str
    prediction_authority: str
    dataset: DatasetId
    stage: str
    requested_cutoff: float
    predicted_class: int
    class_probabilities: tuple[float, ...]
    uncertainty: float
    checkpoint_lineage: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.model_id or not self.prediction_authority or not self.checkpoint_lineage:
            raise ContractValidationError("prediction identity, authority, and lineage are required")
        if not isfinite(self.requested_cutoff):
            raise ContractValidationError("requested cutoff must be finite")
        if not self.class_probabilities or any(
            not isfinite(value) or not 0 <= value <= 1 for value in self.class_probabilities
        ):
            raise ContractValidationError("class probabilities must be finite in [0,1]")
        if abs(sum(self.class_probabilities) - 1.0) > 1e-6:
            raise ContractValidationError("class probabilities must sum to one")
        if self.predicted_class not in range(len(self.class_probabilities)):
            raise ContractValidationError("predicted class is outside probability domain")
        if not isfinite(self.uncertainty) or self.uncertainty < 0:
            raise ContractValidationError("uncertainty must be finite and non-negative")


@dataclass(frozen=True)
class CandidateActionExample(StableContract):
    student_key: str
    action_id: str
    dataset: DatasetId
    stage: str
    requested_cutoff: float
    evidence_fields: tuple[str, ...]
    evidence_observation_end: float | None
    prediction: PredictionContext

    def __post_init__(self) -> None:
        if not self.student_key or not self.action_id:
            raise ContractValidationError("student and action identity are required")
        if self.prediction.dataset is not self.dataset:
            raise ContractValidationError("candidate and prediction datasets must match")
        if self.prediction.stage != self.stage or self.prediction.requested_cutoff != self.requested_cutoff:
            raise ContractValidationError("candidate and prediction stage/cutoff must match")
        if self.evidence_observation_end is not None and self.evidence_observation_end >= self.requested_cutoff:
            raise ContractValidationError("evidence must be strictly before requested cutoff")


@dataclass(frozen=True)
class SourceRecord(StableContract):
    source_id: str
    title: str
    organization_or_authors: str
    publication_year_or_version: str
    source_type: str
    url: str
    retrieved_at: str
    target_population: str
    supported_action_ids: tuple[str, ...]
    evidence_summary: str
    applicability_limits: str
    verification_status: str

    def __post_init__(self) -> None:
        required = (
            self.source_id,
            self.title,
            self.organization_or_authors,
            self.publication_year_or_version,
            self.source_type,
            self.url,
            self.retrieved_at,
            self.target_population,
            self.evidence_summary,
            self.applicability_limits,
            self.verification_status,
        )
        if any(not value for value in required):
            raise ContractValidationError("source metadata is incomplete")


@dataclass(frozen=True)
class ActionEvidenceMapping(StableContract):
    action_id: str
    display_name: str
    description: str
    target_problem: str
    supported_datasets: tuple[DatasetId, ...]
    supported_stages: tuple[str, ...]
    required_evidence: tuple[str, ...]
    prerequisites: tuple[str, ...]
    contraindications: tuple[str, ...]
    default_priority: str
    estimated_minutes: int
    human_review_required: bool
    evidence_source_ids: tuple[str, ...]
    claim_limitations: str
    status: str

    def __post_init__(self) -> None:
        if self.status not in {"EVIDENCE_MAPPED", "INSUFFICIENT_EVIDENCE"}:
            raise ContractValidationError("invalid action evidence status")
        if self.status == "EVIDENCE_MAPPED" and not self.evidence_source_ids:
            raise ContractValidationError("mapped action requires a source")
        if self.status == "INSUFFICIENT_EVIDENCE" and self.evidence_source_ids:
            raise ContractValidationError("insufficient-evidence action cannot cite a supporting source")
        if self.estimated_minutes <= 0:
            raise ContractValidationError("estimated minutes must be positive")


__all__ = [
    "ActionEvidenceMapping",
    "CandidateActionExample",
    "OULAD_STAGES",
    "PredictionContext",
    "SourceRecord",
    "StableContract",
    "UCI_STAGES",
]
