"""Scientific labeling authority, temporal-safety, and registry validation."""

from __future__ import annotations

from collections.abc import Iterable

from src.recommend_hybrid.common.policy_contracts import DatasetId
from src.recommend_hybrid.exceptions import ContractValidationError

from .contracts import (
    OULAD_STAGES,
    UCI_STAGES,
    ActionEvidenceMapping,
    CandidateActionExample,
    PredictionContext,
    SourceRecord,
)

PREDICTION_AUTHORITIES = {
    DatasetId.STUDENT_MAT: ("cnn_bilstm_mat", "FINAL_THESIS_MODEL_AUTHORITY"),
    DatasetId.STUDENT_POR: ("cnn_bilstm_por", "FINAL_THESIS_MODEL_AUTHORITY"),
    DatasetId.OULAD: ("h1_tabular_residual_oulad", "RECOMMEND_HYBRID_MODEL_AUTHORITY"),
}
PROHIBITED_CANDIDATE_FIELDS = frozenset(
    {"G3", "age_band", "disability", "gender", "region", "imd_band", "final_result", "target", "outer_label", "date_unregistration"}
)


def validate_prediction_authority(context: PredictionContext) -> None:
    if (context.model_id, context.prediction_authority) != PREDICTION_AUTHORITIES[context.dataset]:
        raise ContractValidationError("prediction context is not an authorized Hybrid CNN-BiLSTM authority")


def validate_candidate(candidate: CandidateActionExample) -> None:
    validate_prediction_authority(candidate.prediction)
    if set(candidate.evidence_fields) & PROHIBITED_CANDIDATE_FIELDS:
        raise ContractValidationError("candidate contains a prohibited or sensitive field")
    if candidate.dataset in {DatasetId.STUDENT_MAT, DatasetId.STUDENT_POR}:
        if candidate.stage not in UCI_STAGES:
            raise ContractValidationError("invalid UCI stage")
    else:
        if candidate.stage not in OULAD_STAGES:
            raise ContractValidationError("invalid OULAD stage")
        if candidate.requested_cutoff < 20:
            raise ContractValidationError("OULAD requests before 20 percent have no prediction anchor")
        if candidate.stage == "FINAL_EVALUATION":
            raise ContractValidationError("FINAL is evaluation-only and cannot generate intervention candidates")


def validate_registries(
    sources: Iterable[SourceRecord], actions: Iterable[ActionEvidenceMapping]
) -> None:
    sources = tuple(sources)
    actions = tuple(actions)
    source_ids = [record.source_id for record in sources]
    action_ids = [record.action_id for record in actions]
    if len(source_ids) != len(set(source_ids)) or len(action_ids) != len(set(action_ids)):
        raise ContractValidationError("registry IDs must be unique")
    known_sources = set(source_ids)
    known_actions = set(action_ids)
    for action in actions:
        if set(action.evidence_source_ids) - known_sources:
            raise ContractValidationError("action references an unknown source")
    for source in sources:
        if set(source.supported_action_ids) - known_actions:
            raise ContractValidationError("source references an unknown action")


__all__ = [
    "PREDICTION_AUTHORITIES",
    "PROHIBITED_CANDIDATE_FIELDS",
    "validate_candidate",
    "validate_prediction_authority",
    "validate_registries",
]
