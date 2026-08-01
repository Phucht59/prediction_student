from dataclasses import replace

import pytest

from src.recommend_hybrid.common.policy_contracts import DatasetId
from src.recommend_hybrid.exceptions import ContractValidationError
from src.recommend_hybrid.weak_supervision.contracts import (
    CandidateActionExample,
    PredictionContext,
)
from src.recommend_hybrid.weak_supervision.validation import validate_candidate


def _prediction(cutoff: float = 20, stage: str = "EARLY_20") -> PredictionContext:
    return PredictionContext(
        model_id="h1_tabular_residual_oulad",
        prediction_authority="RECOMMEND_HYBRID_MODEL_AUTHORITY",
        dataset=DatasetId.OULAD,
        stage=stage,
        requested_cutoff=cutoff,
        predicted_class=1,
        class_probabilities=(0.2, 0.8),
        uncertainty=0.3,
        checkpoint_lineage=("frozen-oulad-anchor",),
    )


def _candidate(cutoff: float = 20, stage: str = "EARLY_20") -> CandidateActionExample:
    return CandidateActionExample(
        student_key="opaque-student",
        action_id="PROGRESS_MONITORING",
        dataset=DatasetId.OULAD,
        stage=stage,
        requested_cutoff=cutoff,
        evidence_fields=("activity_level",),
        evidence_observation_end=cutoff - 1,
        prediction=_prediction(cutoff, stage),
    )


def test_oulad_post_cutoff_evidence_is_rejected() -> None:
    with pytest.raises(ContractValidationError, match="strictly before"):
        replace(_candidate(), evidence_observation_end=20)


def test_oulad_before_20_is_rejected() -> None:
    candidate = _candidate(19, "EARLY_20")
    with pytest.raises(ContractValidationError, match="before 20"):
        validate_candidate(candidate)


def test_final_is_evaluation_only() -> None:
    candidate = _candidate(100, "FINAL_EVALUATION")
    with pytest.raises(ContractValidationError, match="evaluation-only"):
        validate_candidate(candidate)


def test_uci_g3_is_rejected() -> None:
    prediction = replace(
        _prediction(2, "S2"),
        model_id="cnn_bilstm_por",
        prediction_authority="FINAL_THESIS_MODEL_AUTHORITY",
        dataset=DatasetId.STUDENT_POR,
        class_probabilities=(0.2, 0.6, 0.2),
    )
    candidate = CandidateActionExample(
        student_key="opaque-student",
        action_id="TARGETED_REVISION",
        dataset=DatasetId.STUDENT_POR,
        stage="S2",
        requested_cutoff=2,
        evidence_fields=("G1", "G2", "G3"),
        evidence_observation_end=1,
        prediction=prediction,
    )
    with pytest.raises(ContractValidationError, match="prohibited"):
        validate_candidate(candidate)
