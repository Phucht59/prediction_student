import json

import pytest

from src.recommend_hybrid.contracts import PredictionContext, Stage, StudentRepresentation
from src.recommend_hybrid.exceptions import ContractValidationError


def test_contract_serialization(prediction_context):
    payload = prediction_context.to_dict()
    assert json.loads(json.dumps(payload))["stage"] == "MIDDLE_50"
    rebuilt = PredictionContext(
        **{
            **prediction_context.__dict__,
            "stage": Stage(payload["stage"]),
        }
    )
    assert rebuilt == prediction_context


def test_contract_invalid_stage_rejected(prediction_context):
    with pytest.raises(ContractValidationError):
        PredictionContext(**{**prediction_context.__dict__, "stage": "F2_MIDDLE"})


def test_contract_invalid_dimension_rejected():
    with pytest.raises(ContractValidationError):
        StudentRepresentation(
            student_state_embedding=(0.0,) * 63,
            student_state_dimension=63,
            tabular_expert_embedding=(0.0,) * 32,
            tabular_expert_dimension=32,
            model_authority="RECOMMEND_HYBRID_MODEL_AUTHORITY",
            embedding_source="frozen_forward_output",
            dtype="float32",
            device="cpu",
        )
