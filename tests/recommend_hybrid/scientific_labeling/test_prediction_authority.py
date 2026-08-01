from dataclasses import replace

import pytest

from src.recommend_hybrid.common.policy_contracts import DatasetId
from src.recommend_hybrid.exceptions import ContractValidationError
from src.recommend_hybrid.weak_supervision.contracts import PredictionContext
from src.recommend_hybrid.weak_supervision.validation import validate_prediction_authority


def _context() -> PredictionContext:
    return PredictionContext(
        model_id="cnn_bilstm_mat",
        prediction_authority="FINAL_THESIS_MODEL_AUTHORITY",
        dataset=DatasetId.STUDENT_MAT,
        stage="S1",
        requested_cutoff=1,
        predicted_class=0,
        class_probabilities=(0.7, 0.2, 0.1),
        uncertainty=0.4,
        checkpoint_lineage=("frozen-outer-fold",),
    )


def test_hybrid_authority_is_accepted() -> None:
    validate_prediction_authority(_context())


def test_non_hybrid_authority_is_rejected() -> None:
    with pytest.raises(ContractValidationError, match="authorized Hybrid CNN-BiLSTM"):
        validate_prediction_authority(
            replace(_context(), model_id="random_forest", prediction_authority="BASELINE")
        )
