from __future__ import annotations

import tempfile

import numpy as np
import torch

from src.prediction import Hybrid, HybridConfig, PredictionResult
from src.prediction.baselines import ACTIVE_BASELINES, build_baseline
from src.prediction.contracts import oulad_risk_target, uci_risk_target
from src.prediction.data.uci import build_uci_stage_view
from src.prediction.training.checkpoints import load_checkpoint, save_checkpoint


def test_binary_target_contracts_are_fail_closed():
    np.testing.assert_array_equal(uci_risk_target(np.array([9, 10, 15])), [1, 0, 0])
    np.testing.assert_array_equal(oulad_risk_target(np.array(["Fail", "Withdrawn", "Pass", "Distinction"])), [1, 1, 0, 0])


def test_uci_stage_is_one_combined_binary_view():
    frame = __import__("pandas").DataFrame(
        {
            "G1": [9.0, 12.0], "G2": [10.0, 13.0], "target": [1, 0],
            "record_id": ["r1", "r2"], "global_student_group": ["g1", "g2"],
        }
    )
    view = build_uci_stage_view(frame, "S2")
    view.validate()
    assert view.temporal.shape == (2, 2, 1)
    assert set(np.unique(view.target)) <= {0, 1}


def test_hybrid_has_one_binary_logit_and_masks_future_steps():
    torch.manual_seed(42)
    model = Hybrid(HybridConfig(static_dim=3, temporal_dim=2, aggregate_dim=5))
    model.eval()
    inputs = {
        "static": torch.randn(2, 3), "temporal": torch.randn(2, 4, 2),
        "temporal_mask": torch.tensor([[True, True, False, False], [True, False, False, False]]),
        "lengths": torch.tensor([2, 1]), "aggregate": torch.randn(2, 5),
        "aggregate_available": torch.tensor([1, 1]), "progress": torch.tensor([0.5, 0.5]),
    }
    masked = inputs["temporal"].clone()
    masked[~inputs["temporal_mask"]] = 0.0
    inputs["temporal"] = masked
    output = model(**inputs)
    assert output.shape == (2,)


def test_checkpoint_roundtrip_uses_same_hybrid_class():
    model = Hybrid(HybridConfig(static_dim=3, temporal_dim=2, aggregate_dim=5))
    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/uci.pt"
        save_checkpoint(path, model, instance="uci")
        loaded = load_checkpoint(path)
    assert type(loaded) is Hybrid
    assert set(loaded.state_dict()) == set(model.state_dict())


def test_active_baselines_have_required_comparators():
    assert ACTIVE_BASELINES == ("Logistic Regression", "Decision Tree", "Random Forest", "SVM", "MLP")
    svm = build_baseline("SVM", dataset="uci")
    assert hasattr(svm, "predict_proba")


def test_recommendation_consumes_prediction_result_only():
    result = PredictionResult("uci_combined", "student-1", "S2", 0.8, 1, 0.5, uncertainty=0.2)
    assert result.recommendation_features()["risk_probability"] == 0.8
    assert result.recommendation_features()["model_id"] == "hybrid"
