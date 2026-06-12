from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data_pipeline import DataPreprocessor, FeatureSelector, StudentDataset
from src.explainability import RuleBasedLearningPathEngine, generate_learning_path_report
from src.models import StudentHybridModel, create_model
from src.train_pipeline import calculate_class_weights, suggest_trial_params


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_model_is_cnn_bilstm_mlp_and_outputs_three_class_probabilities():
    model = create_model(
        "student",
        {
            "cnn_channels": 16,
            "cnn_kernel_size": 3,
            "lstm_hidden_dim": 12,
            "context_hidden_dim": 10,
            "fusion_hidden_dim": 8,
            "dropout": 0.1,
        },
        num_numerical=3,
        cat_cardinalities=[2, 4],
    )
    assert isinstance(model.sequence_cnn[0], nn.Conv1d)
    assert isinstance(model.sequence_bilstm, nn.LSTM)
    assert model.sequence_bilstm.bidirectional
    assert isinstance(model.context_mlp[0], nn.Linear)

    seq_x = torch.randn(5, 2, 1)
    num_x = torch.randn(5, 3)
    cat_x = torch.tensor([[0, 1], [1, 2], [0, 3], [1, 0], [0, 2]])
    logits = model(seq_x, num_x, cat_x)
    probabilities = model.predict_proba(seq_x, num_x, cat_x)

    assert logits.shape == (5, 3)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(5), atol=1e-6)


def test_xapi_model_supports_independent_branch_dropouts():
    model = create_model(
        "xapi",
        {
            "sequence_dropout": 0.2,
            "context_dropout": 0.3,
            "fusion_dropout": 0.4,
        },
        num_numerical=4,
        cat_cardinalities=[2],
    )
    assert model.sequence_cnn[-1].p == 0.2
    assert model.context_mlp[2].p == 0.3
    assert model.fusion[2].p == 0.4


def test_xapi_optuna_space_matches_high_trial_configuration():
    class RecordingTrial:
        def __init__(self):
            self.calls = {}

        def suggest_float(self, name, low, high, log=False):
            self.calls[name] = ("float", low, high, log)
            return low

        def suggest_int(self, name, low, high):
            self.calls[name] = ("int", low, high)
            return low

        def suggest_categorical(self, name, choices):
            self.calls[name] = ("categorical", list(choices))
            return choices[0]

    trial = RecordingTrial()
    params = suggest_trial_params(trial, "xapi")

    assert trial.calls["learning_rate"] == ("float", 5e-5, 5e-2, True)
    assert trial.calls["cnn_kernel_size"][1] == [2, 3, 4]
    assert trial.calls["lstm_hidden_dim"][1][-1] == 128
    assert trial.calls["fusion_hidden_dim"][1][-1] == 256
    assert trial.calls["sequence_dropout"] == ("float", 0.1, 0.6, False)
    assert trial.calls["smote_ratio"] == ("float", 0.3, 1.0, False)
    assert "resampling_k_neighbors" in params


def test_resampling_neighbor_count_is_configurable():
    preprocessor = DataPreprocessor("Class", resampling_k_neighbors=7)
    assert preprocessor.resampling_k_neighbors == 7


def test_forbidden_architectures_and_losses_are_removed():
    source = (PROJECT_ROOT / "src" / "models.py").read_text(encoding="utf-8")
    for forbidden in (
        "DeepFM",
        "DCNv2",
        "FTTransformer",
        "TabularTokenizer",
        "HybridLoss",
        "FocalLoss",
    ):
        assert forbidden not in source


def test_weighted_cross_entropy_supports_imbalanced_classes():
    weights = calculate_class_weights(np.array([0, 1, 1, 1, 2, 2]), num_classes=3)
    criterion = nn.CrossEntropyLoss(weight=weights)
    loss = criterion(torch.randn(6, 3), torch.tensor([0, 1, 1, 1, 2, 2]))
    assert loss.item() > 0
    assert weights[0] > weights[1]


def test_feature_selector_keeps_required_sequence_columns():
    frame = pd.DataFrame(
        {
            "G1": [5, 6, 10, 11, 15, 16],
            "G2": [6, 7, 11, 12, 16, 17],
            "noise": [1, 1, 1, 1, 1, 1],
            "G3": [0, 0, 1, 1, 2, 2],
        }
    )
    selector = FeatureSelector("G3", required_features=["G1", "G2"])
    selected = selector.fit_transform(frame, ["G1", "G2", "noise"], [])
    assert {"G1", "G2", "G3"}.issubset(selected.columns)


def test_learning_path_engine_returns_staged_roadmap_not_variable_tweaks():
    engine = RuleBasedLearningPathEngine("student")
    result = engine.generate(
        {"G1": 8, "G2": 7, "absences": 16, "studytime": 1, "failures": 1},
        predicted_class=0,
        confidence=0.82,
    )
    assert result["risk_band"] == "high"
    assert len(result["learning_path"]) >= 3
    assert all({"phase", "goal", "actions"}.issubset(step) for step in result["learning_path"])
    assert any(risk["code"] == "attendance" for risk in result["risk_factors"])


def test_learning_path_report_has_one_row_per_student():
    features = pd.DataFrame(
        [
            {"StudentAbsenceDays": "Above-7", "VisITedResources": 20, "raisedhands": 10, "Discussion": 10},
            {"StudentAbsenceDays": "Under-7", "VisITedResources": 80, "raisedhands": 70, "Discussion": 60},
        ]
    )
    report = generate_learning_path_report(
        features,
        predictions=np.array([0, 2]),
        confidences=np.array([0.8, 0.9]),
        dataset_kind="xapi",
    )
    assert len(report) == 2
    assert set(report["risk_band"]).issubset({"high", "moderate", "stable"})


def test_postgres_schema_stores_features_confidence_and_learning_paths():
    schema = (PROJECT_ROOT / "database" / "schema.sql").read_text(encoding="utf-8").lower()
    for required in (
        "paper_predictions",
        "confidence real",
        "original_features jsonb",
        "paper_learning_recommendations",
        "recommended_learning_path jsonb",
    ):
        assert required in schema
