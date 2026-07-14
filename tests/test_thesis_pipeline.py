from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_pipeline import (
    DataPreprocessor,
    FeatureSelector,
    StudentDataset,
)
from src.explainability import calculate_permutation_importance
from src.recommendation import build_recommendation, generate_learning_path_report
from src.models import create_model
from src.model_selection import student_search_space
from src.train_pipeline import calculate_class_weights, suggest_trial_params
from scripts.run_pipeline import normalize_cnn_bilstm_classifier_params


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_model_is_cnn_bilstm_classifier_without_context_branch():
    model = create_model(
        "student",
        {
            "cnn_channels": 16,
            "cnn_kernel_size": 1,
            "lstm_hidden_dim": 12,
            "dropout": 0.1,
        },
        num_numerical=0,
        cat_cardinalities=[],
    )
    assert isinstance(model.sequence_cnn[0], nn.Conv1d)
    assert isinstance(model.sequence_bilstm, nn.LSTM)
    assert model.sequence_bilstm.bidirectional
    assert model.context_mlp_enabled is False
    assert not hasattr(model, "context_mlp")
    assert not hasattr(model, "fusion")

    seq_x = torch.randn(5, 2, 1)
    logits = model(seq_x)
    probabilities = model.predict_proba(seq_x)

    assert logits.shape == (5, 3)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(5), atol=1e-6)


def test_model_uses_sequence_dropout_and_linear_classifier_head():
    model = create_model(
        "student",
        {
            "dropout": 0.1,
            "sequence_dropout": 0.2,
        },
        num_numerical=0,
        cat_cardinalities=[],
    )
    assert model.sequence_dropout.p == 0.2
    assert model.head_dropout.p == 0.1
    assert model.classifier_head == "linear"
    assert isinstance(model.classifier, nn.Linear)


@pytest.mark.parametrize("variant", ["cnn_only", "bilstm_only", "cnn_lstm", "cnn_bilstm", "cnn_bigru"])
def test_architecture_ablation_variants_produce_three_class_logits(variant):
    model = create_model(
        "student",
        {
            "architecture_variant": variant,
            "cnn_channels": 8,
            "lstm_hidden_dim": 8,
            "cnn_kernel_size": 1,
        },
    )
    logits = model(torch.randn(4, 2, 1))
    assert logits.shape == (4, 3)
    assert model.architecture_variant == variant


def test_cnn_bigru_uses_a_bidirectional_gru():
    model = create_model(
        "student",
        {
            "architecture_variant": "cnn_bigru",
            "cnn_channels": 8,
            "lstm_hidden_dim": 8,
            "cnn_kernel_size": 1,
        },
    )

    assert isinstance(model.sequence_bilstm, nn.GRU)
    assert model.sequence_bilstm.bidirectional
    assert model.recurrent_cell == "gru"


def test_cnn_lstm_uses_a_unidirectional_lstm():
    model = create_model("student", {"architecture_variant": "cnn_lstm", "cnn_channels": 8, "lstm_hidden_dim": 8})

    assert isinstance(model.sequence_bilstm, nn.LSTM)
    assert model.sequence_bilstm.bidirectional is False
    assert model(torch.randn(4, 2, 1)).shape == (4, 3)


def test_xapi_optuna_space_excludes_vanilla_smote():
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
    assert trial.calls["oversample_method"] == ("categorical", ["none", "random", "smotenc"])
    assert "smote" not in trial.calls["oversample_method"][1]
    assert trial.calls["cnn_kernel_size"][1] == [2, 3, 4]
    assert trial.calls["lstm_hidden_dim"][1][-1] == 128
    assert trial.calls["sequence_dropout"] == ("float", 0.1, 0.6, False)
    assert trial.calls["smote_ratio"] == ("float", 0.3, 1.0, False)
    assert "resampling_k_neighbors" in params
    assert "context_hidden_dim" not in trial.calls
    assert "fusion_hidden_dim" not in trial.calls


def test_student_model_selection_space_uses_only_requested_resampling_methods():
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
    student_search_space(trial)

    assert trial.calls["oversample_method"] == ("categorical", ["none", "smote"])
    assert trial.calls["class_weight_mode"] == ("categorical", ["none", "balanced"])
    assert trial.calls["cnn_kernel_size"] == ("categorical", [1])


def test_baseline_search_excludes_inapplicable_architecture_parameters():
    class RecordingTrial:
        def suggest_float(self, name, low, high, log=False):
            return low

        def suggest_int(self, name, low, high):
            return low

        def suggest_categorical(self, name, choices):
            return choices[0]

    cnn_only = student_search_space(RecordingTrial(), architecture_variant="cnn_only")
    bilstm_only = student_search_space(RecordingTrial(), architecture_variant="bilstm_only")

    assert "lstm_hidden_dim" not in cnn_only["suggested_parameters"]
    assert "cnn_channels" not in bilstm_only["suggested_parameters"]
    assert cnn_only["architecture_variant"] == "cnn_only"
    assert bilstm_only["architecture_variant"] == "bilstm_only"


def test_sequence_only_oversampling_uses_only_model_input_columns():
    frame = pd.DataFrame(
        {
            "G1": [6, 7, 8, 9, 14, 15, 16, 17, 18],
            "G2": [7, 8, 9, 10, 15, 16, 17, 18, 19],
            "school": ["GP", "MS", "GP", "MS", "GP", "MS", "GP", "MS", "GP"],
            "G3": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        }
    )
    preprocessor = DataPreprocessor(
        "G3",
        oversample_method="adasyn",
        smote_ratio=1.0,
        resampling_k_neighbors=2,
        oversampling_feature_columns=["G1", "G2"],
    )

    transformed = preprocessor.fit_transform(frame)

    assert set(transformed.columns) == {"G1", "G2", "G3"}
    assert "school" not in transformed.columns


def test_resampling_neighbor_count_is_configurable():
    preprocessor = DataPreprocessor("Class", resampling_k_neighbors=7)
    assert preprocessor.resampling_k_neighbors == 7


def test_forbidden_architectures_and_losses_are_removed():
    source = (PROJECT_ROOT / "src" / "models" / "models.py").read_text(encoding="utf-8")
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


def test_student_sequence_input_uses_only_prior_grade_allowlist():
    frame = pd.DataFrame(
        {
            "G1": [8.0, 12.0],
            "G2": [9.0, 13.0],
            "G3": [0, 2],
            "studytime": [1.0, 3.0],
            "__source_row_number": [0, 1],
        }
    )
    dataset = StudentDataset(
        frame,
        kind="student",
        target_col="G3",
        numerical_cols=["G1", "G2", "studytime", "__source_row_number"],
        categorical_cols=[],
    )

    assert dataset.seq_cols == ["G1", "G2"]
    assert dataset.num_cols == []
    assert dataset.cat_cols == []
    seq_x, num_x, cat_x, *_ = dataset[0]
    assert seq_x.flatten().tolist() == [8.0, 9.0]
    assert num_x.shape == (0,)
    assert cat_x.shape == (0,)


def test_final_train_config_params_disable_context_mlp():
    params = normalize_cnn_bilstm_classifier_params(
        {
            "cnn_channels": 32,
            "lstm_hidden_dim": 64,
            "context_hidden_dim": 128,
            "fusion_hidden_dim": 128,
            "context_dropout": 0.2,
            "fusion_dropout": 0.3,
        }
    )

    assert params["architecture"] == "cnn_bilstm_classifier"
    assert params["context_mlp_enabled"] is False
    assert params["classifier_head"] == "linear"
    assert "context_hidden_dim" not in params
    assert "fusion_hidden_dim" not in params


def test_active_pipeline_uses_cnn_bilstm_classifier_names():
    source = (PROJECT_ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")

    assert "cnn_bilstm_classifier" in source
    assert "cnn_bilstm_mlp" not in source
    assert "CNN-BiLSTM + MLP" not in source
    assert "confidences = mean_probabilities.max(axis=1)" in source


def test_context_permutation_importance_is_empty_for_sequence_only_model():
    frame = pd.DataFrame(
        {
            "G1": [8.0, 12.0, 14.0],
            "G2": [9.0, 13.0, 15.0],
            "G3": [0, 1, 2],
        }
    )
    dataset = StudentDataset(frame, "student", "G3", numerical_cols=["G1", "G2"], categorical_cols=[])
    loader = DataLoader(dataset, batch_size=2)
    model = create_model("student", {}, num_numerical=0, cat_cardinalities=[])

    importance = calculate_permutation_importance(model, loader, torch.device("cpu"), [], [])

    assert list(importance.columns) == ["Feature", "Importance"]
    assert importance.empty


def test_student_dataset_is_sequence_only_and_excludes_context_features():
    frame = pd.DataFrame(
        {
            "raisedhands": [10, 80],
            "VisITedResources": [20, 90],
            "AnnouncementsView": [5, 40],
            "Discussion": [7, 70],
            "engagement_score": [42, 280],
            "absence_risk": [1, 0],
            "hands_resource_ratio": [0.5, 0.9],
            "active_participation": [70, 5600],
            "resource_engagement_ratio": [0.45, 0.32],
            "parent_support_signal": [0, 1],
            "gender": [0, 1],
            "Class": [0, 2],
        }
    )
    numerical_cols = [
        "raisedhands",
        "VisITedResources",
        "AnnouncementsView",
        "Discussion",
        "engagement_score",
        "absence_risk",
        "hands_resource_ratio",
        "active_participation",
        "resource_engagement_ratio",
        "parent_support_signal",
    ]
    dataset = StudentDataset(
        frame,
        kind="xapi",
        target_col="Class",
        numerical_cols=numerical_cols,
        categorical_cols=["gender"],
    )

    assert dataset.seq_cols == ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]
    assert dataset.num_cols == []
    assert dataset.cat_cols == []
    seq_x, num_x, cat_x, *_ = dataset[0]
    assert seq_x.shape == (4, 1)
    assert num_x.shape == (0,)
    assert cat_x.shape == (0,)


def test_final_rule_policy_generates_advisory_output():
    result = build_recommendation({"G1": 8, "G2": 7, "absences": 16, "studytime": 1, "failures": 1}, predicted_class=0, confidence=0.82)
    assert result["risk_band"] == "High"
    assert result["recommended_actions"]
    assert "advisor" in result["disclaimer"].lower()


def test_final_schema_uses_lineage_tables():
    migration = (PROJECT_ROOT / "database" / "migrations" / "001_create_source_ml_schema.sql").read_text(encoding="utf-8").lower()
    for required in ("source_dataset_versions", "source_records", "ml_predictions", "ml_recommendations"):
        assert required in migration
