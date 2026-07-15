from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from src.estimator_factory import resolve_phase_c_neural_config
from src.models import count_trainable_parameters, create_phase_c_model


NEURAL_MAPPING = {
    "B-M0": "N2",
    "B-C0": "A1",
    "B-L1": "A2",
    "B-H1": "N0",
    "B-O0": "N1",
}


def ml_configs(candidate_id: str) -> list[dict[str, Any]]:
    if candidate_id == "B-L0":
        return [{"C": value} for value in [0.1, 1.0, 10.0]]
    if candidate_id == "B-RF0":
        return [
            {"n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf, "max_features": feature}
            for n, depth, leaf, feature in [
                (100, None, 1, "sqrt"), (200, None, 2, "sqrt"), (300, 8, 1, "log2"),
                (200, 5, 2, None), (300, None, 4, "sqrt"), (100, 8, 4, None),
            ]
        ]
    if candidate_id == "B-S0":
        return [{"C": c, "gamma": gamma} for c, gamma in product([0.1, 1.0, 10.0], [0.01, 0.1])]
    if candidate_id == "B-H0":
        return [
            {"learning_rate": lr, "max_leaf_nodes": leaves, "l2_regularization": l2}
            for lr, leaves, l2 in [(0.03, 7, 0.0), (0.05, 15, 0.1), (0.1, 15, 1.0), (0.05, 31, 1.0)]
        ]
    raise KeyError(candidate_id)


def make_ml_model(candidate_id: str, config: dict[str, Any], seed: int):
    if candidate_id == "B-L0":
        return Pipeline([("scale", MinMaxScaler()), ("model", LogisticRegression(C=config["C"], max_iter=2000, random_state=seed))])
    if candidate_id == "B-RF0":
        return RandomForestClassifier(**config, random_state=seed, class_weight=None, n_jobs=-1)
    if candidate_id == "B-S0":
        return Pipeline([("scale", MinMaxScaler()), ("model", SVC(**config, kernel="rbf", probability=True, random_state=seed))])
    if candidate_id == "B-H0":
        return HistGradientBoostingClassifier(**config, max_iter=200, random_state=seed)
    raise KeyError(candidate_id)


def neural_configs(candidate_id: str) -> list[dict[str, Any]]:
    internal = NEURAL_MAPPING[candidate_id]
    common = {
        "oversample_method": "none", "class_weight_mode": "none", "loss": "cross_entropy",
        "smote_ratio": 1.0, "resampling_k_neighbors": 5, "batch_size": 32,
        "normalization": "none", "sequence_dropout": 0.0, "max_epochs": 24, "patience": 4,
        "weight_decay": 1e-5, "num_layers": 1,
    }
    if internal in {"N2"}:
        variants = [
            {**common, "learning_rate": 3e-3, "dropout": 0.1, "hidden_dim": 8, "cnn_channels": 1, "cnn_kernel_size": 1, "lstm_hidden_dim": 1},
            {**common, "learning_rate": 1e-3, "dropout": 0.2, "hidden_dim": 16, "cnn_channels": 1, "cnn_kernel_size": 1, "lstm_hidden_dim": 1},
        ]
    elif internal == "A1":
        variants = [
            {**common, "learning_rate": 3e-3, "dropout": 0.1, "cnn_channels": 8, "cnn_kernel_size": 1, "lstm_hidden_dim": 1},
            {**common, "learning_rate": 1e-3, "dropout": 0.2, "cnn_channels": 8, "cnn_kernel_size": 2, "lstm_hidden_dim": 1},
        ]
    elif internal == "A2":
        variants = [
            {**common, "learning_rate": 3e-3, "dropout": 0.1, "cnn_channels": 1, "cnn_kernel_size": 1, "lstm_hidden_dim": 8},
            {**common, "learning_rate": 1e-3, "dropout": 0.2, "cnn_channels": 1, "cnn_kernel_size": 1, "lstm_hidden_dim": 16},
        ]
    else:
        variants = [
            {**common, "learning_rate": 3e-3, "dropout": 0.1, "cnn_channels": 8, "cnn_kernel_size": 1, "lstm_hidden_dim": 8},
            {**common, "learning_rate": 1e-3, "dropout": 0.2, "cnn_channels": 8, "cnn_kernel_size": 2, "lstm_hidden_dim": 16},
        ]
    resolved = []
    for parameters in variants:
        config = resolve_phase_c_neural_config(internal, parameters, suggested_parameters={key: value for key, value in parameters.items() if key not in {"oversample_method", "class_weight_mode", "loss", "smote_ratio", "resampling_k_neighbors", "patience"}}, evidence_role="study_b_student_por")
        config["parameter_count"] = count_trainable_parameters(create_phase_c_model(config))
        resolved.append(config)
    return resolved


def align_probabilities(model, probabilities: np.ndarray) -> np.ndarray:
    output = np.zeros((len(probabilities), 3), dtype=float)
    classes = np.asarray(model.classes_ if hasattr(model, "classes_") else model.named_steps["model"].classes_, dtype=int)
    output[:, classes] = probabilities
    return output
