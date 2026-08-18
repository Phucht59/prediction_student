"""Shared, leakage-safe UCI loading and tabular comparator utilities."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "artifacts" / "final"

MODEL_NAMES = {
    "cnn_bilstm": "CNN-BiLSTM",
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "svm": "SVM",
    "xgboost": "XGBoost",
    "mlp": "MLP",
}

CONTEXT = (
    "failures",
    "studytime",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "internet",
    "higher",
    "traveltime",
    "freetime",
    "goout",
    "health",
)
CONTEXT_CATEGORICAL = (
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "internet",
    "higher",
)
QUASI_IDENTITY = (
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "nursery",
    "internet",
)


def _stable_id(*parts: object) -> str:
    value = "\x1f".join(map(str, parts)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:24]


def encode_uci_target(values: Iterable[Any]) -> np.ndarray:
    """Map G3 to Low/Medium/High without exposing G3 to predictors."""

    raw = np.asarray(list(values), dtype=float)
    if raw.ndim != 1 or not np.isfinite(raw).all():
        raise ValueError("G3 must be a finite one-dimensional vector")
    if ((raw < 0) | (raw > 20)).any():
        raise ValueError("G3 must be inside 0..20")
    return np.where(raw < 10, 0, np.where(raw < 15, 1, 2)).astype(np.int64)


@dataclass
class UCIStudyData:
    dataset: str
    frame: pd.DataFrame
    target: np.ndarray
    record_ids: np.ndarray
    groups: np.ndarray
    outer_fold: np.ndarray


def _load_uci(dataset: str) -> UCIStudyData:
    if dataset not in {"student_mat", "student_por"}:
        raise ValueError(f"Unknown UCI dataset: {dataset}")
    filename = "student-mat.csv" if dataset == "student_mat" else "student-por.csv"
    namespace = "student-mat" if dataset == "student_mat" else "student-por"
    frame = pd.read_csv(ROOT / "data" / "raw" / filename, sep=";")
    required = {"G1", "G2", "G3", *CONTEXT, *QUASI_IDENTITY}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{dataset} source missing fields: {missing}")
    target = encode_uci_target(frame["G3"])
    record_ids = np.asarray(
        [_stable_id(namespace, index) for index in range(len(frame))], dtype=object
    )
    groups = np.asarray(
        [
            _stable_id(
                "quasi", *(frame.iloc[index][column] for column in QUASI_IDENTITY)
            )
            for index in range(len(frame))
        ],
        dtype=object,
    )
    frozen = pd.read_parquet(
        FINAL / "comparator_completion" / dataset / "oof_predictions.parquet"
    )
    frozen = frozen.loc[frozen["model_id"] == "cnn_bilstm"].copy()
    target_column = "target" if "target" in frozen.columns else "true_label"
    frozen = frozen[["record_id", "outer_fold", target_column]].drop_duplicates()
    if len(frozen) != len(frame) or frozen["record_id"].duplicated().any():
        raise RuntimeError(f"{dataset} frozen OOF assignment is incomplete")
    assignment = frozen.set_index("record_id")
    if set(record_ids) != set(assignment.index):
        raise RuntimeError(f"{dataset} record IDs do not match frozen evidence")
    outer_fold = assignment.loc[record_ids, "outer_fold"].to_numpy(dtype=int)
    frozen_target = assignment.loc[record_ids, target_column].to_numpy(dtype=int)
    if not np.array_equal(target, frozen_target):
        raise RuntimeError(f"{dataset} target does not match frozen evidence")
    return UCIStudyData(dataset, frame, target, record_ids, groups, outer_fold)


def _uci_preprocessor(columns: Iterable[str]) -> ColumnTransformer:
    columns = tuple(columns)
    categorical = [column for column in columns if column in CONTEXT_CATEGORICAL]
    numeric = [column for column in columns if column not in categorical]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0.0,
    )


def _candidate_grid(model_id: str, *, binary: bool) -> list[dict[str, Any]]:
    del binary
    if model_id == "logistic_regression":
        return [
            {"C": value, "class_weight": weight}
            for value in (0.1, 1.0, 10.0)
            for weight in (None, "balanced")
        ]
    if model_id == "decision_tree":
        return [
            {"max_depth": depth, "min_samples_leaf": leaf, "class_weight": weight}
            for depth in (3, 5, None)
            for leaf in (2, 5)
            for weight in (None, "balanced")
        ]
    if model_id == "random_forest":
        return [
            {
                "n_estimators": 300,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "class_weight": weight,
            }
            for depth in (None, 8)
            for leaf in (1, 3)
            for weight in (None, "balanced")
        ]
    if model_id == "hist_gradient_boosting":
        return [
            {
                "learning_rate": rate,
                "max_iter": 250,
                "max_leaf_nodes": leaves,
                "l2_regularization": l2,
            }
            for rate in (0.05, 0.1)
            for leaves in (15, 31)
            for l2 in (0.0, 1.0)
        ]
    if model_id == "svm":
        return [
            {"C": value, "gamma": gamma, "class_weight": weight}
            for value in (0.5, 1.0, 2.0)
            for gamma in ("scale", 0.1)
            for weight in (None, "balanced")
        ]
    if model_id == "xgboost":
        return [
            {
                "n_estimators": 300,
                "max_depth": depth,
                "learning_rate": rate,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": child,
            }
            for depth in (2, 4)
            for rate in (0.03, 0.08)
            for child in (1, 5)
        ]
    if model_id == "mlp":
        return [
            {
                "hidden_layer_sizes": tuple(layers),
                "alpha": alpha,
                "learning_rate_init": 0.001,
            }
            for layers in ((64,), (64, 32), (128, 64))
            for alpha in (0.0001, 0.001, 0.01)
        ]
    raise ValueError(model_id)


def _estimator(
    model_id: str, params: dict[str, Any], *, seed: int, binary: bool
) -> Any:
    if model_id == "logistic_regression":
        return LogisticRegression(
            **params, max_iter=2000, solver="lbfgs", random_state=seed
        )
    if model_id == "decision_tree":
        return DecisionTreeClassifier(**params, random_state=seed)
    if model_id == "random_forest":
        return RandomForestClassifier(**params, random_state=seed, n_jobs=1)
    if model_id == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**params, random_state=seed)
    if model_id == "svm":
        return SVC(**params, probability=True, random_state=seed)
    if model_id == "xgboost":
        return XGBClassifier(
            **params,
            objective="binary:logistic" if binary else "multi:softprob",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
    if model_id == "mlp":
        return MLPClassifier(
            **params,
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            batch_size=128 if binary else 32,
            random_state=seed,
        )
    raise ValueError(model_id)


def _pipeline(
    preprocessor_factory: Callable[[], ColumnTransformer],
    model_id: str,
    params: dict[str, Any],
    *,
    seed: int,
    binary: bool,
) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", preprocessor_factory()),
            ("model", _estimator(model_id, params, seed=seed, binary=binary)),
        ]
    )


def _aligned_probabilities(
    fitted: Pipeline, features: pd.DataFrame, classes: int
) -> np.ndarray:
    raw = np.asarray(fitted.predict_proba(features), dtype=float)
    labels = np.asarray(fitted.named_steps["model"].classes_, dtype=int)
    result = np.zeros((len(features), classes), dtype=float)
    result[:, labels] = raw
    totals = result.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise RuntimeError("Invalid probability matrix")
    return result / totals


def _inner_splits(
    y: np.ndarray, groups: np.ndarray, *, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(
        n_splits=3, shuffle=True, random_state=seed
    )
    return list(splitter.split(np.zeros(len(y)), y, groups))
