from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


CATEGORICAL = ["code_module", "presentation_season"]


def configs(candidate_id: str) -> list[dict[str, Any]]:
    return {
        "C-L0": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
        "C-R0": [{"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"}, {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 4, "max_features": "sqrt"}],
        "C-H0": [{"learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 0.1}, {"learning_rate": 0.08, "max_leaf_nodes": 31, "l2_regularization": 1.0}],
        "C-S0": [{"C": 1.0, "gamma": "scale"}],
    }[candidate_id]


def make_preprocessor(columns: list[str]) -> ColumnTransformer:
    numeric = [column for column in columns if column not in CATEGORICAL and column != "record_id"]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ], sparse_threshold=0.0)


def make_model(candidate_id: str, config: dict[str, Any], seed: int):
    if candidate_id == "C-L0": return LogisticRegression(**config, max_iter=1000, random_state=seed, class_weight="balanced")
    if candidate_id == "C-R0": return RandomForestClassifier(**config, random_state=seed, class_weight="balanced", n_jobs=-1)
    if candidate_id == "C-H0": return HistGradientBoostingClassifier(**config, max_iter=150, random_state=seed, class_weight="balanced")
    if candidate_id == "C-S0": return SVC(**config, kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
    raise KeyError(candidate_id)
