"""CPU classical baselines on the same cutoff-safe parity features."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

ACTIVE_PHASE4 = ("LR", "DT", "RF", "SVM", "MLP")

from .metrics import binary_metrics, select_stop_threshold


def _preprocessor(frame: pd.DataFrame, columns: list[str], categorical: list[str]) -> ColumnTransformer:
    cats = [c for c in categorical if c in columns]
    nums = [c for c in columns if c not in cats]
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), nums),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("enc", OneHotEncoder(handle_unknown="ignore"))]), cats),
        ],
        remainder="drop",
    )


def make_model(name: str, seed: int, y_train: np.ndarray):
    if name == "LR":
        return LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=seed)
    if name == "DT":
        return DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=250, random_state=seed)
    if name == "SVM":
        return make_svm(seed, kernel="linear", C=1.0, class_weight="balanced")
    raise ValueError(f"inactive_or_unknown_baseline:{name}")


def make_svm(seed: int, *, kernel: str = "linear", C: float = 1.0, gamma: str = "scale", class_weight: str = "balanced"):
    cw = None if class_weight in {None, "none"} else class_weight
    if kernel == "linear":
        base = LinearSVC(C=C, class_weight=cw, max_iter=4000, dual="auto", random_state=seed)
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    return SVC(kernel="rbf", C=C, gamma=gamma, class_weight=cw, probability=True, random_state=seed, cache_size=500)


def fit_eval_baseline(
    name: str,
    frame: pd.DataFrame,
    columns: list[str],
    categorical: list[str],
    fit_ids: list[str],
    stop_ids: list[str],
    valid_ids: list[str],
    seed: int,
    return_scores: bool = False,
    model=None,
) -> dict[str, Any]:
    ids = frame.record_id.astype(str)
    train = frame[ids.isin(fit_ids)]
    stop = frame[ids.isin(stop_ids)]
    valid = frame[ids.isin(valid_ids)]
    prep = _preprocessor(train, columns, categorical)
    x_train = prep.fit_transform(train)
    x_stop = prep.transform(stop)
    x_valid = prep.transform(valid)
    y_train = train.target.to_numpy()
    y_stop = stop.target.to_numpy()
    y_valid = valid.target.to_numpy()
    model = make_model(name, seed, y_train) if model is None else model
    model.fit(x_train, y_train)
    if hasattr(model, "predict_proba"):
        stop_p = model.predict_proba(x_stop)[:, 1]
        valid_p = model.predict_proba(x_valid)[:, 1]
        train_p = model.predict_proba(x_train)[:, 1]
    else:
        stop_p = model.decision_function(x_stop)
        valid_p = model.decision_function(x_valid)
        train_p = model.decision_function(x_train)
    threshold = select_stop_threshold(y_stop, stop_p)
    metrics = binary_metrics(y_valid, valid_p, threshold=threshold)
    metrics["stop_pr_auc"] = binary_metrics(y_stop, stop_p)["pr_auc"]
    metrics["train_pr_auc"] = binary_metrics(y_train, train_p)["pr_auc"]
    metrics["generalization_gap"] = float(metrics["train_pr_auc"] - metrics["pr_auc"])
    metrics["family"] = name
    metrics["n_features"] = int(x_train.shape[1])
    metrics["outer_test_used"] = False
    if return_scores:
        metrics["valid_record_id"] = valid.record_id.astype(str).to_numpy()
        metrics["valid_p"] = np.asarray(valid_p, dtype=np.float32)
        metrics["valid_y"] = np.asarray(y_valid)
    return metrics
