"""Persistence classifier. Features are cutoff-safe; labels are 14-day logs."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import FEATURE_COLUMNS, PersistLabel
from .feasibility import rule_label

CLASSES = (PersistLabel.ASSESS.value, PersistLabel.ENGAGE.value, PersistLabel.COUNSEL.value)


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    values = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    for column in (
        "vle_access_available",
        "quiz_available",
        "study_material_available",
    ):
        values[column] = values[column].fillna(False).astype(float)
    return values.to_numpy(dtype=np.float64)


def _sample_weight(y: np.ndarray) -> np.ndarray:
    values, counts = np.unique(y, return_counts=True)
    freq = {lab: n for lab, n in zip(values, counts)}
    n = len(y)
    k = max(len(freq), 1)
    return np.array([n / (k * freq[lab]) for lab in y], dtype=np.float64)


def build_candidates(random_state: int = 2026) -> dict[str, Pipeline]:
    return {
        "hgb": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=4,
                        learning_rate=0.08,
                        max_iter=200,
                        min_samples_leaf=40,
                        l2_regularization=0.1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "logreg": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=400,
                        class_weight="balanced",
                        solver="lbfgs",
                        C=1.0,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _proba(estimator: Pipeline, X: np.ndarray, classes: tuple[str, ...]) -> np.ndarray:
    raw = estimator.predict_proba(X)
    order = list(estimator.classes_)
    aligned = np.zeros((len(X), len(classes)), dtype=np.float64)
    for i, name in enumerate(classes):
        if name in order:
            aligned[:, i] = raw[:, order.index(name)]
    return aligned


def macro_average_precision(y_true: np.ndarray, proba: np.ndarray, classes: tuple[str, ...]) -> float:
    scores = []
    for i, name in enumerate(classes):
        truth = (y_true == name).astype(int)
        if truth.sum() == 0 or truth.sum() == len(truth):
            continue
        scores.append(average_precision_score(truth, proba[:, i]))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def fit_select(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    random_state: int = 2026,
) -> dict:
    fitted = {}
    best_name = None
    best_ap = -1.0
    weights = _sample_weight(y_train)
    for name, pipe in build_candidates(random_state).items():
        clf = pipe.named_steps["clf"]
        if isinstance(clf, HistGradientBoostingClassifier):
            pipe.fit(X_train, y_train, clf__sample_weight=weights)
        else:
            pipe.fit(X_train, y_train)
        proba = _proba(pipe, X_val, CLASSES)
        ap = macro_average_precision(y_val, proba, CLASSES)
        pred = np.array(CLASSES)[proba.argmax(axis=1)]
        f1 = float(f1_score(y_val, pred, average="macro", labels=list(CLASSES), zero_division=0))
        fitted[name] = {"pipeline": pipe, "val_macro_ap": ap, "val_macro_f1": f1}
        if ap > best_ap:
            best_ap = ap
            best_name = name
    if best_name is None:
        raise RuntimeError("no persistence model fitted")
    return {
        "selected": best_name,
        "classes": list(CLASSES),
        "models": fitted,
        "pipeline": fitted[best_name]["pipeline"],
        "val_macro_ap": fitted[best_name]["val_macro_ap"],
        "val_macro_f1": fitted[best_name]["val_macro_f1"],
    }


class PersistenceClassifier:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.pipeline: Pipeline = bundle["pipeline"]
        self.classes = tuple(bundle["classes"])

    @classmethod
    def load(cls, path: Path) -> "PersistenceClassifier":
        return cls(joblib.load(path))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.bundle, path)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return _proba(self.pipeline, feature_matrix(frame), self.classes)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(frame)
        return np.array(self.classes)[proba.argmax(axis=1)]

    def constrained_predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Argmax over feasible labels only. COUNSEL always feasible."""
        proba = self.predict_proba(frame)
        chosen = []
        scores = []
        for i, (_, row) in enumerate(frame.iterrows()):
            from .feasibility import feasible_labels

            allowed = {item.value for item in feasible_labels(row.to_dict())}
            mask = np.array([c in allowed for c in self.classes], dtype=bool)
            masked = np.where(mask, proba[i], -1.0)
            idx = int(masked.argmax())
            chosen.append(self.classes[idx])
            scores.append(float(proba[i, idx]))
        return np.array(chosen, dtype=object), np.asarray(scores, dtype=np.float64)


def rule_predict(frame: pd.DataFrame) -> np.ndarray:
    return np.array([rule_label(row.to_dict()).value for _, row in frame.iterrows()], dtype=object)


__all__ = [
    "CLASSES",
    "PersistenceClassifier",
    "feature_matrix",
    "fit_select",
    "macro_average_precision",
    "rule_predict",
]
