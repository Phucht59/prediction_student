"""Explainable Boosting Regressor wrappers for one action model."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor

from ..evaluation.metrics import clip_score
from .features import APPROVED_FEATURES

FEATURE_TYPES = ["continuous"] * len(APPROVED_FEATURES)


def make_ebm(config: dict) -> ExplainableBoostingRegressor:
    return ExplainableBoostingRegressor(
        feature_names=list(APPROVED_FEATURES),
        feature_types=FEATURE_TYPES,
        max_bins=int(config["max_bins"]),
        interactions=int(config["interactions"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_leaves=int(config.get("max_leaves", 3)),
        outer_bags=int(config.get("outer_bags", 8)),
        inner_bags=int(config.get("inner_bags", 0)),
        learning_rate=float(config.get("learning_rate", 0.04)),
        early_stopping_rounds=int(config.get("early_stopping_rounds", 50)),
        early_stopping_tolerance=float(config.get("early_stopping_tolerance", 1e-5)),
        validation_size=float(config.get("validation_size", 0.15)),
        max_rounds=int(config.get("max_rounds", 5000)),
        n_jobs=int(config.get("n_jobs", 1)),
        random_state=int(config.get("random_state", 2026)),
    )


def fit_ebm(X: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray | None, config: dict) -> ExplainableBoostingRegressor:
    model = make_ebm(config)
    if sample_weight is None:
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


def predict_raw(model: ExplainableBoostingRegressor, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def predict_clipped(model: ExplainableBoostingRegressor, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = predict_raw(model, X)
    return raw, np.asarray(clip_score(raw), dtype=float)


def save_model(model: ExplainableBoostingRegressor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> ExplainableBoostingRegressor:
    return joblib.load(path)


def local_contributions(model: ExplainableBoostingRegressor, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    terms = np.asarray(model.eval_terms(X), dtype=float)
    names = np.asarray(model.term_names_)
    intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
    return terms, names, intercept


def global_importances(model: ExplainableBoostingRegressor) -> pd.DataFrame:
    names = list(model.term_names_)
    values = np.asarray(model.term_importances(), dtype=float)
    frame = pd.DataFrame({"term": names, "importance": values}).sort_values("importance", ascending=False)
    return frame.reset_index(drop=True)


def top_local_reasons(model: ExplainableBoostingRegressor, x_row: np.ndarray, *, n_pos: int = 3, n_neg: int = 2) -> dict:
    terms, names, intercept = local_contributions(model, np.asarray(x_row, dtype=float).reshape(1, -1))
    contrib = terms[0]
    order = np.argsort(contrib)
    negative = [{"term": str(names[i]), "contribution": float(contrib[i])} for i in order[:n_neg] if contrib[i] < 0]
    positive = [{"term": str(names[i]), "contribution": float(contrib[i])} for i in order[::-1][:n_pos] if contrib[i] > 0]
    raw = float(intercept + contrib.sum())
    return {
        "intercept": intercept,
        "raw_score": raw,
        "relevance_score": float(clip_score(raw)),
        "top_positive_reasons": positive,
        "top_negative_reasons": negative,
        "contributions": {str(name): float(value) for name, value in zip(names, contrib)},
    }
