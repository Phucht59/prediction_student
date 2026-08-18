"""Fair Panel-A-only baselines sharing the Phase 8 target and feature contract."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from .features import APPROVED_FEATURES


def stage_prior_predict(train_stage: np.ndarray, train_y: np.ndarray, train_w: np.ndarray, pred_stage: np.ndarray) -> np.ndarray:
    priors = {}
    for stage in np.unique(train_stage):
        mask = train_stage == stage
        priors[float(stage)] = float(np.average(train_y[mask], weights=train_w[mask]))
    global_prior = float(np.average(train_y, weights=train_w))
    return np.asarray([priors.get(float(stage), global_prior) for stage in pred_stage], dtype=float)


def fit_ridge(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, *, alpha: float = 1.0) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def fit_random_forest(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, *, config: dict) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=int(config.get("n_estimators", 100)),
        max_depth=int(config.get("max_depth", 4)),
        min_samples_leaf=int(config.get("min_samples_leaf", 10)),
        random_state=int(config.get("random_state", 2026)),
        n_jobs=int(config.get("n_jobs", 1)),
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def stage_column(X: np.ndarray) -> np.ndarray:
    return X[:, list(APPROVED_FEATURES).index("stage_code")]
