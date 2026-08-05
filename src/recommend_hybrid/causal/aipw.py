"""Cross-fitted doubly robust estimation for binary educational outcomes."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPSILON = 1.0e-6


class _ConstantProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(np.clip(probability, EPSILON, 1.0 - EPSILON))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        count = len(features)
        positive = np.full(count, self.probability, dtype=np.float64)
        return np.column_stack([1.0 - positive, positive])


def _default_logistic(random_state: int) -> Any:
    """Scale mixed confounders inside every cross-fit training partition."""

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            penalty="l2",
            C=1.0,
            random_state=random_state,
        ),
    )


def _fit_probability_model(model: Any, features: np.ndarray, target: np.ndarray) -> Any:
    values = np.asarray(target, dtype=np.int8)
    if len(np.unique(values)) < 2:
        return _ConstantProbabilityModel(float(np.mean(values)))
    fitted = clone(model)
    fitted.fit(features, values)
    return fitted


def _positive_probability(model: Any, features: np.ndarray) -> np.ndarray:
    probability = np.asarray(model.predict_proba(features), dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != 2:
        raise ValueError("nuisance models must expose binary predict_proba")
    return np.clip(probability[:, 1], EPSILON, 1.0 - EPSILON)


@dataclass(frozen=True)
class AIPWConfig:
    n_splits: int = 3
    random_state: int = 20260806
    confidence_level: float = 0.95
    propensity_clip: tuple[float, float] = (0.01, 0.99)

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least two")
        low, high = self.propensity_clip
        if not 0.0 < low < high < 1.0:
            raise ValueError("propensity_clip must lie strictly inside (0, 1)")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0, 1)")


@dataclass(frozen=True)
class AIPWResult:
    ate: float
    standard_error: float
    confidence_interval: tuple[float, float]
    propensity: np.ndarray
    outcome_if_control: np.ndarray
    outcome_if_treated: np.ndarray
    doubly_robust_score: np.ndarray
    cate: np.ndarray
    fold_id: np.ndarray

    def summary(self) -> dict[str, object]:
        return {
            "ate": self.ate,
            "standard_error": self.standard_error,
            "confidence_interval": list(self.confidence_interval),
            "sample_count": int(len(self.doubly_robust_score)),
            "positive_cate_fraction": float(np.mean(self.cate > 0.0)),
            "mean_propensity": float(np.mean(self.propensity)),
        }


class CrossFittedAIPW:
    """Estimate ATE and out-of-fold CATE without training on evaluation rows."""

    def __init__(
        self,
        *,
        propensity_model: Any | None = None,
        outcome_model: Any | None = None,
        cate_model: Any | None = None,
        config: AIPWConfig = AIPWConfig(),
    ) -> None:
        self.propensity_model = propensity_model or _default_logistic(
            config.random_state
        )
        self.outcome_model = outcome_model or _default_logistic(config.random_state)
        self.cate_model = cate_model or HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=config.random_state,
        )
        self.config = config

    def _splits(
        self,
        treatment: np.ndarray,
        groups: np.ndarray | None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            splitter = StratifiedKFold(
                n_splits=self.config.n_splits,
                shuffle=True,
                random_state=self.config.random_state,
            )
            return list(splitter.split(np.zeros(len(treatment)), treatment))
        group_values = np.asarray(groups)
        if len(group_values) != len(treatment):
            raise ValueError("groups must align with treatment")
        splitter = StratifiedGroupKFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )
        return list(
            splitter.split(np.zeros(len(treatment)), treatment, groups=group_values)
        )

    def fit_predict(
        self,
        features: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        *,
        groups: np.ndarray | None = None,
    ) -> AIPWResult:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("features must be a finite two-dimensional matrix")
        t = np.asarray(treatment, dtype=np.int8).reshape(-1)
        y = np.asarray(outcome, dtype=np.int8).reshape(-1)
        if len(x) != len(t) or len(t) != len(y):
            raise ValueError("features, treatment, and outcome must align")
        if not np.isin(t, [0, 1]).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("treatment and outcome must be binary")
        if min(int(np.sum(t == 0)), int(np.sum(t == 1))) < self.config.n_splits:
            raise ValueError("each treatment arm needs at least n_splits rows")

        splits = self._splits(t, groups)
        propensity = np.zeros(len(t), dtype=np.float64)
        mu0 = np.zeros(len(t), dtype=np.float64)
        mu1 = np.zeros(len(t), dtype=np.float64)
        fold_id = np.full(len(t), -1, dtype=np.int16)

        for fold, (train_index, test_index) in enumerate(splits):
            x_train = x[train_index]
            t_train = t[train_index]
            y_train = y[train_index]
            propensity_model = _fit_probability_model(
                self.propensity_model,
                x_train,
                t_train,
            )
            propensity[test_index] = _positive_probability(
                propensity_model,
                x[test_index],
            )
            control = t_train == 0
            treated = t_train == 1
            control_model = _fit_probability_model(
                self.outcome_model,
                x_train[control],
                y_train[control],
            )
            treated_model = _fit_probability_model(
                self.outcome_model,
                x_train[treated],
                y_train[treated],
            )
            mu0[test_index] = _positive_probability(control_model, x[test_index])
            mu1[test_index] = _positive_probability(treated_model, x[test_index])
            fold_id[test_index] = fold

        low, high = self.config.propensity_clip
        propensity = np.clip(propensity, low, high)
        dr_score = (
            mu1
            - mu0
            + t * (y - mu1) / propensity
            - (1 - t) * (y - mu0) / (1.0 - propensity)
        )
        ate = float(np.mean(dr_score))
        standard_error = float(np.std(dr_score - ate, ddof=1) / np.sqrt(len(dr_score)))
        alpha = 1.0 - self.config.confidence_level
        z_value = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        confidence_interval = (
            float(ate - z_value * standard_error),
            float(ate + z_value * standard_error),
        )

        cate = np.zeros(len(t), dtype=np.float64)
        for train_index, test_index in splits:
            model = clone(self.cate_model)
            model.fit(x[train_index], dr_score[train_index])
            cate[test_index] = np.asarray(model.predict(x[test_index]), dtype=np.float64)

        if (fold_id < 0).any():
            raise RuntimeError("cross-fitting failed to cover every row")
        return AIPWResult(
            ate=ate,
            standard_error=standard_error,
            confidence_interval=confidence_interval,
            propensity=propensity,
            outcome_if_control=mu0,
            outcome_if_treated=mu1,
            doubly_robust_score=dr_score,
            cate=cate,
            fold_id=fold_id,
        )


__all__ = [
    "AIPWConfig",
    "AIPWResult",
    "CrossFittedAIPW",
]
