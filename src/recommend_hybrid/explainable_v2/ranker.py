"""Explainable action relevance ranking without an action-identity shortcut."""

from __future__ import annotations

from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import joblib

from .contracts import ActionScore, CanonicalAction, RecommendationFeatures


# Only learner state known before the stage cutoff is admissible here.
# Weak-label conflict and OOD diagnostics are routing/audit signals and are
# deliberately excluded to prevent target-derived or evaluation-derived leakage.
FEATURE_COLUMNS = (
    "risk_probability",
    "hybrid_uncertainty",
    "course_progress",
    "inactivity_streak",
    "active_day_rate",
    "assessments_due",
    "regularity_score",
    "content_coverage",
    "quiz_activity",
    "missing_assessment_count",
    "due_soon_count",
    "completion_rate",
    "vle_available",
    "study_material_available",
    "quiz_available",
    "stage",
)

ORDINAL_RELEVANCE_MAX = 3.0


def canonical_ordinal_score_from_model_prediction(raw_prediction: float) -> float:
    """Bound an unconstrained EBM regressor output to the frozen ordinal scale."""

    if not np.isfinite(raw_prediction):
        raise ValueError("raw EBM prediction must be finite")
    return float(np.clip(raw_prediction, 0.0, ORDINAL_RELEVANCE_MAX))


def public_score_from_ordinal_prediction(native_prediction: float) -> float:
    """The single adapter from native ordinal EBM scale to public [0, 1]."""

    if not np.isfinite(native_prediction):
        raise ValueError("native EBM prediction must be finite")
    return float(np.clip(native_prediction / ORDINAL_RELEVANCE_MAX, 0.0, 1.0))


class ActionRanker(Protocol):
    def score(
        self,
        features: RecommendationFeatures,
        eligible_actions: tuple[CanonicalAction, ...],
    ) -> tuple[ActionScore, ...]: ...


def feature_frame(features: RecommendationFeatures) -> pd.DataFrame:
    raw = asdict(features)
    row = {column: raw.get(column) for column in FEATURE_COLUMNS}
    row["vle_available"] = features.vle_access_available
    row["stage"] = features.stage.value
    return pd.DataFrame([row], columns=list(FEATURE_COLUMNS))


class FiveEBMRanker:
    """One independent EBM relevance regressor per canonical action.

    Targets are ordinal weak-supervision relevance scores in {0, 1, 2, 3},
    normalized to [0, 1]. The action identity is never passed as a feature.
    Hyperparameters and any score calibration must be selected using validation
    data only. The class intentionally raises when an action model is missing.
    """

    def __init__(
        self,
        model_parameters: dict | None = None,
        *,
        native_prediction_scale: str = "NORMALIZED_0_1",
    ) -> None:
        if native_prediction_scale not in {"NORMALIZED_0_1", "ORDINAL_0_3"}:
            raise ValueError("unsupported native prediction scale")
        self.model_parameters = dict(model_parameters or {})
        self.native_prediction_scale = native_prediction_scale
        self.models: dict[CanonicalAction, object] = {}

    @classmethod
    def from_frozen_ordinal_artifacts(cls, model_dir: Path) -> "FiveEBMRanker":
        ranker = cls(native_prediction_scale="ORDINAL_0_3")
        for action in CanonicalAction:
            path = Path(model_dir) / f"{action.value}.joblib"
            if not path.is_file():
                raise RuntimeError(f"missing frozen action model: {path}")
            ranker.models[action] = joblib.load(path)
        return ranker

    @staticmethod
    def _regressor_class():
        try:
            module = import_module("interpret.glassbox")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "FiveEBMRanker requires the optional 'interpret' package. "
                "Install requirements-recommend-v2.txt in the locked local environment."
            ) from exc
        return module.ExplainableBoostingRegressor

    def fit(
        self,
        frame: pd.DataFrame,
        targets: dict[CanonicalAction, pd.Series],
        sample_weights: dict[CanonicalAction, pd.Series] | None = None,
    ) -> "FiveEBMRanker":
        missing = set(FEATURE_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"training frame is missing features: {sorted(missing)}")
        regressor_class = self._regressor_class()
        feature_list = list(FEATURE_COLUMNS)
        for action in CanonicalAction:
            if action not in targets:
                raise ValueError(f"missing relevance target for {action.value}")
            target = pd.to_numeric(targets[action], errors="coerce")
            retained = target.notna() & target.ge(0) & target.le(3)
            if int(retained.sum()) < 30:
                raise ValueError(f"insufficient retained labels for {action.value}")
            model = regressor_class(**self.model_parameters)
            fit_kwargs: dict[str, object] = {}
            if sample_weights is not None:
                if action not in sample_weights:
                    raise ValueError(f"missing sample weights for {action.value}")
                weights = pd.to_numeric(sample_weights[action], errors="coerce")
                if weights.isna().any() or (weights < 0).any():
                    raise ValueError(f"invalid sample weights for {action.value}")
                fit_kwargs["sample_weight"] = weights.loc[retained].to_numpy(dtype=float)
            model.fit(
                frame.loc[retained, feature_list],
                target.loc[retained] / 3.0,
                **fit_kwargs,
            )
            self.models[action] = model
        return self

    @staticmethod
    def _local_explanation(model: object, frame: pd.DataFrame) -> tuple[tuple[str, float], ...]:
        try:
            payload = model.explain_local(frame).data(0)
            names = payload.get("names", ())
            scores = payload.get("scores", ())
            ranked = sorted(
                ((str(name), float(score)) for name, score in zip(names, scores, strict=False)),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            return tuple(ranked[:8])
        except Exception:
            return ()

    def score(
        self,
        features: RecommendationFeatures,
        eligible_actions: tuple[CanonicalAction, ...],
    ) -> tuple[ActionScore, ...]:
        frame = feature_frame(features)
        results: list[ActionScore] = []
        for action in eligible_actions:
            model = self.models.get(action)
            if model is None:
                raise RuntimeError(f"no fitted relevance model for {action.value}")
            raw_prediction = float(np.asarray(model.predict(frame))[0])
            prediction = canonical_ordinal_score_from_model_prediction(raw_prediction)
            bounded = (
                public_score_from_ordinal_prediction(prediction)
                if self.native_prediction_scale == "ORDINAL_0_3"
                else float(np.clip(prediction, 0.0, 1.0))
            )
            explanation = self._local_explanation(model, frame)
            results.append(ActionScore(action, bounded, explanation))
        results.sort(key=lambda item: (-item.score, item.action.value))
        return tuple(results)


class FixedActionRanker:
    """Deterministic ranker used only in unit tests and protocol fixtures."""

    def __init__(self, scores: dict[CanonicalAction, float]) -> None:
        self.scores = dict(scores)

    def score(
        self,
        features: RecommendationFeatures,
        eligible_actions: tuple[CanonicalAction, ...],
    ) -> tuple[ActionScore, ...]:
        del features
        results = [ActionScore(action, self.scores[action]) for action in eligible_actions]
        return tuple(sorted(results, key=lambda item: (-item.score, item.action.value)))


__all__ = [
    "ActionRanker",
    "FEATURE_COLUMNS",
    "FiveEBMRanker",
    "FixedActionRanker",
    "ORDINAL_RELEVANCE_MAX",
    "canonical_ordinal_score_from_model_prediction",
    "feature_frame",
    "public_score_from_ordinal_prediction",
]
