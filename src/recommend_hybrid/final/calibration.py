"""Validation-only calibration for independently trained action relevance models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .contracts import ActionScore, CanonicalAction, RecommendationFeatures
from .ranker import ActionRanker


@dataclass
class PerActionIsotonicCalibrator:
    """Calibrate each action score against ordinal relevance on validation only."""

    minimum_rows: int = 30

    def __post_init__(self) -> None:
        if self.minimum_rows < 10:
            raise ValueError("minimum_rows must be at least 10")
        self.models: dict[CanonicalAction, IsotonicRegression] = {}

    def fit_action(
        self,
        action: CanonicalAction,
        raw_scores: np.ndarray,
        ordinal_targets: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        scores = np.asarray(raw_scores, dtype=float).reshape(-1)
        targets = np.asarray(ordinal_targets, dtype=float).reshape(-1)
        if len(scores) != len(targets):
            raise ValueError("calibration scores and targets must align")
        retained = np.isfinite(scores) & np.isfinite(targets) & (targets >= 0) & (targets <= 3)
        if int(retained.sum()) < self.minimum_rows:
            raise ValueError(f"insufficient validation rows to calibrate {action.value}")
        weights = None
        if sample_weight is not None:
            candidate = np.asarray(sample_weight, dtype=float).reshape(-1)
            if len(candidate) != len(scores) or not np.isfinite(candidate).all() or (candidate < 0).any():
                raise ValueError("invalid calibration sample weights")
            weights = candidate[retained]
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(scores[retained], targets[retained] / 3.0, sample_weight=weights)
        self.models[action] = model

    def transform(self, action: CanonicalAction, raw_score: float) -> float:
        model = self.models.get(action)
        if model is None:
            raise RuntimeError(f"no validation calibrator for {action.value}")
        value = float(model.predict(np.asarray([raw_score], dtype=float))[0])
        return min(max(value, 0.0), 1.0)


class CalibratedActionRanker:
    """Apply validation-fitted calibration after a base action ranker."""

    def __init__(
        self,
        base_ranker: ActionRanker,
        calibrator: PerActionIsotonicCalibrator,
    ) -> None:
        self.base_ranker = base_ranker
        self.calibrator = calibrator

    def score(
        self,
        features: RecommendationFeatures,
        eligible_actions: tuple[CanonicalAction, ...],
    ) -> tuple[ActionScore, ...]:
        raw = self.base_ranker.score(features, eligible_actions)
        calibrated = [
            ActionScore(
                action=item.action,
                score=self.calibrator.transform(item.action, item.score),
                explanation=item.explanation,
            )
            for item in raw
        ]
        calibrated.sort(key=lambda item: (-item.score, item.action.value))
        return tuple(calibrated)


__all__ = ["CalibratedActionRanker", "PerActionIsotonicCalibrator"]
