"""Compatibility boundary from canonical PredictionResult to recommendation.

Recommendation ranking, actions, EBM logic, weak labels, and safety routing are
unchanged. Only the prediction provenance boundary is canonicalized here.
"""

from __future__ import annotations

from collections.abc import Iterable

from src.prediction.contracts import PredictionResult


def prediction_result_to_features(result: PredictionResult) -> dict:
    """Expose risk probability without inspecting any model class or dataset head."""

    return result.recommendation_features()


def prediction_results_to_features(results: Iterable[PredictionResult]) -> list[dict]:
    return [prediction_result_to_features(result) for result in results]


__all__ = ["PredictionResult", "prediction_result_to_features", "prediction_results_to_features"]
