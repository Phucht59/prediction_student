"""Stable data contracts between frozen Prediction and Recommendation."""

from .prediction import PredictionArtifactAdapter, PredictionRecord
from .state import make_case_id

__all__ = ["PredictionArtifactAdapter", "PredictionRecord", "make_case_id"]
