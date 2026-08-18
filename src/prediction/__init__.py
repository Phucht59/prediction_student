"""Canonical binary student-risk prediction API.

The public prediction surface intentionally contains one architecture only:
``Hybrid``. Dataset differences belong to adapters and fitted instances.
"""

from .contracts import PredictionResult, UCI_RISK_RULE, OULAD_RISK_RULE
from .model import Hybrid, HybridConfig
from .registry import ACTIVE_BASELINES, ACTIVE_PREDICTION_REGISTRY

__all__ = [
    "Hybrid",
    "HybridConfig",
    "PredictionResult",
    "UCI_RISK_RULE",
    "OULAD_RISK_RULE",
    "ACTIVE_BASELINES",
    "ACTIVE_PREDICTION_REGISTRY",
]
