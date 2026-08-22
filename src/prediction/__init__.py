"""Canonical binary student-risk prediction API.

Public model: Hybrid CNN–BiLSTM. UCI and OULAD share the architecture.
Information states are views of one fitted model per dataset.
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
