"""Hybrid CNN-BiLSTM Learning Support Recommender public API."""

from .action_catalog import ActionCatalog
from .candidate_generator import HybridCandidateGenerator
from .common.service import HybridRecommendationService
from .contracts import ObservedLearningState, PredictionContext, StudentRepresentation
from .counterfactual import (
    CounterfactualPlanResult,
    CounterfactualPlanStatus,
    OULADReferenceProfile,
    OULADReferenceProfileBuilder,
    RecommendHybridCounterfactualPipeline,
)
from .oulad.policy import RecommendHybridOULAD
from .pipeline import OULADPlanRequest, RecommendHybridPipeline, UCIPlanRequest
from .prediction_adapter import HybridPredictionAdapter
from .uci.policy import RecommendHybridUCI

__all__ = [
    "ActionCatalog",
    "CounterfactualPlanResult",
    "CounterfactualPlanStatus",
    "HybridCandidateGenerator",
    "HybridPredictionAdapter",
    "HybridRecommendationService",
    "OULADPlanRequest",
    "OULADReferenceProfile",
    "OULADReferenceProfileBuilder",
    "ObservedLearningState",
    "PredictionContext",
    "RecommendHybridCounterfactualPipeline",
    "RecommendHybridOULAD",
    "RecommendHybridPipeline",
    "RecommendHybridUCI",
    "StudentRepresentation",
    "UCIPlanRequest",
]
