"""Hybrid CNN-BiLSTM learning-support recommendation APIs.

`ConditionalHybridActionRanker` is the scientifically validated final module.
The older end-to-end plan pipelines remain available for compatibility, but
their recommendation-issuance accuracy is not validated.
"""

from .action_catalog import ActionCatalog
from .candidate_generator import HybridCandidateGenerator
from .common.service import HybridRecommendationService
from .contracts import ObservedLearningState, PredictionContext, StudentRepresentation
from .final import ConditionalHybridActionRanker
from .oulad.policy import RecommendHybridOULAD
from .pipeline import OULADPlanRequest, RecommendHybridPipeline, UCIPlanRequest
from .prediction_adapter import HybridPredictionAdapter
from .uci.policy import RecommendHybridUCI

__all__ = [
    "ActionCatalog",
    "ConditionalHybridActionRanker",
    "HybridCandidateGenerator",
    "HybridPredictionAdapter",
    "HybridRecommendationService",
    "RecommendHybridPipeline",
    "RecommendHybridOULAD",
    "RecommendHybridUCI",
    "ObservedLearningState",
    "PredictionContext",
    "StudentRepresentation",
    "OULADPlanRequest",
    "UCIPlanRequest",
]
