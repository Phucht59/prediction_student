"""Hybrid learning-support recommendation APIs.

The scientifically validated final boundary is the conditional action ranker in
``recommend_hybrid.final``. Legacy policy/plan APIs remain importable for
research reproducibility, but they are not validated as an end-to-end automatic
recommendation system.
"""

from .action_catalog import ActionCatalog
from .candidate_generator import HybridCandidateGenerator
from .common.service import HybridRecommendationService
from .contracts import ObservedLearningState, PredictionContext, StudentRepresentation
from .final import (
    MODEL_SCORE_AUTHORITY,
    SCIENTIFIC_ACTION_ORDER,
    ConditionalHybridActionRanker,
)
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
    "MODEL_SCORE_AUTHORITY",
    "RecommendHybridPipeline",
    "RecommendHybridOULAD",
    "RecommendHybridUCI",
    "ObservedLearningState",
    "PredictionContext",
    "SCIENTIFIC_ACTION_ORDER",
    "StudentRepresentation",
    "OULADPlanRequest",
    "UCIPlanRequest",
]
