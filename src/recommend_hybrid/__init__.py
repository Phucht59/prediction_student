"""Phase 2 foundation for the Hybrid CNN-BiLSTM Learning Support Recommender."""

from .action_catalog import ActionCatalog
from .candidate_generator import HybridCandidateGenerator
from .contracts import ObservedLearningState, PredictionContext, StudentRepresentation
from .prediction_adapter import HybridPredictionAdapter
from .oulad.policy import RecommendHybridOULAD
from .uci.policy import RecommendHybridUCI

__all__ = [
    "ActionCatalog",
    "HybridCandidateGenerator",
    "HybridPredictionAdapter",
    "RecommendHybridOULAD",
    "RecommendHybridUCI",
    "ObservedLearningState",
    "PredictionContext",
    "StudentRepresentation",
]
