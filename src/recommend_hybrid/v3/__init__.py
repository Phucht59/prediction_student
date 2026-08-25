"""Recommendation V: Hybrid CNN–BiLSTM risk + Five-EBM + deterministic plans."""

from .contracts import (
    CanonicalAction,
    RecommendationDecision,
    RecommendationFeatures,
    RiskRoute,
    RouteStatus,
    Stage,
)
from .pipeline import RecommendationV3Pipeline
from .feasibility import evaluate_action, feasible_actions
from .ranker import FiveEBMC0Ranker
from .risk_router import stratify_risk
from .safety_router import route_ranked_actions

RecommendationPipeline = RecommendationV3Pipeline

__all__ = [
    "CanonicalAction",
    "FiveEBMC0Ranker",
    "RecommendationDecision",
    "RecommendationFeatures",
    "RecommendationPipeline",
    "RecommendationV3Pipeline",
    "RiskRoute",
    "RouteStatus",
    "Stage",
    "evaluate_action",
    "feasible_actions",
    "route_ranked_actions",
    "stratify_risk",
]
