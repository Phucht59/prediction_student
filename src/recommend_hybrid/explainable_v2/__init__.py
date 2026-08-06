"""Risk-guided explainable recommendation V2 public API."""

from .contracts import (
    ActionScore,
    CanonicalAction,
    FeasibilityResult,
    RecommendationDecision,
    RecommendationFeatures,
    RiskBand,
    RiskThresholds,
    RouteStatus,
    SafetyThresholds,
)
from .feasibility import evaluate_action, feasible_actions
from .pipeline import ExplainableRecommendationPipeline
from .ranker import FiveEBMRanker, FixedActionRanker
from .risk_policy import stratify_risk
from .safety_router import route_ranked_actions

__all__ = [
    "ActionScore",
    "CanonicalAction",
    "ExplainableRecommendationPipeline",
    "FeasibilityResult",
    "FiveEBMRanker",
    "FixedActionRanker",
    "RecommendationDecision",
    "RecommendationFeatures",
    "RiskBand",
    "RiskThresholds",
    "RouteStatus",
    "SafetyThresholds",
    "evaluate_action",
    "feasible_actions",
    "route_ranked_actions",
    "stratify_risk",
]
