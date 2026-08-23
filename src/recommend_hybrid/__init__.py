"""Recommendation V: Hybrid CNN–BiLSTM risk-guided actions.

Serving runtime is ``src.recommend_hybrid.v3``. The package still re-exports
the frozen public contracts used by tests and the live database CLI.
"""

from .final import (
    ActionScore,
    CanonicalAction,
    ExplainableRecommendationPipeline,
    FeasibilityResult,
    FiveEBMRanker,
    RecommendationDecision,
    RecommendationFeatures,
    RecommendationPipeline,
    RiskBand,
    RiskThresholds,
    RouteStatus,
    SafetyThresholds,
    evaluate_action,
    feasible_actions,
    route_ranked_actions,
    stratify_risk,
)

__all__ = [
    "ActionScore",
    "CanonicalAction",
    "ExplainableRecommendationPipeline",
    "FeasibilityResult",
    "FiveEBMRanker",
    "RecommendationDecision",
    "RecommendationFeatures",
    "RecommendationPipeline",
    "RiskBand",
    "RiskThresholds",
    "RouteStatus",
    "SafetyThresholds",
    "evaluate_action",
    "feasible_actions",
    "route_ranked_actions",
    "stratify_risk",
]
