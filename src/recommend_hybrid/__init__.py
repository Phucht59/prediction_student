"""Hybrid CNN-BiLSTM risk-guided recommendation package.

The production recommendation authority is ``src.recommend_hybrid.final``.
Older research and compatibility modules remain in the repository for lineage,
but they are not exported as the validated recommendation model.
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
