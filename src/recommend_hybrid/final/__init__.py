"""Production API for the scientifically frozen recommendation release.

The validated Recommendation V2 implementation is exposed here through a
version-neutral namespace.  The implementation package
``recommend_hybrid.explainable_v2`` is intentionally retained unchanged as
scientific lineage so the frozen hashes and audit trail remain reproducible.
New application code should import recommendation components from this package.
"""

from ..explainable_v2 import (
    ActionScore,
    CanonicalAction,
    ExplainableRecommendationPipeline,
    FeasibilityResult,
    FiveEBMRanker,
    RecommendationDecision,
    RecommendationFeatures,
    RiskBand,
    RiskThresholds,
    RouteStatus,
    SafetyThresholds,
    evaluate_action,
    feasible_actions,
    route_ranked_actions,
    stratify_risk,
)

RecommendationPipeline = ExplainableRecommendationPipeline

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
