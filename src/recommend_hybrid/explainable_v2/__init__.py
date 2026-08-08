"""Risk-guided explainable recommendation V2 public API."""

from .audits import (
    FORBIDDEN_RANKER_FEATURES,
    assert_pre_cutoff_lineage,
    assert_ranker_schema,
    assert_student_disjoint_splits,
    context_permutation_degradation,
    permute_context_by_query,
)
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
from .metrics import RankingMetrics, evaluate_grouped_ranking, grouped_bootstrap_difference
from .pipeline import ExplainableRecommendationPipeline
from .ranker import FiveEBMRanker, FixedActionRanker
from .risk_policy import stratify_risk
from .safety_router import route_ranked_actions

__all__ = [
    "FORBIDDEN_RANKER_FEATURES",
    "ActionScore",
    "CanonicalAction",
    "ExplainableRecommendationPipeline",
    "FeasibilityResult",
    "FiveEBMRanker",
    "FixedActionRanker",
    "RankingMetrics",
    "RecommendationDecision",
    "RecommendationFeatures",
    "RiskBand",
    "RiskThresholds",
    "RouteStatus",
    "SafetyThresholds",
    "assert_pre_cutoff_lineage",
    "assert_ranker_schema",
    "assert_student_disjoint_splits",
    "context_permutation_degradation",
    "evaluate_action",
    "evaluate_grouped_ranking",
    "feasible_actions",
    "grouped_bootstrap_difference",
    "permute_context_by_query",
    "route_ranked_actions",
    "stratify_risk",
]
