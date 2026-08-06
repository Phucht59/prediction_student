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
from .metrics import RankingMetrics, evaluate_grouped_ranking, grouped_bootstrap_difference
from .pipeline import ExplainableRecommendationPipeline
from .ranker import FiveEBMRanker, FixedActionRanker
from .risk_policy import stratify_risk
from .safety_router import route_ranked_actions
from .weak_labels import (
    ABSTAIN,
    WeakLabelSource,
    aggregate_votes,
    fit_label_model,
    source_correlation_audit,
    validate_vote_matrix,
)

__all__ = [
    "ABSTAIN",
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
    "WeakLabelSource",
    "aggregate_votes",
    "evaluate_action",
    "evaluate_grouped_ranking",
    "feasible_actions",
    "fit_label_model",
    "grouped_bootstrap_difference",
    "route_ranked_actions",
    "source_correlation_audit",
    "stratify_risk",
    "validate_vote_matrix",
]
