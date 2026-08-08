"""Final risk-guided recommendation public API."""

from .audits import (
    FORBIDDEN_RANKER_FEATURES,
    assert_pre_cutoff_lineage,
    assert_ranker_schema,
    assert_student_disjoint_splits,
    context_permutation_degradation,
    permute_context_by_query,
)
from .calibration import CalibratedActionRanker, PerActionIsotonicCalibrator
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
from .model_selection import CandidateEvidence, select_final_candidate
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

RecommendationPipeline = ExplainableRecommendationPipeline

__all__ = [
    "ABSTAIN",
    "FORBIDDEN_RANKER_FEATURES",
    "ActionScore",
    "CalibratedActionRanker",
    "CandidateEvidence",
    "CanonicalAction",
    "ExplainableRecommendationPipeline",
    "FeasibilityResult",
    "FiveEBMRanker",
    "FixedActionRanker",
    "PerActionIsotonicCalibrator",
    "RankingMetrics",
    "RecommendationDecision",
    "RecommendationFeatures",
    "RecommendationPipeline",
    "RiskBand",
    "RiskThresholds",
    "RouteStatus",
    "SafetyThresholds",
    "WeakLabelSource",
    "aggregate_votes",
    "assert_pre_cutoff_lineage",
    "assert_ranker_schema",
    "assert_student_disjoint_splits",
    "context_permutation_degradation",
    "evaluate_action",
    "evaluate_grouped_ranking",
    "feasible_actions",
    "fit_label_model",
    "grouped_bootstrap_difference",
    "permute_context_by_query",
    "route_ranked_actions",
    "select_final_candidate",
    "source_correlation_audit",
    "stratify_risk",
    "validate_vote_matrix",
]
