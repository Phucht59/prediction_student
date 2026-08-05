"""Stage-aware causal recommendation evaluation.

This namespace estimates observational action effects at the exact prediction
landmark. It does not claim randomized-treatment or deployment effectiveness.
"""

from .aipw import AIPWConfig, AIPWResult, CrossFittedAIPW
from .bootstrap import ClusterBootstrapResult, cluster_bootstrap_mean
from .diagnostics import (
    IdentifiabilityReport,
    IdentifiabilityThresholds,
    assess_identifiability,
    effective_sample_size,
    stabilized_iptw,
    standardized_mean_difference,
)
from .imbalance import (
    IMBALANCE_MODES,
    ImbalanceStudyResult,
    run_frozen_embedding_imbalance_study,
    select_validation_threshold,
)
from .pipeline import (
    StageActionEvaluation,
    StageActionTrialData,
    StageAwareCausalEvaluator,
)
from .policy import (
    CausalActionDecision,
    CausalPolicyThresholds,
    RecommendationEvent,
    gate_causal_action,
    resolve_recommendation_lifecycle,
)
from .protocol import (
    LANDMARK_STAGES,
    STAGE_ORDER,
    LandmarkStage,
    TargetTrialProtocol,
    stage_from_fraction,
    validate_temporal_columns,
)
from .study_regularity import (
    StudyRegularityTreatmentDefinition,
    study_regularity_components,
    study_regularity_score,
)
from .treatments import (
    ACTION_TREATMENT_SPECS,
    ActionTreatmentSpec,
    FittedActionTreatmentRule,
    fit_action_treatment_rule,
)

__all__ = [
    "ACTION_TREATMENT_SPECS",
    "AIPWConfig",
    "AIPWResult",
    "ActionTreatmentSpec",
    "CausalActionDecision",
    "CausalPolicyThresholds",
    "ClusterBootstrapResult",
    "CrossFittedAIPW",
    "FittedActionTreatmentRule",
    "IMBALANCE_MODES",
    "IdentifiabilityReport",
    "IdentifiabilityThresholds",
    "ImbalanceStudyResult",
    "LANDMARK_STAGES",
    "LandmarkStage",
    "RecommendationEvent",
    "STAGE_ORDER",
    "StageActionEvaluation",
    "StageActionTrialData",
    "StageAwareCausalEvaluator",
    "StudyRegularityTreatmentDefinition",
    "TargetTrialProtocol",
    "assess_identifiability",
    "cluster_bootstrap_mean",
    "effective_sample_size",
    "fit_action_treatment_rule",
    "gate_causal_action",
    "resolve_recommendation_lifecycle",
    "run_frozen_embedding_imbalance_study",
    "select_validation_threshold",
    "stabilized_iptw",
    "stage_from_fraction",
    "standardized_mean_difference",
    "study_regularity_components",
    "study_regularity_score",
    "validate_temporal_columns",
]
