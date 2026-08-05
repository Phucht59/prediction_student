"""Stage-aware causal recommendation evaluation.

This namespace estimates observational action effects at the exact prediction
landmark. It does not claim randomized-treatment or deployment effectiveness.
"""

from .aipw import AIPWConfig, AIPWResult, CrossFittedAIPW
from .diagnostics import (
    IdentifiabilityReport,
    IdentifiabilityThresholds,
    assess_identifiability,
    effective_sample_size,
    stabilized_iptw,
    standardized_mean_difference,
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

__all__ = [
    "AIPWConfig",
    "AIPWResult",
    "CausalActionDecision",
    "CausalPolicyThresholds",
    "CrossFittedAIPW",
    "IdentifiabilityReport",
    "IdentifiabilityThresholds",
    "LANDMARK_STAGES",
    "LandmarkStage",
    "RecommendationEvent",
    "STAGE_ORDER",
    "StudyRegularityTreatmentDefinition",
    "TargetTrialProtocol",
    "assess_identifiability",
    "effective_sample_size",
    "gate_causal_action",
    "resolve_recommendation_lifecycle",
    "stabilized_iptw",
    "stage_from_fraction",
    "standardized_mean_difference",
    "study_regularity_components",
    "study_regularity_score",
    "validate_temporal_columns",
]
