"""Recommendation package compatibility surface plus versioned policies.

The historical :mod:`src.recommendation` module remains on disk for immutable
source audits.  This package exposes the same public safety API while allowing
new policies to live under an explicit version namespace.
"""

from src.governed_recommendation import (  # noqa: F401
    CLASS_NAMES,
    POLICY_VERSION,
    action_catalog,
    advisor_decision,
    assess_snapshot,
    build_governed_recommendation,
    feature_registry,
    follow_up,
    prediction_snapshot,
    validate_recommendation,
    validate_scores,
)


def _deprecated_legacy_path(*_args, **_kwargs):
    raise RuntimeError(
        "Legacy recommendation generation is disabled. Use the registered "
        "governed recommendation policy."
    )


build_recommendation = _deprecated_legacy_path
generate_learning_path_report = _deprecated_legacy_path
prepare_recommendation_features = _deprecated_legacy_path
validate_recommendation_schema = _deprecated_legacy_path
recommendation_to_persistence_row = _deprecated_legacy_path


def structural_validity_metrics(recommendation_payloads):
    return {
        "count": len(recommendation_payloads),
        "historical_only": True,
        "phase_d_active_builder": "governed_recommendation",
    }


__all__ = [
    "CLASS_NAMES",
    "POLICY_VERSION",
    "action_catalog",
    "advisor_decision",
    "assess_snapshot",
    "build_governed_recommendation",
    "build_recommendation",
    "feature_registry",
    "follow_up",
    "generate_learning_path_report",
    "prediction_snapshot",
    "prepare_recommendation_features",
    "recommendation_to_persistence_row",
    "structural_validity_metrics",
    "validate_recommendation",
    "validate_recommendation_schema",
    "validate_scores",
]
"""Canonical recommendation API."""

from .engine import StudentRiskRecommendationSystem
from .schemas import RecommendationPlan

__all__ = ["RecommendationPlan", "StudentRiskRecommendationSystem"]
