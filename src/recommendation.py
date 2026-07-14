"""Compatibility export for the single Phase D canonical recommendation builder.

Legacy v3 functions are intentionally removed: they had unsafe defaults,
unverified context-feature rules, and confidence wording that Phase D forbids.
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
    """Fail closed: the v3 materializer cannot create Phase D recommendations."""
    raise RuntimeError(
        "Legacy recommendation generation is disabled. Use the frozen-N0 "
        "Phase D prediction snapshot and build_governed_recommendation()."
    )


# These names remain only to prevent historical scripts from silently producing
# unsafe recommendations.  They intentionally do not provide an alternative
# builder or a default-value fallback.
build_recommendation = _deprecated_legacy_path
generate_learning_path_report = _deprecated_legacy_path
prepare_recommendation_features = _deprecated_legacy_path
validate_recommendation_schema = _deprecated_legacy_path
recommendation_to_persistence_row = _deprecated_legacy_path
FORBIDDEN_INPUT_COLUMNS = frozenset({"G3", "G3_raw", "true_label", "outcome", "target"})


def structural_validity_metrics(recommendation_payloads):
    """Report only whether historical payloads are present; never generate them."""
    total = len(recommendation_payloads)
    return {"count": total, "historical_only": True, "phase_d_active_builder": "governed_recommendation"}

__all__ = [
    "CLASS_NAMES", "POLICY_VERSION", "action_catalog", "advisor_decision", "assess_snapshot",
    "build_governed_recommendation", "feature_registry", "follow_up",
    "prediction_snapshot", "validate_recommendation", "validate_scores",
    "build_recommendation", "generate_learning_path_report", "prepare_recommendation_features",
    "validate_recommendation_schema", "recommendation_to_persistence_row", "structural_validity_metrics",
]
