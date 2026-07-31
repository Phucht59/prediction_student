"""Backward-compatible imports for the canonical project metric authority."""

from src.evaluation.metrics import (
    evaluate_binary_risk as binary_metrics,
    evaluate_multiclass as multiclass_metrics,
    expected_calibration_error as ece,
)
from src.evaluation.metrics import paired_bootstrap_macro_f1 as _paired_bootstrap


def paired_bootstrap_macro_f1(*args, **kwargs):
    """Preserve the Phase 11 keyword interface without duplicating metric math."""
    return _paired_bootstrap(
        args[0],
        args[1],
        args[2],
        left_threshold=kwargs.pop("hybrid_threshold"),
        right_threshold=kwargs.pop("comparator_threshold"),
        multiclass=kwargs.pop("multiclass"),
        **kwargs,
    )
