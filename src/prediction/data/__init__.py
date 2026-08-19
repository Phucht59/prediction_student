"""Dataset adapters for the shared Hybrid architecture."""

from .common import UnifiedHybridData
from .uci import build_uci_combined, build_uci_stage_view, load_raw_uci
from .oulad import (
    apply_d3_variant,
    build_oulad_array_view,
    load_oulad_static_tables,
    oulad_risk_target,
    validate_oulad_predictor_columns,
)
from .oulad_features import (
    assert_predictor_contract,
    build_oulad_information_state,
    events_strictly_before_cutoff,
    filter_events_cutoff_safe,
    fit_oulad_preprocessor,
)
from .preprocessing import ContextPreprocessor

__all__ = [
    "UnifiedHybridData",
    "ContextPreprocessor",
    "load_raw_uci",
    "build_uci_combined",
    "build_uci_stage_view",
    "apply_d3_variant",
    "build_oulad_array_view",
    "load_oulad_static_tables",
    "validate_oulad_predictor_columns",
    "oulad_risk_target",
    "build_oulad_information_state",
    "fit_oulad_preprocessor",
    "filter_events_cutoff_safe",
    "events_strictly_before_cutoff",
    "assert_predictor_contract",
]
