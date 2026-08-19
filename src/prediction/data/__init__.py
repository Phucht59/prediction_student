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
)
from .final100 import FINAL100_ENDPOINT, assert_final100_view, build_final100_view

__all__ = [
    "UnifiedHybridData",
    "load_raw_uci",
    "build_uci_combined",
    "build_uci_stage_view",
    "apply_d3_variant",
    "build_oulad_array_view",
    "load_oulad_static_tables",
    "validate_oulad_predictor_columns",
    "oulad_risk_target",
    "build_oulad_information_state",
    "filter_events_cutoff_safe",
    "events_strictly_before_cutoff",
    "assert_predictor_contract",
    "FINAL100_ENDPOINT",
    "build_final100_view",
    "assert_final100_view",
]
