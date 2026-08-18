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
    "FINAL100_ENDPOINT",
    "build_final100_view",
    "assert_final100_view",
]
