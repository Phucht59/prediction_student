"""Phase 7 unified representation and Hybrid architecture.

This package is intentionally isolated from frozen Phase 1--6 code.  It lets
the new protocol evolve without changing historic loaders, models, or results.
"""

from .contracts import UnifiedHybridData
from .data import (
    OULAD_PHASE7_AGGREGATE_CHANNELS,
    OULAD_PHASE7_TEMPORAL_CHANNELS,
    UCI_PHASE7_AGGREGATE_CHANNELS,
    build_oulad_phase7_view,
    build_phase7_baseline_frame,
    phase7_eligible_oulad,
    build_uci_phase7_view,
)
from .model import UnifiedHybrid, UnifiedHybridConfig

__all__ = [
    "UnifiedHybridData", "UnifiedHybrid", "UnifiedHybridConfig",
    "build_uci_phase7_view", "build_oulad_phase7_view",
    "build_phase7_baseline_frame", "phase7_eligible_oulad",
    "UCI_PHASE7_AGGREGATE_CHANNELS", "OULAD_PHASE7_TEMPORAL_CHANNELS",
    "OULAD_PHASE7_AGGREGATE_CHANNELS",
]
