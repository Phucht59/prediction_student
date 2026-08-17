"""Hybrid data processing, splitting, contract, and feature extraction modules."""

from src.hybrid.contracts import HybridDataView, MaskedStandardScaler, assert_train_only_fit
from src.hybrid.data.common import make_deterministic_id, sha256_hash_str, truncate_history
from src.hybrid.data.oulad import (
    OULAD_CATEGORICAL_CONTEXT,
    OULAD_FORBIDDEN_PREDICTORS,
    OULAD_NUMERIC_CONTEXT,
    OULAD_SENSITIVE_CONTEXT,
    OULAD_TEMPORAL_CHANNELS,
    build_compact_vle_daily,
    compute_weekly_features_at_cutoff,
    load_assessment_events,
    load_oulad_static_tables,
)
from src.hybrid.data.preprocessing import TabularContextPreprocessor
from src.hybrid.data.splits import (
    SplitManifest,
    create_group_stratified_splits,
    verify_inner_group_disjointness,
    verify_inner_no_outer_test,
    verify_split_disjointness,
)
from src.hybrid.data.tabular import build_oulad_tabular_baseline, build_uci_tabular_baseline
from src.hybrid.data.uci import (
    UCI_CATEGORICAL_CONTEXT,
    UCI_FORBIDDEN_PREDICTORS,
    UCI_NUMERIC_CONTEXT,
    UCI_QUASI_IDENTITY_FIELDS,
    build_uci_combined,
    build_uci_stage_view,
    load_raw_uci,
)

__all__ = [
    "HybridDataView",
    "MaskedStandardScaler",
    "assert_train_only_fit",
    "make_deterministic_id",
    "sha256_hash_str",
    "truncate_history",
    "SplitManifest",
    "create_group_stratified_splits",
    "verify_split_disjointness",
    "verify_inner_no_outer_test",
    "TabularContextPreprocessor",
    "UCI_QUASI_IDENTITY_FIELDS",
    "UCI_CATEGORICAL_CONTEXT",
    "UCI_NUMERIC_CONTEXT",
    "UCI_FORBIDDEN_PREDICTORS",
    "load_raw_uci",
    "build_uci_combined",
    "build_uci_stage_view",
    "OULAD_SENSITIVE_CONTEXT",
    "OULAD_CATEGORICAL_CONTEXT",
    "OULAD_NUMERIC_CONTEXT",
    "OULAD_FORBIDDEN_PREDICTORS",
    "OULAD_TEMPORAL_CHANNELS",
    "load_oulad_static_tables",
    "build_compact_vle_daily",
    "load_assessment_events",
    "compute_weekly_features_at_cutoff",
    "build_uci_tabular_baseline",
    "build_oulad_tabular_baseline",
]
