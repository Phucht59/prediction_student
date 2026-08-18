"""Phase 7 weak supervision: matrices, aggregation, and silver labels."""

from .diagnostics import assign_quality_status, pre_snorkel_diagnostics
from .label_model import fit_label_models, majority_vote
from .matrix import (
    A4GemmaGateError,
    A4SourceGateError,
    FINAL_ACTIONS,
    SOURCES_BY_ACTION,
    build_matrices,
    load_canonical_sources,
    load_sources,
    validate_phase7_authority,
    validate_source_manifest,
)
from .silver import SILVER_COLUMNS, attach_feasibility, validate_silver

__all__ = [
    "A4GemmaGateError",
    "A4SourceGateError",
    "FINAL_ACTIONS",
    "SILVER_COLUMNS",
    "SOURCES_BY_ACTION",
    "assign_quality_status",
    "attach_feasibility",
    "build_matrices",
    "fit_label_models",
    "load_canonical_sources",
    "load_sources",
    "majority_vote",
    "pre_snorkel_diagnostics",
    "validate_phase7_authority",
    "validate_silver",
    "validate_source_manifest",
]
