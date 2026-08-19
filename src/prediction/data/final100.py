"""OULAD 100% information-state alias. Not a separate model."""

from __future__ import annotations

from .common import UnifiedHybridData
from .oulad import build_oulad_array_view


FINAL100_ENDPOINT = "100pct"
HISTORICAL_FINAL100_ALIAS = "FINAL-100"


def build_final100_view(**kwargs) -> UnifiedHybridData:
    """Build the 100% observation view of the same OULAD Hybrid."""
    kwargs["endpoint"] = "100pct"
    return build_oulad_array_view(**kwargs)


def assert_final100_view(view: UnifiedHybridData) -> None:
    if view.metadata.get("information_state") != "100pct":
        raise ValueError("expected the OULAD 100pct information state of the same Hybrid")


__all__ = ["FINAL100_ENDPOINT", "build_final100_view", "assert_final100_view"]
