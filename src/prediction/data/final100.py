"""Named OULAD FINAL-100 endpoint wrapper."""

from __future__ import annotations

from .common import UnifiedHybridData
from .oulad import build_oulad_array_view


FINAL100_ENDPOINT = "FINAL-100"


def build_final100_view(**kwargs) -> UnifiedHybridData:
    """Build the principal OULAD endpoint with the frozen endpoint name."""
    kwargs["endpoint"] = FINAL100_ENDPOINT
    return build_oulad_array_view(**kwargs)


def assert_final100_view(view: UnifiedHybridData) -> None:
    if view.metadata.get("endpoint") != FINAL100_ENDPOINT:
        raise ValueError("expected the OULAD FINAL-100 endpoint")


__all__ = ["FINAL100_ENDPOINT", "build_final100_view", "assert_final100_view"]
