"""Deterministic, identity-disjoint Panel A/B sampling."""

from .panels import PANEL_BANDS, sample_panel, validate_panels

__all__ = ["PANEL_BANDS", "sample_panel", "validate_panels"]
