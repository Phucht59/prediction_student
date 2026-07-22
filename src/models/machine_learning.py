"""Canonical comparator catalog; no fitting is performed here."""

from src.final_release.catalog import COMPARISON_MODELS

MACHINE_LEARNING_COMPARATORS = dict(COMPARISON_MODELS[3:])

__all__ = ["MACHINE_LEARNING_COMPARATORS"]
