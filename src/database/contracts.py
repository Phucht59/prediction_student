"""Stable contracts for the version-neutral runtime database API."""

from __future__ import annotations

from enum import Enum


class Schema(str, Enum):
    SYSTEM = "system"
    CATALOG = "catalog"
    ML = "ml"
    RECOMMENDATION = "recommendation"


class RunType(str, Enum):
    OFFICIAL_FINAL = "official_final"
    COMPARATOR_COMPLETION = "comparator_completion"
    EVALUATION_ONLY = "evaluation_only"


class ReviewType(str, Enum):
    ADVISOR = "advisor"
    EXPERT = "expert"
    FOLLOW_UP = "follow_up"
    SYSTEM_VALIDATION = "system_validation"


FINAL_APPLICATION_SEARCH_PATH = ("catalog", "ml", "recommendation")
EXPERT_PENDING_STATUS = "PENDING_EXPERT_LABELS"


__all__ = [
    "EXPERT_PENDING_STATUS",
    "FINAL_APPLICATION_SEARCH_PATH",
    "ReviewType",
    "RunType",
    "Schema",
]
