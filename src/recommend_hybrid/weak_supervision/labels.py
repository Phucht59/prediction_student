"""Locked label values for scientific labeling."""

from enum import IntEnum


class TargetLabel(IntEnum):
    INAPPROPRIATE = 0
    CONDITIONAL = 1
    APPROPRIATE = 2


class RelevanceGrade(IntEnum):
    INAPPROPRIATE = 0
    CONDITIONAL = 1
    APPROPRIATE = 2


LF_ABSTAIN = -1
TARGET_VALUES = frozenset(int(value) for value in TargetLabel)
RELEVANCE_VALUES = frozenset(int(value) for value in RelevanceGrade)

__all__ = [
    "LF_ABSTAIN",
    "RELEVANCE_VALUES",
    "TARGET_VALUES",
    "RelevanceGrade",
    "TargetLabel",
]
