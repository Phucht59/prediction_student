"""Canonical action identity schema for the validated conditional ranker.

The integrated action head was trained with five fixed action slots. Runtime
policy actions may use different public IDs, so every action must be mapped by
identity rather than by the position in a caller-supplied list.
"""
from __future__ import annotations

ACTION_ORDER: tuple[str, ...] = (
    "ASSESSMENT_COMPLETION",
    "STUDY_REGULARITY",
    "VLE_ENGAGEMENT",
    "QUIZ_OR_RETRIEVAL_PRACTICE",
    "CONTENT_REVIEW",
)
ACTION_COUNT = len(ACTION_ORDER)
ACTION_INDEX = {action_id: index for index, action_id in enumerate(ACTION_ORDER)}

# Public policy/catalog IDs mapped to the scientific action identities used by
# the integrated head and the held-out evidence.
ACTION_ALIASES: dict[str, str] = {
    "ASSESSMENT_COMPLETION": "ASSESSMENT_COMPLETION",
    "STUDY_REGULARITY": "STUDY_REGULARITY",
    "STUDY_SCHEDULE": "STUDY_REGULARITY",
    "VLE_ENGAGEMENT": "VLE_ENGAGEMENT",
    "QUIZ_OR_RETRIEVAL_PRACTICE": "QUIZ_OR_RETRIEVAL_PRACTICE",
    "RETRIEVAL_PRACTICE": "QUIZ_OR_RETRIEVAL_PRACTICE",
    "CONTENT_REVIEW": "CONTENT_REVIEW",
    "LEARNING_CONSOLIDATION": "CONTENT_REVIEW",
}


def canonical_action_id(value: object) -> str:
    """Return the trained action identity or reject unsupported actions."""

    if value is None:
        raise ValueError("every eligible action requires an explicit action_id")
    supplied = str(value).strip().upper()
    canonical = ACTION_ALIASES.get(supplied)
    if canonical is None:
        raise ValueError(
            f"action {supplied!r} is outside the validated conditional action set"
        )
    return canonical


__all__ = [
    "ACTION_ALIASES",
    "ACTION_COUNT",
    "ACTION_INDEX",
    "ACTION_ORDER",
    "canonical_action_id",
]
