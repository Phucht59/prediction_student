"""Legacy annotation adapter for converting legacy expert & LLM reviews to V2 canonical actions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import CanonicalAction

LEGACY_TO_V2_MAP = {
    "ASSESSMENT_COMPLETION": CanonicalAction.ASSESSMENT_COMPLETION,
    "STUDY_SCHEDULE": CanonicalAction.STUDY_REGULARITY,
    "STUDY_REGULARITY": CanonicalAction.STUDY_REGULARITY,
    "LEARNING_CONSOLIDATION": CanonicalAction.TARGETED_CONTENT_REVIEW,
    "CONTENT_REVIEW": CanonicalAction.TARGETED_CONTENT_REVIEW,
    "TARGETED_REVISION": CanonicalAction.TARGETED_CONTENT_REVIEW,
    "TARGETED_CONTENT_REVIEW": CanonicalAction.TARGETED_CONTENT_REVIEW,
    "RETRIEVAL_PRACTICE": CanonicalAction.QUIZ_RETRIEVAL_PRACTICE,
    "PRACTICE_EXERCISES": CanonicalAction.QUIZ_RETRIEVAL_PRACTICE,
    "ASSESSMENT_PREPARATION": CanonicalAction.QUIZ_RETRIEVAL_PRACTICE,
    "QUIZ_RETRIEVAL_PRACTICE": CanonicalAction.QUIZ_RETRIEVAL_PRACTICE,
    "RECOVER_ENGAGEMENT": CanonicalAction.RECOVER_ENGAGEMENT,
}


@dataclass(frozen=True)
class NormalizedAnnotationRecord:
    case_id: str
    action: CanonicalAction | None
    relevance_score: int  # 0..3
    reviewer_id: str
    reviewer_type: str  # REAL_HUMAN_REVIEW, REAL_LLM_GENERATED_REVIEW, LEGACY_WEAK_SOURCE
    model_name: str | None
    prompt_version: str | None
    evidence_ids: tuple[str, ...]
    contraindication_detected: bool
    safety_flag: bool
    abstain: bool
    original_action_id: str
    candidate_order: int | None = None


def map_legacy_action(
    legacy_action_id: str,
    evidence_context: dict[str, Any] | None = None,
) -> CanonicalAction | None:
    """Map legacy action string to V2 CanonicalAction.

    For VLE_ENGAGEMENT, map to RECOVER_ENGAGEMENT only if evidence shows engagement drop
    or inactivity streak; otherwise return None (ABSTAIN).
    """
    cleaned = legacy_action_id.strip().upper()
    if cleaned in LEGACY_TO_V2_MAP:
        return LEGACY_TO_V2_MAP[cleaned]

    if cleaned in {"VLE_ENGAGEMENT", "ENGAGEMENT_RECOVERY"}:
        if evidence_context is not None:
            inactivity = evidence_context.get("inactivity_streak", 0)
            recent_drop = evidence_context.get("recent_activity_trend", 0.0)
            if inactivity > 3 or recent_drop < -0.2:
                return CanonicalAction.RECOVER_ENGAGEMENT
        return None  # ABSTAIN if evidence is weak or unconfirmed

    return None


def adapt_legacy_review_record(
    record: dict[str, Any],
    case_evidence: dict[str, Any] | None = None,
) -> NormalizedAnnotationRecord:
    """Adapt a single raw annotation record into a NormalizedAnnotationRecord."""
    case_id = str(record["case_id"])
    orig_action_id = str(record["action_id"])
    reviewer_id = str(record.get("reviewer_id", record.get("expert_id", "unknown")))
    reviewer_type = str(record.get("reviewer_type", "LEGACY_WEAK_SOURCE"))

    # Check for abstain flag
    is_abstain = bool(record.get("abstain", False))
    raw_score = record.get("relevance_score")

    if raw_score is None or raw_score == "" or is_abstain:
        mapped_action = None
        score = -1
        is_abstain = True
    else:
        mapped_action = map_legacy_action(orig_action_id, case_evidence)
        if mapped_action is None:
            score = -1
            is_abstain = True
        else:
            score = int(raw_score)
            if score not in (0, 1, 2, 3):
                raise ValueError(f"relevance score must be in 0..3, got {score}")

    return NormalizedAnnotationRecord(
        case_id=case_id,
        action=mapped_action,
        relevance_score=score,
        reviewer_id=reviewer_id,
        reviewer_type=reviewer_type,
        model_name=record.get("model_name"),
        prompt_version=record.get("prompt_version"),
        evidence_ids=tuple(record.get("evidence_ids", ())),
        contraindication_detected=bool(record.get("contraindication_detected", False)),
        safety_flag=bool(record.get("safety_flag", False)),
        abstain=is_abstain,
        original_action_id=orig_action_id,
        candidate_order=record.get("candidate_order"),
    )
