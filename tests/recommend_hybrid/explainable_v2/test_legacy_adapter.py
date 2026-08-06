"""Tests for Legacy Annotation Adapter."""

from __future__ import annotations

import pytest
from src.recommend_hybrid.explainable_v2.contracts import CanonicalAction
from src.recommend_hybrid.explainable_v2.legacy_annotation_adapter import (
    adapt_legacy_review_record,
    map_legacy_action,
)


def test_map_legacy_action_direct():
    assert map_legacy_action("ASSESSMENT_COMPLETION") == CanonicalAction.ASSESSMENT_COMPLETION
    assert map_legacy_action("STUDY_SCHEDULE") == CanonicalAction.STUDY_REGULARITY
    assert map_legacy_action("RETRIEVAL_PRACTICE") == CanonicalAction.QUIZ_RETRIEVAL_PRACTICE


def test_map_legacy_vle_engagement_with_evidence():
    # Weak evidence -> ABSTAIN (None)
    assert map_legacy_action("VLE_ENGAGEMENT", {"inactivity_streak": 1}) is None

    # Strong evidence -> RECOVER_ENGAGEMENT
    assert map_legacy_action("VLE_ENGAGEMENT", {"inactivity_streak": 5}) == CanonicalAction.RECOVER_ENGAGEMENT


def test_adapt_legacy_review_record():
    rec = {
        "case_id": "case_123",
        "action_id": "STUDY_SCHEDULE",
        "relevance_score": 3,
        "reviewer_id": "expert_01",
        "reviewer_type": "REAL_HUMAN_REVIEW",
    }
    norm = adapt_legacy_review_record(rec)
    assert norm.case_id == "case_123"
    assert norm.action == CanonicalAction.STUDY_REGULARITY
    assert norm.relevance_score == 3
    assert norm.abstain is False


def test_adapt_legacy_review_record_abstain():
    rec = {
        "case_id": "case_123",
        "action_id": "UNKNOWN_ACTION",
        "relevance_score": None,
        "abstain": True,
    }
    norm = adapt_legacy_review_record(rec)
    assert norm.abstain is True
    assert norm.relevance_score == -1
