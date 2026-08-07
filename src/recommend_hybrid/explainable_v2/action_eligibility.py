"""Action eligibility evaluator with configurable authority policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

DEFAULT_V2_POLICY = {
    "version": "v2.0_policy",
    "thresholds": {
        "active_day_rate_recovery": 0.5,
        "regularity_score_target": 0.8,
        "study_consistency_target": 0.8,
        "content_coverage_target": 0.8,
        "active_day_rate_regularity": 0.8,
    },
}


def evaluate_action_eligibility(
    case_features: dict[str, Any],
    action_id: str,
    policy: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Evaluate whether an action_id is feasible for a case based on authority policy."""
    pol = policy or DEFAULT_V2_POLICY
    t = pol.get("thresholds", DEFAULT_V2_POLICY["thresholds"])

    quiz_avail = bool(case_features.get("quiz_available", True))

    if action_id == "QUIZ_RETRIEVAL_PRACTICE":
        if not quiz_avail:
            return False, "CONTRAINDICATED_QUIZ_UNAVAILABLE"
        return True, "ELIGIBLE_QUIZ_AVAILABLE"

    if action_id == "ASSESSMENT_COMPLETION":
        missing = case_features.get("missing_assessment_count", 0)
        due = case_features.get("assessments_due", 0)
        due_soon = case_features.get("due_soon_count", 0)
        if missing > 0 or due > 0 or due_soon > 0:
            return True, "ELIGIBLE_ASSESSMENT_EVIDENCE"
        return False, "INELIGIBLE_NO_ASSESSMENT_EVIDENCE"

    if action_id == "RECOVER_ENGAGEMENT":
        streak = case_features.get("inactivity_streak", 0)
        adr = case_features.get("active_day_rate", 1.0)
        rec_poss = bool(case_features.get("engagement_recovery_possible", False))
        if streak > 0 or adr < t["active_day_rate_recovery"] or rec_poss:
            return True, "ELIGIBLE_ENGAGEMENT_DROP"
        return False, "INELIGIBLE_HIGH_ENGAGEMENT"

    if action_id == "STUDY_REGULARITY":
        reg = case_features.get("regularity_score", 1.0)
        cons = case_features.get("study_consistency", 1.0)
        adr = case_features.get("active_day_rate", 1.0)
        if reg < t["regularity_score_target"] or cons < t["study_consistency_target"] or adr < t["active_day_rate_regularity"]:
            return True, "ELIGIBLE_IRREGULAR_STUDY"
        return False, "INELIGIBLE_HIGH_REGULARITY"

    if action_id == "TARGETED_CONTENT_REVIEW":
        unviewed = case_features.get("unviewed_content", 0)
        low_cov = case_features.get("low_coverage_topics", 0)
        cov = case_features.get("content_coverage", 1.0)
        if unviewed > 0 or low_cov > 0 or cov < t["content_coverage_target"]:
            return True, "ELIGIBLE_CONTENT_GAPS"
        return False, "INELIGIBLE_HIGH_COVERAGE"

    return False, f"UNKNOWN_ACTION_ID_{action_id}"
