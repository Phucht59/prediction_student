"""Query-level annotation candidate eligibility policy."""

from __future__ import annotations

from typing import Any

DEFAULT_V2_POLICY = {
    "version": "v4_query_level_annotation_policy",
    "thresholds": {
        "active_day_rate_recovery": 0.5,
        "regularity_score_target": 0.8,
        "content_coverage_target": 0.8,
        "active_day_rate_regularity": 0.8,
    },
}


def evaluate_action_eligibility(
    case_features: dict[str, Any],
    action_id: str,
    policy: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    pol = policy or DEFAULT_V2_POLICY
    thresholds = pol.get(
        "thresholds",
        DEFAULT_V2_POLICY["thresholds"],
    )
    stage = str(case_features.get("stage", ""))
    quiz_available = bool(
        case_features.get("quiz_available", False)
    )
    vle_available = bool(
        case_features.get("vle_available", False)
    )
    study_material_available = bool(
        case_features.get("study_material_available", False)
    )

    if action_id == "QUIZ_RETRIEVAL_PRACTICE":
        if not quiz_available:
            return False, "CONTRAINDICATED_QUIZ_UNAVAILABLE"
        return True, "ELIGIBLE_QUIZ_AVAILABLE"

    if action_id == "ASSESSMENT_COMPLETION":
        missing = int(
            case_features.get("missing_assessment_count", 0) or 0
        )
        due_soon = int(
            case_features.get("due_soon_count", 0) or 0
        )
        if missing > 0 or due_soon > 0:
            return (
                True,
                "ELIGIBLE_ASSESSMENT_GAP_OR_UPCOMING_DUE",
            )
        return False, "INELIGIBLE_NO_ASSESSMENT_GAP"

    if action_id == "RECOVER_ENGAGEMENT":
        if not vle_available:
            return False, "CONTRAINDICATED_VLE_UNAVAILABLE"
        active_day_rate = case_features.get("active_day_rate")
        if active_day_rate is None:
            return False, "INELIGIBLE_MISSING_ACTIVE_DAY_RATE"
        if (
            float(active_day_rate)
            < thresholds["active_day_rate_recovery"]
        ):
            return True, "ELIGIBLE_LOW_ACTIVE_DAY_RATE"
        return False, "INELIGIBLE_ENGAGEMENT_NOT_LOW"

    if action_id == "STUDY_REGULARITY":
        regularity = case_features.get("regularity_score")
        active_day_rate = case_features.get("active_day_rate")
        if regularity is None or active_day_rate is None:
            return (
                False,
                "INELIGIBLE_MISSING_REGULARITY_EVIDENCE",
            )
        if (
            float(regularity)
            < thresholds["regularity_score_target"]
            or float(active_day_rate)
            < thresholds["active_day_rate_regularity"]
        ):
            return True, "ELIGIBLE_IRREGULAR_STUDY"
        return False, "INELIGIBLE_HIGH_REGULARITY"

    if action_id == "TARGETED_CONTENT_REVIEW":
        if stage == "EARLY_20":
            return False, "INELIGIBLE_STAGE_TOO_EARLY"
        if not study_material_available:
            return (
                False,
                "CONTRAINDICATED_STUDY_MATERIAL_UNAVAILABLE",
            )
        coverage = case_features.get("content_coverage")
        if coverage is None:
            return (
                False,
                "INELIGIBLE_MISSING_CONTENT_COVERAGE",
            )
        if (
            float(coverage)
            < thresholds["content_coverage_target"]
        ):
            return True, "ELIGIBLE_LOW_CONTENT_COVERAGE"
        return False, "INELIGIBLE_HIGH_CONTENT_COVERAGE"

    return False, f"UNKNOWN_ACTION_ID_{action_id}"
