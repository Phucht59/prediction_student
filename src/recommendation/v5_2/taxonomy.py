from __future__ import annotations


POLICY_VERSION = "recommendation-v5.2.0"
WEEKLY_MINUTES_CAP = 360
ACTIONS_CAP = 5


def _action(
    action_id: str,
    datasets: list[str],
    trigger: str,
    contraindication: str,
    priority: int,
    minutes: int,
    target_week: int,
    reason_code: str,
    evidence_source: str,
    requires_advisor: bool,
    safety_notes: str,
) -> dict[str, object]:
    return {
        "id": action_id,
        "dataset_applicability": datasets,
        "trigger": trigger,
        "contraindication": contraindication,
        "default_priority": priority,
        "weekly_minutes": minutes,
        "target_week": target_week,
        "reason_code": reason_code,
        "evidence_source": evidence_source,
        "requires_advisor": requires_advisor,
        "safety_notes": safety_notes,
    }


ACTION_TAXONOMY = {
    "FOUNDATION_REVIEW": _action(
        "FOUNDATION_REVIEW", ["student-mat", "student-por"], "low achievement probability", "high achievement with no foundation gap", 3, 60, 1, "LOW_ACHIEVEMENT_RISK", "deep+ML prediction and G1/G2", False, "Use only already-taught material.",
    ),
    "TARGETED_PRACTICE": _action(
        "TARGETED_PRACTICE", ["student-mat", "student-por"], "non-positive grade trend", "no confirmed topic or excessive workload", 2, 75, 1, "NON_POSITIVE_GRADE_TREND", "G1/G2 before outcome", False, "Advisor confirms topics; no final-outcome data.",
    ),
    "STUDY_SCHEDULE": _action(
        "STUDY_SCHEDULE", ["student-mat", "student-por", "oulad"], "medium/high risk", "existing plan already exceeds cap", 3, 30, 1, "PLAN_STRUCTURE_NEEDED", "prediction and pre-cutoff behavior", False, "Keep total weekly workload within cap.",
    ),
    "ATTENDANCE_SUPPORT": _action(
        "ATTENDANCE_SUPPORT", ["student-mat", "student-por"], "attendance-support indicator", "attendance data absent or sensitivity-only", 4, 30, 1, "ATTENDANCE_SUPPORT_SIGNAL", "approved pre-prediction context", True, "Never infer a reason for absence.",
    ),
    "ASSESSMENT_COMPLETION": _action(
        "ASSESSMENT_COMPLETION", ["oulad"], "low assessment progress", "no assessment currently available", 1, 60, 1, "ASSESSMENT_PROGRESS_DEFICIT", "pre-cutoff assessment progress", False, "Do not expose or infer future assessments.",
    ),
    "VLE_ENGAGEMENT": _action(
        "VLE_ENGAGEMENT", ["oulad"], "low recent activity or inactivity streak", "platform unavailable", 1, 45, 1, "LOW_VLE_ENGAGEMENT", "pre-cutoff VLE sequence", False, "Recommend a short session, not indiscriminate clicking.",
    ),
    "PEER_STUDY": _action(
        "PEER_STUDY", ["student-mat", "student-por", "oulad"], "medium risk and sufficient confidence", "student preference or accessibility conflict", 5, 45, 2, "COLLABORATIVE_SUPPORT_OPTION", "risk and confidence", False, "Optional and subject to consent.",
    ),
    "INSTRUCTOR_CONTACT": _action(
        "INSTRUCTOR_CONTACT", ["student-mat", "student-por", "oulad"], "persistent risk signal", "no unresolved academic question", 2, 30, 1, "PERSISTENT_RISK_SIGNAL", "deep+ML prediction", True, "Contact should identify a specific question.",
    ),
    "ADVISOR_ESCALATION": _action(
        "ADVISOR_ESCALATION", ["student-mat", "student-por", "oulad"], "large disagreement, high uncertainty, or high risk", "none", 1, 30, 1, "ADVISOR_REVIEW_REQUIRED", "prediction disagreement and confidence", True, "Human review is mandatory; no automated adverse decision.",
    ),
    "PROGRESS_MONITORING": _action(
        "PROGRESS_MONITORING", ["student-mat", "student-por", "oulad"], "any non-abstained plan", "none", 4, 20, 4, "FOLLOW_UP_REQUIRED", "policy contract", False, "Monitor completion only; do not claim causal impact.",
    ),
}


if tuple(ACTION_TAXONOMY) != (
    "FOUNDATION_REVIEW",
    "TARGETED_PRACTICE",
    "STUDY_SCHEDULE",
    "ATTENDANCE_SUPPORT",
    "ASSESSMENT_COMPLETION",
    "VLE_ENGAGEMENT",
    "PEER_STUDY",
    "INSTRUCTOR_CONTACT",
    "ADVISOR_ESCALATION",
    "PROGRESS_MONITORING",
):
    raise RuntimeError("Recommendation V5.2 taxonomy order changed")


__all__ = ["ACTIONS_CAP", "ACTION_TAXONOMY", "POLICY_VERSION", "WEEKLY_MINUTES_CAP"]
