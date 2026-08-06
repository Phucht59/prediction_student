"""Deterministic, fail-closed feasibility checks for canonical actions."""

from __future__ import annotations

from src.recommend_hybrid.contracts import Stage

from .contracts import CanonicalAction, FeasibilityResult, RecommendationFeatures


_ACTION_CONTRAINDICATIONS: dict[CanonicalAction, frozenset[str]] = {
    CanonicalAction.ASSESSMENT_COMPLETION: frozenset(
        {"NO_OPEN_ASSESSMENT", "EXTENSION_PENDING"}
    ),
    CanonicalAction.RECOVER_ENGAGEMENT: frozenset({"NO_VLE_ACCESS"}),
    CanonicalAction.STUDY_REGULARITY: frozenset({"ACUTE_PERSONAL_CIRCUMSTANCE"}),
    CanonicalAction.TARGETED_CONTENT_REVIEW: frozenset(
        {"NO_STUDY_MATERIAL", "ASSESSMENT_OVERLOAD"}
    ),
    CanonicalAction.QUIZ_RETRIEVAL_PRACTICE: frozenset(
        {"NO_PRACTICE_MATERIAL", "ASSESSMENT_OVERLOAD"}
    ),
}


def _blocked(action: CanonicalAction, features: RecommendationFeatures) -> tuple[str, ...]:
    active = sorted(_ACTION_CONTRAINDICATIONS[action] & features.contraindications)
    return tuple(f"CONTRAINDICATED_{item}" for item in active)


def evaluate_action(
    action: CanonicalAction,
    features: RecommendationFeatures,
) -> FeasibilityResult:
    blocked = _blocked(action, features)
    if blocked:
        return FeasibilityResult(action, False, blocked)

    if action is CanonicalAction.ASSESSMENT_COMPLETION:
        missing = []
        if features.assessment_progress is None:
            missing.append("MISSING_ASSESSMENT_PROGRESS")
        if features.assessments_due is None:
            missing.append("MISSING_ASSESSMENTS_DUE")
        if features.assessment_window_open is None:
            missing.append("MISSING_ASSESSMENT_WINDOW_STATUS")
        if features.time_to_deadline_days is None:
            missing.append("MISSING_TIME_TO_DEADLINE")
        if missing:
            return FeasibilityResult(action, False, tuple(missing))
        if features.assessments_due <= 0:
            return FeasibilityResult(action, False, ("NO_DUE_ASSESSMENT",))
        if not features.assessment_window_open:
            return FeasibilityResult(action, False, ("ASSESSMENT_WINDOW_CLOSED",))
        if features.time_to_deadline_days <= 0:
            return FeasibilityResult(action, False, ("ASSESSMENT_DEADLINE_PASSED",))

    elif action is CanonicalAction.RECOVER_ENGAGEMENT:
        if features.vle_access_available is not True:
            return FeasibilityResult(action, False, ("VLE_ACCESS_NOT_VERIFIED",))
        required = {
            "inactivity_streak": features.inactivity_streak,
            "active_day_rate": features.active_day_rate,
            "recent_activity_trend": features.recent_activity_trend,
        }
        missing = tuple(f"MISSING_{name.upper()}" for name, value in required.items() if value is None)
        if missing:
            return FeasibilityResult(action, False, missing)

    elif action is CanonicalAction.STUDY_REGULARITY:
        required = {
            "regularity_score": features.regularity_score,
            "active_day_rate": features.active_day_rate,
            "inactivity_streak": features.inactivity_streak,
        }
        missing = tuple(f"MISSING_{name.upper()}" for name, value in required.items() if value is None)
        if missing:
            return FeasibilityResult(action, False, missing)

    elif action is CanonicalAction.TARGETED_CONTENT_REVIEW:
        if features.stage is Stage.EARLY_20:
            return FeasibilityResult(action, False, ("STAGE_TOO_EARLY_FOR_TARGETED_REVIEW",))
        if features.content_coverage is None:
            return FeasibilityResult(action, False, ("MISSING_CONTENT_COVERAGE",))
        if features.knowledge_gap_evidence is not True:
            return FeasibilityResult(action, False, ("KNOWLEDGE_GAP_NOT_VERIFIED",))
        if features.study_material_available is not True:
            return FeasibilityResult(action, False, ("STUDY_MATERIAL_NOT_VERIFIED",))

    elif action is CanonicalAction.QUIZ_RETRIEVAL_PRACTICE:
        if features.quiz_available is not True:
            return FeasibilityResult(action, False, ("QUIZ_NOT_AVAILABLE",))
        if features.quiz_activity is None:
            return FeasibilityResult(action, False, ("MISSING_QUIZ_ACTIVITY",))
        if features.course_progress <= 0.0:
            return FeasibilityResult(action, False, ("NO_STUDIED_CONTENT",))

    return FeasibilityResult(action, True, ("FEASIBLE",))


def feasible_actions(features: RecommendationFeatures) -> tuple[FeasibilityResult, ...]:
    """Evaluate all canonical actions in deterministic catalog order."""

    return tuple(evaluate_action(action, features) for action in CanonicalAction)


__all__ = ["evaluate_action", "feasible_actions"]
