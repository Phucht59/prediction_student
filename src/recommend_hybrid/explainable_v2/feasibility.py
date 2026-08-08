"""Deterministic, fail-closed feasibility checks for canonical actions."""

from __future__ import annotations

from .contracts import CanonicalAction, FeasibilityResult, RecommendationFeatures
from .action_eligibility import evaluate_action_eligibility


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

    case_features = {
        "stage": features.stage.value,
        "quiz_available": features.quiz_available,
        "vle_available": features.vle_access_available,
        "study_material_available": features.study_material_available,
        "missing_assessment_count": features.missing_assessment_count,
        "due_soon_count": features.due_soon_count,
        "active_day_rate": features.active_day_rate,
        "regularity_score": features.regularity_score,
        "content_coverage": features.content_coverage,
    }
    eligible, reason = evaluate_action_eligibility(
        case_features,
        action.value,
    )
    return FeasibilityResult(action, bool(eligible), (reason,))


def feasible_actions(features: RecommendationFeatures) -> tuple[FeasibilityResult, ...]:
    """Evaluate all canonical actions in deterministic catalog order."""

    return tuple(evaluate_action(action, features) for action in CanonicalAction)


__all__ = ["evaluate_action", "feasible_actions"]
