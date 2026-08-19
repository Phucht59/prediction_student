"""Deterministic hard feasibility. Same five-action policy as V2 v4."""

from __future__ import annotations

from .contracts import CanonicalAction, FeasibilityResult, RecommendationFeatures, Stage


DEFAULT_V3_POLICY = {
    "version": "v3_c0_feasibility_v1",
    "thresholds": {
        "active_day_rate_recovery": 0.5,
        "regularity_score_target": 0.8,
        "content_coverage_target": 0.8,
        "active_day_rate_regularity": 0.8,
    },
}

_ACTION_CONTRAINDICATIONS: dict[CanonicalAction, frozenset[str]] = {
    CanonicalAction.ASSESSMENT_COMPLETION: frozenset({"NO_OPEN_ASSESSMENT", "EXTENSION_PENDING"}),
    CanonicalAction.RECOVER_ENGAGEMENT: frozenset({"NO_VLE_ACCESS"}),
    CanonicalAction.STUDY_REGULARITY: frozenset({"ACUTE_PERSONAL_CIRCUMSTANCE"}),
    CanonicalAction.TARGETED_CONTENT_REVIEW: frozenset({"NO_STUDY_MATERIAL", "ASSESSMENT_OVERLOAD"}),
    CanonicalAction.QUIZ_RETRIEVAL_PRACTICE: frozenset({"NO_PRACTICE_MATERIAL", "ASSESSMENT_OVERLOAD"}),
}


def evaluate_action(
    action: CanonicalAction,
    features: RecommendationFeatures,
    policy: dict | None = None,
) -> FeasibilityResult:
    blocked = sorted(_ACTION_CONTRAINDICATIONS[action] & features.contraindications)
    if blocked:
        return FeasibilityResult(action, False, tuple(f"CONTRAINDICATED_{item}" for item in blocked))
    thresholds = (policy or DEFAULT_V3_POLICY)["thresholds"]
    if action is CanonicalAction.QUIZ_RETRIEVAL_PRACTICE:
        if not features.quiz_available:
            return FeasibilityResult(action, False, ("CONTRAINDICATED_QUIZ_UNAVAILABLE",))
        return FeasibilityResult(action, True, ("ELIGIBLE_QUIZ_AVAILABLE",))
    if action is CanonicalAction.ASSESSMENT_COMPLETION:
        missing = int(features.missing_assessment_count or 0)
        due_soon = int(features.due_soon_count or 0)
        if missing > 0 or due_soon > 0:
            return FeasibilityResult(action, True, ("ELIGIBLE_ASSESSMENT_GAP_OR_UPCOMING_DUE",))
        return FeasibilityResult(action, False, ("INELIGIBLE_NO_ASSESSMENT_GAP",))
    if action is CanonicalAction.RECOVER_ENGAGEMENT:
        if not features.vle_access_available:
            return FeasibilityResult(action, False, ("CONTRAINDICATED_VLE_UNAVAILABLE",))
        if features.active_day_rate is None:
            return FeasibilityResult(action, False, ("INELIGIBLE_MISSING_ACTIVE_DAY_RATE",))
        if float(features.active_day_rate) < thresholds["active_day_rate_recovery"]:
            return FeasibilityResult(action, True, ("ELIGIBLE_LOW_ACTIVE_DAY_RATE",))
        return FeasibilityResult(action, False, ("INELIGIBLE_ENGAGEMENT_NOT_LOW",))
    if action is CanonicalAction.STUDY_REGULARITY:
        if features.regularity_score is None or features.active_day_rate is None:
            return FeasibilityResult(action, False, ("INELIGIBLE_MISSING_REGULARITY_EVIDENCE",))
        if (
            float(features.regularity_score) < thresholds["regularity_score_target"]
            or float(features.active_day_rate) < thresholds["active_day_rate_regularity"]
        ):
            return FeasibilityResult(action, True, ("ELIGIBLE_IRREGULAR_STUDY",))
        return FeasibilityResult(action, False, ("INELIGIBLE_HIGH_REGULARITY",))
    if action is CanonicalAction.TARGETED_CONTENT_REVIEW:
        if features.stage is Stage.EARLY_20:
            return FeasibilityResult(action, False, ("INELIGIBLE_STAGE_TOO_EARLY",))
        if not features.study_material_available:
            return FeasibilityResult(action, False, ("CONTRAINDICATED_STUDY_MATERIAL_UNAVAILABLE",))
        if features.content_coverage is None:
            return FeasibilityResult(action, False, ("INELIGIBLE_MISSING_CONTENT_COVERAGE",))
        if float(features.content_coverage) < thresholds["content_coverage_target"]:
            return FeasibilityResult(action, True, ("ELIGIBLE_LOW_CONTENT_COVERAGE",))
        return FeasibilityResult(action, False, ("INELIGIBLE_HIGH_CONTENT_COVERAGE",))
    return FeasibilityResult(action, False, (f"UNKNOWN_ACTION_{action.value}",))


def feasible_actions(features: RecommendationFeatures) -> tuple[FeasibilityResult, ...]:
    return tuple(evaluate_action(action, features) for action in CanonicalAction)
