from src.recommend_hybrid.explainable_v2.action_eligibility import (
    evaluate_action_eligibility,
)
from src.recommend_hybrid.contracts import Stage
from src.recommend_hybrid.explainable_v2.contracts import (
    CanonicalAction,
    RecommendationFeatures,
)
from src.recommend_hybrid.explainable_v2.feasibility import evaluate_action


def _base():
    return {
        "stage": "EARLY_35",
        "quiz_available": True,
        "vle_available": True,
        "study_material_available": True,
        "missing_assessment_count": 1,
        "due_soon_count": 0,
        "active_day_rate": 0.3,
        "regularity_score": 0.5,
        "content_coverage": 0.4,
    }


def test_all_five_actions_can_be_eligible_from_query_level_evidence():
    case = _base()
    actions = [
        "ASSESSMENT_COMPLETION",
        "RECOVER_ENGAGEMENT",
        "STUDY_REGULARITY",
        "TARGETED_CONTENT_REVIEW",
        "QUIZ_RETRIEVAL_PRACTICE",
    ]
    assert all(
        evaluate_action_eligibility(
            case,
            action,
        )[0]
        for action in actions
    )


def test_targeted_content_review_is_not_sent_at_early20():
    case = _base()
    case["stage"] = "EARLY_20"
    eligible, code = evaluate_action_eligibility(
        case,
        "TARGETED_CONTENT_REVIEW",
    )
    assert eligible is False
    assert code == "INELIGIBLE_STAGE_TOO_EARLY"


def test_quiz_unavailable_is_fail_closed_contraindication():
    case = _base()
    case["quiz_available"] = False
    eligible, code = evaluate_action_eligibility(
        case,
        "QUIZ_RETRIEVAL_PRACTICE",
    )
    assert eligible is False
    assert (
        code
        == "CONTRAINDICATED_QUIZ_UNAVAILABLE"
    )


def test_runtime_feasibility_uses_query_level_policy_for_every_action():
    case = _base()
    features = RecommendationFeatures(
        student_key="s",
        course_key="AAA:2014J",
        stage=Stage(case["stage"]),
        cutoff_day=40,
        risk_probability=0.8,
        hybrid_uncertainty=0.1,
        seed_disagreement=None,
        course_progress=0.35,
        missing_assessment_count=case["missing_assessment_count"],
        due_soon_count=case["due_soon_count"],
        active_day_rate=case["active_day_rate"],
        regularity_score=case["regularity_score"],
        content_coverage=case["content_coverage"],
        quiz_available=case["quiz_available"],
        vle_access_available=case["vle_available"],
        study_material_available=case["study_material_available"],
    )

    for action in CanonicalAction:
        expected = evaluate_action_eligibility(case, action.value)
        actual = evaluate_action(action, features)
        assert (actual.eligible, actual.reason_codes[0]) == expected


def test_runtime_only_fields_do_not_change_canonical_feasibility():
    base = RecommendationFeatures(
        student_key="s",
        course_key="AAA:2014J",
        stage=Stage.EARLY_35,
        cutoff_day=40,
        risk_probability=0.8,
        hybrid_uncertainty=0.1,
        seed_disagreement=None,
        course_progress=0.35,
        missing_assessment_count=1,
        due_soon_count=0,
        active_day_rate=0.3,
        regularity_score=0.5,
        content_coverage=0.4,
        quiz_available=True,
        vle_access_available=True,
        study_material_available=True,
    )
    altered = RecommendationFeatures(
        **{
            **base.__dict__,
            "assessment_window_open": False,
            "knowledge_gap_evidence": False,
            "recent_activity_trend": 1.0,
            "time_to_deadline_days": -10,
        }
    )
    assert feasible_signature(base) == feasible_signature(altered)


def feasible_signature(features):
    return tuple(
        (action, evaluate_action(action, features).eligible)
        for action in CanonicalAction
    )
