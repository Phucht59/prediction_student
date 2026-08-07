from src.recommend_hybrid.explainable_v2.action_eligibility import (
    evaluate_action_eligibility,
)


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
