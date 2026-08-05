from src.recommend_hybrid import (
    MODEL_SCORE_AUTHORITY,
    SCIENTIFIC_ACTION_ORDER,
    ConditionalHybridActionRanker,
)


def test_root_package_exposes_validated_conditional_ranker():
    assert ConditionalHybridActionRanker.module_boundary == (
        "conditional_hybrid_action_ranker"
    )
    assert ConditionalHybridActionRanker.runtime_authorized is False
    assert MODEL_SCORE_AUTHORITY == "integrated_conditional_action_head"
    assert SCIENTIFIC_ACTION_ORDER == (
        "ASSESSMENT_COMPLETION",
        "STUDY_REGULARITY",
        "VLE_ENGAGEMENT",
        "QUIZ_OR_RETRIEVAL_PRACTICE",
        "CONTENT_REVIEW",
    )
