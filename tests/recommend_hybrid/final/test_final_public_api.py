from src.recommend_hybrid import (
    MODEL_SCORE_AUTHORITY,
    SCIENTIFIC_ACTION_ORDER,
    ConditionalHybridActionRanker,
    IntegratedActionScoreOutput,
)


def test_root_package_exposes_validated_conditional_ranker():
    assert ConditionalHybridActionRanker.module_boundary == (
        "conditional_hybrid_action_ranker"
    )
    assert ConditionalHybridActionRanker.runtime_authorized is False
    assert MODEL_SCORE_AUTHORITY == "integrated_conditional_action_head"
    assert IntegratedActionScoreOutput.score_authority == MODEL_SCORE_AUTHORITY
    assert SCIENTIFIC_ACTION_ORDER == (
        "ASSESSMENT_COMPLETION",
        "STUDY_REGULARITY",
        "VLE_ENGAGEMENT",
        "QUIZ_OR_RETRIEVAL_PRACTICE",
        "CONTENT_REVIEW",
    )
