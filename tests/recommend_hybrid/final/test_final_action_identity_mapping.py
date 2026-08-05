import math

import pytest

from src.recommend_hybrid.final import (
    ACTION_ORDER,
    ConditionalHybridActionRanker,
    canonical_action_id,
)


def test_canonical_action_order_matches_trained_head_slots():
    assert ACTION_ORDER == (
        "ASSESSMENT_COMPLETION",
        "STUDY_REGULARITY",
        "VLE_ENGAGEMENT",
        "QUIZ_OR_RETRIEVAL_PRACTICE",
        "CONTENT_REVIEW",
    )


def test_policy_aliases_map_to_scientific_action_identities():
    assert canonical_action_id("STUDY_SCHEDULE") == "STUDY_REGULARITY"
    assert (
        canonical_action_id("RETRIEVAL_PRACTICE")
        == "QUIZ_OR_RETRIEVAL_PRACTICE"
    )
    assert canonical_action_id("LEARNING_CONSOLIDATION") == "CONTENT_REVIEW"


def test_full_model_vector_is_mapped_by_identity_not_eligible_list_position():
    # Canonical slots: assessment, study, VLE, retrieval, content.
    state = {
        "model_id": "conditional_hybrid_action_ranker",
        "action_scores": [0.05, 0.95, 0.40, 0.30, 0.20],
    }
    result = ConditionalHybridActionRanker().rank_actions(
        state,
        [
            {"action_id": "VLE_ENGAGEMENT"},
            {"action_id": "STUDY_SCHEDULE"},
        ],
        policy_authorized=True,
    )
    assert result.status == "RANKED_ELIGIBLE_ACTIONS"
    assert [item["action_id"] for item in result.actions] == [
        "STUDY_SCHEDULE",
        "VLE_ENGAGEMENT",
    ]
    assert result.actions[0]["model_action_id"] == "STUDY_REGULARITY"
    assert result.actions[0]["model_action_index"] == 1
    assert result.actions[0]["score"] == pytest.approx(0.95)


def test_mapping_output_can_use_policy_aliases():
    result = ConditionalHybridActionRanker().rank_actions(
        {
            "action_scores": {
                "STUDY_REGULARITY": 0.20,
                "VLE_ENGAGEMENT": 0.80,
            }
        },
        [
            {"action_id": "STUDY_SCHEDULE"},
            {"action_id": "VLE_ENGAGEMENT"},
        ],
        policy_authorized=True,
    )
    assert result.actions[0]["action_id"] == "VLE_ENGAGEMENT"
    assert result.actions[1]["action_id"] == "STUDY_SCHEDULE"


def test_logits_are_converted_to_probabilities():
    result = ConditionalHybridActionRanker().rank_actions(
        {"action_logits": [0.0, 2.0, -2.0, 1.0, -1.0]},
        [{"action_id": "STUDY_SCHEDULE"}],
        policy_authorized=True,
    )
    assert result.actions[0]["score"] == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))


def test_eligible_subset_length_is_not_accepted_as_model_vector_length():
    with pytest.raises(ValueError, match="exactly 5 canonical slots"):
        ConditionalHybridActionRanker().rank_actions(
            {"action_scores": [0.2, 0.8]},
            [
                {"action_id": "STUDY_SCHEDULE"},
                {"action_id": "VLE_ENGAGEMENT"},
            ],
            policy_authorized=True,
        )


def test_unsupported_policy_action_is_rejected():
    with pytest.raises(ValueError, match="outside the validated"):
        ConditionalHybridActionRanker().rank_actions(
            {"action_scores": [0.1, 0.2, 0.3, 0.4, 0.5]},
            [{"action_id": "INSTRUCTOR_CONTACT"}],
            policy_authorized=True,
        )


def test_duplicate_aliases_for_same_trained_action_are_rejected():
    with pytest.raises(ValueError, match="duplicate eligible action identity"):
        ConditionalHybridActionRanker().rank_actions(
            {"action_scores": [0.1, 0.2, 0.3, 0.4, 0.5]},
            [
                {"action_id": "STUDY_REGULARITY"},
                {"action_id": "STUDY_SCHEDULE"},
            ],
            policy_authorized=True,
        )
