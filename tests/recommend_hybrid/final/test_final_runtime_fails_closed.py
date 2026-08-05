import pytest

from src.recommend_hybrid.final import (
    MODEL_SCORE_AUTHORITY,
    ConditionalHybridActionRanker,
)


ACTIONS = [
    {"action_id": "VLE_ENGAGEMENT"},
    {"action_id": "STUDY_SCHEDULE"},
]
VERIFIED_STATE = {
    "score_authority": MODEL_SCORE_AUTHORITY,
    # Canonical slots: assessment, regularity, VLE, retrieval, content.
    "action_logits": [0.1, 0.8, 0.9, 0.2, 0.3],
}


def test_runtime_requires_external_eligibility_by_default():
    result = ConditionalHybridActionRanker().rank_actions(VERIFIED_STATE, ACTIONS)
    assert result.status == "ELIGIBILITY_REQUIRED"
    assert result.actions == ()


def test_production_runtime_is_not_authorized():
    result = ConditionalHybridActionRanker().rank_actions(
        VERIFIED_STATE,
        ACTIONS,
        policy_authorized=True,
        execution_context="runtime",
    )
    assert result.status == "RUNTIME_NOT_AUTHORIZED"
    assert result.actions == ()


def test_caller_authored_action_scores_are_rejected():
    result = ConditionalHybridActionRanker().rank_actions(
        {},
        [
            {"action_id": "VLE_ENGAGEMENT", "score": 0.99},
            {"action_id": "STUDY_SCHEDULE", "action_probability": 0.98},
        ],
        policy_authorized=True,
    )
    assert result.status == "MODEL_OUTPUT_AUTHORITY_REQUIRED"
    assert result.actions == ()


def test_verified_authority_still_requires_model_output():
    result = ConditionalHybridActionRanker().rank_actions(
        {"score_authority": MODEL_SCORE_AUTHORITY},
        ACTIONS,
        policy_authorized=True,
    )
    assert result.status == "MODEL_OUTPUT_REQUIRED"
    assert result.actions == ()


def test_ranker_maps_fixed_head_slots_to_reordered_eligible_actions():
    result = ConditionalHybridActionRanker().rank_actions(
        VERIFIED_STATE,
        ACTIONS,
        policy_authorized=True,
    )
    assert result.status == "RANKED_ELIGIBLE_ACTIONS_OFFLINE"
    assert result.score_authority == MODEL_SCORE_AUTHORITY
    assert [item["action_id"] for item in result.actions] == [
        "VLE_ENGAGEMENT",
        "STUDY_SCHEDULE",
    ]
    assert [item["score"] for item in result.actions] == [0.9, 0.8]
    assert result.actions[1]["canonical_action_id"] == "STUDY_REGULARITY"


def test_integrated_head_output_must_have_five_slots():
    state = {
        "score_authority": MODEL_SCORE_AUTHORITY,
        "action_logits": [0.1, 0.2],
    }
    with pytest.raises(ValueError, match="exactly five"):
        ConditionalHybridActionRanker().rank_actions(
            state,
            ACTIONS,
            policy_authorized=True,
        )


def test_unknown_action_fails_closed():
    result = ConditionalHybridActionRanker().rank_actions(
        VERIFIED_STATE,
        [{"action_id": "UNREGISTERED_ACTION"}],
        policy_authorized=True,
    )
    assert result.status == "UNSUPPORTED_ACTION_SET"
    assert result.actions == ()
