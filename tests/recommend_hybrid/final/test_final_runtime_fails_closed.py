import pytest
import torch

from src.recommend_hybrid.final import (
    MODEL_SCORE_AUTHORITY,
    ActionAwareOutput,
    ConditionalHybridActionRanker,
    IntegratedActionScoreOutput,
)


ACTIONS = [
    {"action_id": "VLE_ENGAGEMENT"},
    {"action_id": "STUDY_SCHEDULE"},
]
VERIFIED_OUTPUT = IntegratedActionScoreOutput((0.1, 0.8, 0.9, 0.2, 0.3))


def test_runtime_requires_external_eligibility_by_default():
    result = ConditionalHybridActionRanker().rank_actions(VERIFIED_OUTPUT, ACTIONS)
    assert result.status == "ELIGIBILITY_REQUIRED"
    assert result.actions == ()


def test_production_runtime_is_not_authorized():
    result = ConditionalHybridActionRanker().rank_actions(
        VERIFIED_OUTPUT,
        ACTIONS,
        policy_authorized=True,
        execution_context="runtime",
    )
    assert result.status == "RUNTIME_NOT_AUTHORIZED"
    assert result.actions == ()


def test_caller_authored_action_scores_are_rejected():
    result = ConditionalHybridActionRanker().rank_actions(
        {"action_scores": [0.1, 0.8, 0.9, 0.2, 0.3]},
        [
            {"action_id": "VLE_ENGAGEMENT", "score": 0.99},
            {"action_id": "STUDY_SCHEDULE", "action_probability": 0.98},
        ],
        policy_authorized=True,
    )
    assert result.status == "INTEGRATED_HEAD_OUTPUT_REQUIRED"
    assert result.actions == ()


def test_model_output_is_required():
    result = ConditionalHybridActionRanker().rank_actions(
        None,
        ACTIONS,
        policy_authorized=True,
    )
    assert result.status == "MODEL_OUTPUT_REQUIRED"
    assert result.actions == ()


def test_score_envelope_is_created_from_neural_head_output():
    head_output = ActionAwareOutput(
        direct_gate_logit=torch.tensor([0.0]),
        action_logits=torch.tensor([[0.1, 0.8, 0.9, 0.2, 0.3]]),
        action_any_probability=torch.tensor([0.5]),
        group_embedding=torch.zeros((1, 4)),
    )
    model_output = IntegratedActionScoreOutput.from_head_output(head_output)
    assert model_output.score_authority == MODEL_SCORE_AUTHORITY
    assert model_output.scores == pytest.approx((0.1, 0.8, 0.9, 0.2, 0.3))


def test_ranker_maps_fixed_head_slots_to_reordered_eligible_actions():
    result = ConditionalHybridActionRanker().rank_actions(
        VERIFIED_OUTPUT,
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
    with pytest.raises(ValueError, match="exactly five"):
        IntegratedActionScoreOutput((0.1, 0.2))


def test_unknown_action_fails_closed():
    result = ConditionalHybridActionRanker().rank_actions(
        VERIFIED_OUTPUT,
        [{"action_id": "UNREGISTERED_ACTION"}],
        policy_authorized=True,
    )
    assert result.status == "UNSUPPORTED_ACTION_SET"
    assert result.actions == ()
