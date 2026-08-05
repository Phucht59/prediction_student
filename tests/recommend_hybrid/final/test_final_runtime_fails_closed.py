from src.recommend_hybrid.final import ConditionalHybridActionRanker


def test_runtime_requires_explicit_external_eligibility():
    result = ConditionalHybridActionRanker().rank_actions(
        {"action_scores": [0.1, 0.2, 0.3, 0.4, 0.5]},
        [{"action_id": "VLE_ENGAGEMENT"}],
    )
    assert result.status == "ELIGIBILITY_REQUIRED"
    assert result.actions == ()


def test_runtime_requires_integrated_head_output():
    result = ConditionalHybridActionRanker().rank_actions(
        {},
        [{"action_id": "VLE_ENGAGEMENT", "score": 1.0}],
        policy_authorized=True,
    )
    assert result.status == "MODEL_SCORES_REQUIRED"
    assert result.actions == ()


def test_runtime_rejects_wrong_model_authority():
    result = ConditionalHybridActionRanker().rank_actions(
        {
            "model_id": "external_ranker",
            "action_scores": [0.1, 0.2, 0.3, 0.4, 0.5],
        },
        [{"action_id": "VLE_ENGAGEMENT"}],
        policy_authorized=True,
    )
    assert result.status == "MODEL_AUTHORITY_MISMATCH"
    assert result.actions == ()


def test_runtime_does_not_use_scores_embedded_in_action_payloads():
    result = ConditionalHybridActionRanker().rank_actions(
        {},
        [
            {"action_id": "STUDY_SCHEDULE", "score": 0.0},
            {"action_id": "VLE_ENGAGEMENT", "score": 1.0},
        ],
        policy_authorized=True,
    )
    assert result.status == "MODEL_SCORES_REQUIRED"
    assert result.actions == ()
