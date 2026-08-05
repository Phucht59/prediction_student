from src.recommend_hybrid.final import ConditionalHybridActionRanker


def test_runtime_requires_external_eligibility():
    result = ConditionalHybridActionRanker().rank_actions({}, [{"action_id": "x", "score": 1.0}], policy_authorized=False)
    assert result.status == "ELIGIBILITY_REQUIRED"
    assert result.actions == ()


def test_ranker_only_orders_supplied_actions():
    result = ConditionalHybridActionRanker().rank_actions({}, [{"action_id": "x", "score": 0.1}, {"action_id": "y", "score": 0.9}])
    assert result.status == "RANKED_ELIGIBLE_ACTIONS"
    assert result.actions[0]["action_id"] == "y"
