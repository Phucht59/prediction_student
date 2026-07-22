import pytest

from src.studies.v5.common.recommendation import build_recommendation, revise_recommendation


def test_v5_recommendation_has_four_weeks_limits_and_human_review():
    value = build_recommendation(case_reference="case-1", dataset="student-mat", model_version="m1", prediction_set="p1", feature_snapshot="s1", probabilities=[0.7, 0.2, 0.1], features={"activity_level": 0.2, "grade_trend": -1.0}, created_at="2026-07-18T00:00:00+00:00")
    assert len(value["weeks"]) == 4
    assert all(week["workload_minutes"] <= 180 for week in value["weeks"])
    assert value["advisor_review"]["required"] is True
    assert value["causal_claim"] == "PROHIBITED"
    action_codes = [action["action_code"] for week in value["weeks"] for action in week["actions"]]
    assert len(action_codes) == len(set(action_codes))


def test_v5_recommendation_abstains_when_required_feature_is_missing():
    value = build_recommendation(case_reference="case-2", dataset="oulad", model_version="m1", prediction_set="p1", feature_snapshot="s1", probabilities=[0.3, 0.7], features={"activity_level": None}, created_at="2026-07-18T00:00:00+00:00")
    assert value["abstained"] is True
    assert all(not week["actions"] for week in value["weeks"])
    assert value["escalation_required"] is True


def test_v5_revision_rejects_mutated_history():
    value = build_recommendation(case_reference="case-3", dataset="oulad", model_version="m1", prediction_set="p1", feature_snapshot="s1", probabilities=[0.4, 0.6], features={"activity_level": 0.2}, created_at="2026-07-18T00:00:00+00:00")
    revised = revise_recommendation(value, value["weeks"], "advisor clarification")
    assert revised["revision_no"] == 2
    value["goal"] = "mutated"
    with pytest.raises(ValueError):
        revise_recommendation(value, value["weeks"], "invalid")
