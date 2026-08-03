from __future__ import annotations

from pathlib import Path

from src.recommend_hybrid.counterfactual import (
    CounterfactualEffectCatalog,
    CounterfactualStateSimulator,
    CounterfactualUtilityConfig,
    CounterfactualUtilityRanker,
    RiskEstimate,
    SimulationStatus,
)

ROOT = Path(__file__).resolve().parents[3]


class LinearRiskPredictor:
    def __init__(self, uncertainty: float = 0.10) -> None:
        self.uncertainty = uncertainty

    def predict_risk(self, state):
        activity = float(state.get("activity_level") or 0.0)
        inactivity = float(state.get("inactivity_streak") or 0.0)
        progress = float(state.get("assessment_progress") or 0.0)
        recent = float(state.get("recent_activity") or 0.0)
        risk = (
            0.80
            - 0.004 * activity
            + 0.02 * inactivity
            - 0.45 * progress
            - 0.002 * recent
        )
        risk = min(1.0, max(0.0, risk))
        return RiskEstimate(
            risk_probability=risk,
            uncertainty=self.uncertainty,
            source="TEST",
        )


def _catalog():
    return CounterfactualEffectCatalog.load(
        ROOT / "configs/recommend_hybrid/counterfactual_oulad.yaml"
    )


def _state():
    return {
        "activity_level": 20.0,
        "inactivity_streak": 8,
        "assessment_progress": 0.40,
        "recent_activity_trend": -0.30,
        "recent_activity": 10.0,
        "assessments_completed": 1,
        "assessments_due": 2,
        "course_progress": 0.50,
        "grade_trend": -0.10,
    }


def _references():
    return {
        "activity_level_p50": 45.0,
        "activity_level_p65": 60.0,
        "recent_activity_p50": 30.0,
        "recent_activity_p65": 45.0,
    }


def test_catalog_never_changes_protected_features():
    catalog = _catalog()
    assert catalog.mutable_features.isdisjoint(catalog.protected_features)
    assert all(
        effect.feature_name not in catalog.protected_features
        for action in catalog.actions
        for effect in action.effects
    )


def test_simulator_is_deterministic_and_does_not_mutate_input():
    simulator = CounterfactualStateSimulator(_catalog())
    state = _state()
    original = dict(state)
    first = simulator.simulate("VLE_ENGAGEMENT", state, _references())
    second = simulator.simulate("VLE_ENGAGEMENT", state, _references())

    assert state == original
    assert first == second
    assert first.status is SimulationStatus.SIMULATED
    simulated = first.simulated_mapping()
    assert simulated["activity_level"] == 60.0
    assert simulated["inactivity_streak"] == 4
    assert simulated["course_progress"] == state["course_progress"]
    assert simulated["grade_trend"] == state["grade_trend"]


def test_non_scorable_action_routes_to_policy_fallback():
    simulator = CounterfactualStateSimulator(_catalog())
    scenario = simulator.simulate(
        "ADVISOR_ESCALATION",
        _state(),
        _references(),
    )
    assert scenario.status is SimulationStatus.NOT_SCORABLE
    assert scenario.reason_codes == (
        "HUMAN_SUPPORT_ACTION_REQUIRES_POLICY_FALLBACK",
    )


def test_ranker_orders_by_model_estimated_utility():
    simulator = CounterfactualStateSimulator(_catalog())
    ranker = CounterfactualUtilityRanker(simulator)
    result = ranker.rank(
        candidate_action_ids=(
            "STUDY_SCHEDULE",
            "VLE_ENGAGEMENT",
            "ASSESSMENT_COMPLETION",
            "ADVISOR_ESCALATION",
        ),
        baseline_state=_state(),
        reference_values=_references(),
        predictor=LinearRiskPredictor(),
        workload_minutes={
            "STUDY_SCHEDULE": 30,
            "VLE_ENGAGEMENT": 90,
            "ASSESSMENT_COMPLETION": 150,
            "ADVISOR_ESCALATION": 30,
        },
    )

    ranked_ids = [item.action_id for item in result.ranked_actions]
    assert ranked_ids[0] == "VLE_ENGAGEMENT"
    assert "ASSESSMENT_COMPLETION" in ranked_ids
    assert result.rejected_actions[0].action_id == "ADVISOR_ESCALATION"
    assert all(item.risk_reduction > 0 for item in result.ranked_actions)


def test_uncertainty_reduces_utility_without_changing_risk_delta():
    simulator = CounterfactualStateSimulator(_catalog())
    ranker = CounterfactualUtilityRanker(
        simulator,
        CounterfactualUtilityConfig(minimum_risk_reduction=0.0),
    )
    common = dict(
        candidate_action_ids=("VLE_ENGAGEMENT",),
        baseline_state=_state(),
        reference_values=_references(),
        workload_minutes={"VLE_ENGAGEMENT": 90},
    )
    low = ranker.rank(
        predictor=LinearRiskPredictor(0.05),
        **common,
    ).ranked_actions[0]
    high = ranker.rank(
        predictor=LinearRiskPredictor(0.80),
        **common,
    ).ranked_actions[0]
    assert low.risk_reduction == high.risk_reduction
    assert low.utility_score > high.utility_score


def test_ranking_replay_is_byte_stable_at_contract_level():
    simulator = CounterfactualStateSimulator(_catalog())
    ranker = CounterfactualUtilityRanker(simulator)
    kwargs = dict(
        candidate_action_ids=(
            "TARGETED_PRACTICE",
            "VLE_ENGAGEMENT",
            "STUDY_SCHEDULE",
        ),
        baseline_state=_state(),
        reference_values=_references(),
        predictor=LinearRiskPredictor(),
        workload_minutes={
            "TARGETED_PRACTICE": 120,
            "VLE_ENGAGEMENT": 90,
            "STUDY_SCHEDULE": 30,
        },
    )
    assert ranker.rank(**kwargs).to_dict() == ranker.rank(**kwargs).to_dict()
