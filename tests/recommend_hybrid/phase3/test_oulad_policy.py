from __future__ import annotations

import pytest

from src.recommend_hybrid.common.policy_contracts import AutomationStatus, Priority
from src.recommend_hybrid.oulad.cutoff_router import route_oulad_cutoff

from .conftest import decision


def _request(policy, prediction, cutoff=50, **updates):
    values = dict(
        student_key="oulad-student",
        course_key="oulad-course",
        requested_cutoff=cutoff,
        prediction=prediction,
        max_observation_cutoff=cutoff - 1,
        activity_level=12.0,
        recent_activity_trend=1.0,
        inactivity_streak=2,
        assessment_progress=0.8,
        assessments_due=2,
        grade_trend=None,
        grade_release_verified=False,
        knowledge_gap=None,
    )
    values.update(updates)
    return policy.recommend(**values)


def test_oulad_cutoff_19_abstains(oulad_policy):
    result = _request(oulad_policy, None, cutoff=19, max_observation_cutoff=None)
    assert result.automation_status is AutomationStatus.ABSTAIN
    assert result.abstention_reasons == ("NO_VALIDATED_PREDICTION_ANCHOR",)


def test_oulad_cutoff_20_anchor(oulad_policy, oulad_prediction):
    result = _request(oulad_policy, oulad_prediction, cutoff=20)
    assert result.prediction_anchor.anchor_stage == "EARLY_20"


def test_oulad_cutoff_25_uses_20(oulad_policy, oulad_prediction):
    result = _request(oulad_policy, oulad_prediction, cutoff=25)
    assert result.prediction_anchor.anchor_cutoff == 20
    assert result.prediction_anchor.prediction_age == 5


def test_oulad_cutoff_34_uses_20(oulad_policy, oulad_prediction):
    assert _request(oulad_policy, oulad_prediction, cutoff=34).prediction_anchor.anchor_stage == "EARLY_20"


def test_oulad_cutoff_35_anchor(oulad_policy, oulad_prediction):
    assert _request(oulad_policy, oulad_prediction, cutoff=35).prediction_anchor.anchor_stage == "EARLY_35"


def test_oulad_cutoff_63_uses_50(oulad_policy, oulad_prediction):
    assert _request(oulad_policy, oulad_prediction, cutoff=63).prediction_anchor.anchor_stage == "MIDDLE_50"


def test_oulad_cutoff_76_uses_75(oulad_policy, oulad_prediction):
    assert _request(oulad_policy, oulad_prediction, cutoff=76).prediction_anchor.anchor_stage == "LATE_75"


def test_oulad_final_evaluation_only(oulad_policy, oulad_prediction):
    result = _request(oulad_policy, oulad_prediction, cutoff=100, max_observation_cutoff=None)
    assert result.automation_status is AutomationStatus.EVALUATION_ONLY
    assert result.action_decisions == ()


@pytest.mark.parametrize("cutoff", [-1, 101, float("nan")])
def test_oulad_invalid_cutoff_rejected(oulad_policy, oulad_prediction, cutoff):
    with pytest.raises(ValueError, match="outside"):
        _request(
            oulad_policy,
            oulad_prediction,
            cutoff=cutoff,
            max_observation_cutoff=None,
        )


@pytest.mark.parametrize("cutoff", [20, 25, 34, 35, 49, 50, 63, 74, 75, 76, 99])
def test_oulad_never_uses_future_anchor(oulad_policy, oulad_prediction, cutoff):
    result = _request(oulad_policy, oulad_prediction, cutoff=cutoff)
    assert result.prediction_anchor.anchor_cutoff <= cutoff


def test_oulad_post_cutoff_rejected(oulad_policy, oulad_prediction):
    with pytest.raises(ValueError, match="strictly before"):
        _request(oulad_policy, oulad_prediction, cutoff=50, max_observation_cutoff=50)


def test_oulad_inactivity_monotonicity(oulad_policy, oulad_prediction):
    low = _request(oulad_policy, oulad_prediction, activity_level=8, inactivity_streak=7)
    high = _request(oulad_policy, oulad_prediction, activity_level=8, inactivity_streak=28)
    order = {Priority.LOW: 1, Priority.MEDIUM: 2, Priority.HIGH: 3, Priority.CRITICAL: 4}
    assert order[decision(high, "VLE_ENGAGEMENT").priority] >= order[decision(low, "VLE_ENGAGEMENT").priority]


def test_oulad_completion_monotonicity(oulad_policy, oulad_prediction):
    moderate = _request(oulad_policy, oulad_prediction, assessment_progress=0.7)
    low = _request(oulad_policy, oulad_prediction, assessment_progress=0.2)
    order = {Priority.LOW: 1, Priority.MEDIUM: 2, Priority.HIGH: 3, Priority.CRITICAL: 4}
    assert order[decision(low, "ASSESSMENT_COMPLETION").priority] >= order[decision(moderate, "ASSESSMENT_COMPLETION").priority]


def test_oulad_resolved_assessment_removes_action(oulad_policy, oulad_prediction):
    unresolved = _request(oulad_policy, oulad_prediction, assessment_progress=0.4)
    resolved = _request(oulad_policy, oulad_prediction, assessment_progress=1.0)
    assert decision(unresolved, "ASSESSMENT_COMPLETION").priority is not Priority.NOT_APPLICABLE
    assert decision(resolved, "ASSESSMENT_COMPLETION").priority is Priority.NOT_APPLICABLE


def test_oulad_no_due_assessment_removes_action(oulad_policy, oulad_prediction):
    result = _request(oulad_policy, oulad_prediction, assessments_due=0, assessment_progress=None)
    assert decision(result, "ASSESSMENT_COMPLETION").priority is Priority.NOT_APPLICABLE


def test_oulad_targeted_practice_needs_knowledge_gap(oulad_policy, oulad_prediction):
    missing = _request(oulad_policy, oulad_prediction, knowledge_gap=None)
    present = _request(oulad_policy, oulad_prediction, knowledge_gap="algebraic reasoning")
    assert decision(missing, "TARGETED_PRACTICE").priority is Priority.NOT_APPLICABLE
    assert decision(present, "TARGETED_PRACTICE").priority is not Priority.NOT_APPLICABLE


OULAD_SCENARIOS = tuple(
    (cutoff, activity, inactivity, progress, due)
    for cutoff in (20, 25, 35, 50, 63, 75)
    for activity, inactivity, progress, due in (
        (15.0, 1, 1.0, 2),
        (8.0, 7, 0.7, 2),
        (4.0, 14, 0.4, 3),
        (1.0, 28, 0.2, 4),
        (None, None, None, 0),
    )
)


@pytest.mark.parametrize("cutoff,activity,inactivity,progress,due", OULAD_SCENARIOS)
def test_oulad_scenario_matrix(
    oulad_policy, oulad_prediction, cutoff, activity, inactivity, progress, due
):
    result = _request(
        oulad_policy,
        oulad_prediction,
        cutoff=cutoff,
        activity_level=activity,
        inactivity_streak=inactivity,
        assessment_progress=progress,
        assessments_due=due,
    )
    assert result.prediction_anchor.anchor_cutoff <= cutoff
    assert all(item.action_id != "ATTENDANCE_IMPROVEMENT" for item in result.action_decisions)
