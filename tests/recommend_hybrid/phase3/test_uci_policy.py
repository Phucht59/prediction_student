from __future__ import annotations

from dataclasses import replace

import pytest

from src.recommend_hybrid.common.policy_contracts import (
    AutomationStatus,
    DatasetId,
    EligibilityStatus,
    Priority,
)
from src.recommend_hybrid.uci.stage_router import route_uci_stage

from .conftest import decision


def _request(policy, prediction, **updates):
    values = dict(
        student_key="uci-student",
        course_key="uci-course",
        prediction=prediction,
        g1=None,
        g2=None,
        absences=4,
        study_time=3,
        previous_failures=0,
        next_assessment_available=None,
    )
    values.update(updates)
    return policy.recommend(**values)


def test_uci_stage_s0(uci_prediction):
    anchor = route_uci_stage(g1=None, g2=None, checkpoint_lineage=uci_prediction.checkpoint_lineage)
    assert anchor.anchor_stage == "S0"


def test_uci_stage_s1(uci_prediction):
    anchor = route_uci_stage(g1=11, g2=None, checkpoint_lineage=uci_prediction.checkpoint_lineage)
    assert anchor.anchor_stage == "S1"


def test_uci_stage_s2(uci_prediction):
    anchor = route_uci_stage(g1=11, g2=12, checkpoint_lineage=uci_prediction.checkpoint_lineage)
    assert anchor.anchor_stage == "S2"


def test_uci_rejects_g3(uci_mat_policy, uci_prediction):
    with pytest.raises(ValueError, match="G3"):
        _request(uci_mat_policy, uci_prediction, extra_features={"G3": 8})


def test_uci_mat_por_config_isolation(uci_mat_policy, uci_por_policy, uci_prediction):
    assert uci_mat_policy.config["policy_version"] != uci_por_policy.config["policy_version"]
    assert uci_mat_policy.config["severity_rules"]["absences"] != uci_por_policy.config["severity_rules"]["absences"]
    mat = _request(uci_mat_policy, uci_prediction, absences=10)
    por = _request(
        uci_por_policy,
        replace(uci_prediction, dataset_id=DatasetId.STUDENT_POR),
        absences=10,
    )
    assert mat.policy_version != por.policy_version


def test_uci_absence_monotonicity(uci_mat_policy, uci_prediction):
    low = _request(uci_mat_policy, uci_prediction, absences=6)
    high = _request(uci_mat_policy, uci_prediction, absences=20)
    order = {Priority.LOW: 1, Priority.MEDIUM: 2, Priority.HIGH: 3, Priority.CRITICAL: 4}
    assert order[decision(high, "ATTENDANCE_IMPROVEMENT").priority] >= order[decision(low, "ATTENDANCE_IMPROVEMENT").priority]


def test_uci_studytime_monotonicity(uci_mat_policy, uci_prediction):
    moderate = _request(uci_mat_policy, uci_prediction, study_time=2)
    low = _request(uci_mat_policy, uci_prediction, study_time=1)
    order = {Priority.LOW: 1, Priority.MEDIUM: 2, Priority.HIGH: 3, Priority.CRITICAL: 4}
    assert order[decision(low, "STUDY_SCHEDULE").priority] >= order[decision(moderate, "STUDY_SCHEDULE").priority]


def test_uci_missing_grade_evidence(uci_mat_policy, uci_prediction):
    result = _request(uci_mat_policy, uci_prediction)
    revision = decision(result, "TARGETED_REVISION")
    assert revision.priority is Priority.NOT_APPLICABLE
    assert not any("G1=" in evidence for explanation in result.explanation for evidence in explanation.observed_evidence)


def test_uci_percentage_without_grade_state_abstains(uci_mat_policy, uci_prediction):
    result = _request(uci_mat_policy, uci_prediction, requested_cutoff=40)
    assert result.automation_status is AutomationStatus.ABSTAIN
    assert result.prediction_anchor.anchor_stage is None


def test_uci_g2_without_g1_abstains(uci_mat_policy, uci_prediction):
    result = _request(uci_mat_policy, uci_prediction, g2=10)
    assert result.automation_status is AutomationStatus.ABSTAIN


def test_uci_assessment_preparation_needs_future_assessment(uci_mat_policy, uci_prediction):
    absent = _request(uci_mat_policy, uci_prediction, g1=8, next_assessment_available=False)
    present = _request(uci_mat_policy, uci_prediction, g1=8, next_assessment_available=True)
    assert decision(absent, "ASSESSMENT_PREPARATION").priority is Priority.NOT_APPLICABLE
    assert decision(present, "ASSESSMENT_PREPARATION").priority is not Priority.NOT_APPLICABLE


def test_uci_improvement_never_increases_escalation(uci_mat_policy, uci_prediction):
    flat = _request(uci_mat_policy, uci_prediction, g1=5, g2=5, next_assessment_available=False)
    improved = _request(uci_mat_policy, uci_prediction, g1=5, g2=10, next_assessment_available=False)
    order = {Priority.NOT_APPLICABLE: 0, Priority.LOW: 1, Priority.MEDIUM: 2, Priority.HIGH: 3, Priority.CRITICAL: 4}
    assert order[decision(improved, "ADVISOR_SUPPORT").priority] <= order[decision(flat, "ADVISOR_SUPPORT").priority]


UCI_SCENARIOS = (
    (None, None, 2, 4, 0),
    (None, None, 6, 2, 0),
    (None, None, 12, 1, 1),
    (None, None, 20, 1, 2),
    (14, None, 2, 4, 0),
    (11, None, 6, 3, 0),
    (9, None, 12, 2, 1),
    (5, None, 20, 1, 2),
    (15, 16, 2, 4, 0),
    (12, 14, 6, 3, 0),
    (10, 9, 12, 2, 1),
    (5, 4, 20, 1, 2),
    (8, 12, 0, 4, 0),
    (12, 8, 8, 2, 0),
    (6, 6, 16, 1, 1),
    (18, 18, 1, 4, 0),
    (13, 13, 7, 3, 0),
    (9, 11, 10, 2, 1),
    (11, 9, 14, 2, 1),
    (4, 10, 22, 1, 3),
)


@pytest.mark.parametrize("g1,g2,absences,study_time,failures", UCI_SCENARIOS)
def test_uci_scenario_matrix(
    uci_mat_policy, uci_prediction, g1, g2, absences, study_time, failures
):
    result = _request(
        uci_mat_policy,
        uci_prediction,
        g1=g1,
        g2=g2,
        absences=absences,
        study_time=study_time,
        previous_failures=failures,
        next_assessment_available=g1 is not None,
    )
    assert result.dataset_id.value == "student_mat"
    assert all(decision.priority is Priority.NOT_APPLICABLE for decision in result.action_decisions if decision.eligibility_status not in {EligibilityStatus.ELIGIBLE, EligibilityStatus.REQUIRES_HUMAN_CONTACT})
