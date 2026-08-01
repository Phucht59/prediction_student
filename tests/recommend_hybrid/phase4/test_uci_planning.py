from __future__ import annotations

from src.recommend_hybrid.common.policy_contracts import DatasetId
from src.recommend_hybrid.oulad.action_catalog import OULAD_ACTIONS

from .conftest import uci_request


def _features(plan):
    return {e.feature_name for action in plan.selected_actions for e in action.supporting_evidence}


def test_s0_plan_does_not_use_g1_g2(pipeline, uci_prediction):
    plan = pipeline.generate(uci_request(uci_prediction, g1=None, g2=None, next_assessment_available=None))
    assert not {"G1", "G2"} & _features(plan)


def test_s1_plan_does_not_use_g2(pipeline, uci_prediction):
    plan = pipeline.generate(uci_request(uci_prediction, g1=8, g2=None))
    assert "G2" not in _features(plan)


def test_s2_plan_does_not_use_g3(pipeline, uci_prediction):
    plan = pipeline.generate(uci_request(uci_prediction, g1=10, g2=8))
    assert "G3" not in _features(plan)


def test_mat_por_isolation(pipeline, uci_prediction):
    mat = pipeline.generate(uci_request(uci_prediction, DatasetId.STUDENT_MAT))
    por = pipeline.generate(uci_request(uci_prediction, DatasetId.STUDENT_POR))
    assert mat.dataset_id == "student_mat" and por.dataset_id == "student_por"
    assert mat.policy_version != por.policy_version


def test_uci_rejects_oulad_actions(pipeline, uci_prediction):
    plan = pipeline.generate(uci_request(uci_prediction))
    assert not set(OULAD_ACTIONS).difference({"STUDY_SCHEDULE", "INSTRUCTOR_CONTACT", "PROGRESS_MONITORING", "LEARNING_CONSOLIDATION"}) & {item.action_id for item in plan.selected_actions}


def test_uci_periods_are_business_periods(pipeline, uci_prediction):
    plan = pipeline.generate(uci_request(uci_prediction))
    assert plan.plan_periods == ("CURRENT_PERIOD", "NEXT_ASSESSMENT", "FOLLOW_UP")
    assert all(item.scheduled_period in plan.plan_periods for item in plan.selected_actions)
