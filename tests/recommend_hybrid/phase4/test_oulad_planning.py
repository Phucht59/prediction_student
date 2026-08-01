from __future__ import annotations

import pytest

from src.recommend_hybrid.common.plan_contracts import PlanStatus
from src.recommend_hybrid.uci.action_catalog import UCI_ACTIONS

from .conftest import oulad_request


def test_arbitrary_cutoff_preserves_anchor(pipeline, oulad_prediction):
    plan = pipeline.generate(oulad_request(oulad_prediction, cutoff=63))
    assert plan.requested_cutoff == 63 and plan.prediction_anchor == 50


def test_plan_rejects_post_cutoff_data(pipeline, oulad_prediction):
    with pytest.raises(ValueError, match="strictly before"):
        pipeline.generate(oulad_request(oulad_prediction, cutoff=63, max_observation_cutoff=63))


def test_plan_does_not_extend_past_course_end(pipeline, oulad_prediction):
    plan = pipeline.generate(oulad_request(oulad_prediction, cutoff=96))
    assert plan.plan_periods == ("IMMEDIATE",)


def test_final_has_no_intervention(pipeline, oulad_prediction):
    plan = pipeline.generate(oulad_request(oulad_prediction, cutoff=100, max_observation_cutoff=None))
    assert plan.automation_status is PlanStatus.EVALUATION_ONLY and not plan.selected_actions


def test_short_remaining_time_is_partial(pipeline, oulad_prediction):
    plan = pipeline.generate(oulad_request(oulad_prediction, cutoff=96))
    assert plan.automation_status is PlanStatus.PARTIAL
    assert len(plan.selected_actions) <= 1


def test_oulad_rejects_uci_only_actions(pipeline, oulad_prediction):
    plan = pipeline.generate(oulad_request(oulad_prediction))
    shared = {"STUDY_SCHEDULE", "INSTRUCTOR_CONTACT", "PROGRESS_MONITORING", "LEARNING_CONSOLIDATION"}
    assert not (set(UCI_ACTIONS) - shared) & {item.action_id for item in plan.selected_actions}
