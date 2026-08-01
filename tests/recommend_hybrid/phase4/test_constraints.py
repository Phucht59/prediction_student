from __future__ import annotations

from collections import Counter

from src.recommend_hybrid.common.plan_contracts import PlanStatus

from .conftest import oulad_request, uci_request


def test_action_cap(pipeline, oulad_prediction):
    plan = pipeline.generate(oulad_request(oulad_prediction))
    assert len(plan.selected_actions) <= 4


def test_workload_cap(pipeline, oulad_prediction):
    plan = pipeline.generate(oulad_request(oulad_prediction))
    workload = Counter()
    for action in plan.selected_actions:
        workload[action.scheduled_period] += action.weekly_minutes
    assert all(value <= 180 for value in workload.values())


def test_duplicate_removal(pipeline, oulad_prediction):
    plan = pipeline.generate(oulad_request(oulad_prediction))
    ids = [item.action_id for item in plan.selected_actions]
    assert len(ids) == len(set(ids))


def test_prerequisite_ordering(pipeline, oulad_prediction):
    plan = pipeline.generate(
        oulad_request(
            oulad_prediction,
            activity_level=12,
            recent_activity_trend=1,
            inactivity_streak=2,
            assessment_progress=0.2,
            assessments_due=2,
            active_contraindications=(
                "ACTIVE_ADVISOR_CASE",
                "CONTACT_ALREADY_OPEN",
                "NO_OPEN_ASSESSMENT",
            ),
        )
    )
    ids = [item.action_id for item in plan.selected_actions]
    assert "TARGETED_PRACTICE" in ids
    assert ids.index("DIAGNOSTIC_CHECK") < ids.index("TARGETED_PRACTICE")


def test_contraindication_rejection(pipeline, uci_prediction):
    plan = pipeline.generate(
        uci_request(uci_prediction, active_contraindications=("CONTACT_ALREADY_OPEN",))
    )
    assert "INSTRUCTOR_CONTACT" not in {item.action_id for item in plan.selected_actions}
    assert plan.automation_status is PlanStatus.PARTIAL


def test_deterministic_tie_break(pipeline, uci_prediction):
    request = uci_request(uci_prediction)
    assert pipeline.generate(request).to_dict() == pipeline.generate(request).to_dict()


def test_abstain_has_zero_actions(pipeline):
    plan = pipeline.generate(oulad_request(None, cutoff=19, max_observation_cutoff=None))
    assert plan.automation_status is PlanStatus.ABSTAIN
    assert plan.selected_actions == ()


def test_evaluation_only_has_zero_actions(pipeline, oulad_prediction):
    plan = pipeline.generate(oulad_request(oulad_prediction, cutoff=100, max_observation_cutoff=None))
    assert plan.automation_status is PlanStatus.EVALUATION_ONLY
    assert plan.selected_actions == ()
