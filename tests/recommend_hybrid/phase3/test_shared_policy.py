from __future__ import annotations

from dataclasses import replace

from src.recommend_hybrid.common.policy_contracts import (
    AutomationStatus,
    EligibilityStatus,
    Priority,
)

from .conftest import decision


def test_risk_alone_does_not_create_action(oulad_policy, oulad_prediction):
    result = oulad_policy.recommend(
        student_key="student",
        course_key="course",
        requested_cutoff=50,
        prediction=oulad_prediction,
        max_observation_cutoff=None,
    )
    assert result.automation_status is AutomationStatus.ABSTAIN
    assert not any(item.priority is not Priority.NOT_APPLICABLE for item in result.action_decisions)


def test_uncertainty_never_increases_automation(oulad_policy, oulad_prediction):
    common = dict(
        student_key="student",
        course_key="course",
        requested_cutoff=50,
        max_observation_cutoff=49,
        activity_level=4,
        recent_activity_trend=-4,
        inactivity_streak=14,
        assessment_progress=0.4,
        assessments_due=2,
    )
    low = oulad_policy.recommend(prediction=oulad_prediction, **common)
    caution = oulad_policy.recommend(
        prediction=replace(oulad_prediction, uncertainty=0.60), **common
    )
    high = oulad_policy.recommend(
        prediction=replace(oulad_prediction, uncertainty=0.69), **common
    )
    order = {AutomationStatus.ABSTAIN: 0, AutomationStatus.PARTIAL: 1, AutomationStatus.FULL: 2}
    assert order[high.automation_status] <= order[caution.automation_status] <= order[low.automation_status]


def test_missing_evidence_not_zero_imputed(oulad_policy, oulad_prediction):
    result = oulad_policy.recommend(
        student_key="student",
        course_key="course",
        requested_cutoff=50,
        prediction=oulad_prediction,
        max_observation_cutoff=None,
    )
    evidence = [item for item in result.action_decisions if item.missing_evidence]
    assert evidence
    assert all(0 not in item.missing_evidence for item in evidence)


def test_ineligible_action_has_no_priority(oulad_policy, oulad_prediction):
    result = oulad_policy.recommend(
        student_key="student",
        course_key="course",
        requested_cutoff=20,
        prediction=oulad_prediction,
        max_observation_cutoff=19,
        activity_level=8,
        inactivity_streak=7,
    )
    assert all(
        item.priority is Priority.NOT_APPLICABLE
        for item in result.action_decisions
        if item.eligibility_status
        not in {EligibilityStatus.ELIGIBLE, EligibilityStatus.REQUIRES_HUMAN_CONTACT}
    )


def test_explanation_matches_evidence(oulad_policy, oulad_prediction):
    result = oulad_policy.recommend(
        student_key="student",
        course_key="course",
        requested_cutoff=50,
        prediction=oulad_prediction,
        max_observation_cutoff=49,
        activity_level=4,
        inactivity_streak=14,
        assessment_progress=0.4,
        assessments_due=2,
    )
    by_action = {item.action_id: item for item in result.action_decisions}
    for explanation in result.explanation:
        supporting = by_action[explanation.action].supporting_evidence
        assert supporting
        assert all(item.feature_name in " ".join(explanation.observed_evidence) for item in supporting)
        assert all(item.source_lineage in " ".join(explanation.observed_evidence) for item in supporting)


def test_deterministic_replay(oulad_policy, oulad_prediction):
    arguments = dict(
        student_key="student",
        course_key="course",
        requested_cutoff=63,
        prediction=oulad_prediction,
        max_observation_cutoff=62,
        activity_level=4,
        recent_activity_trend=-5,
        inactivity_streak=14,
        assessment_progress=0.4,
        assessments_due=2,
    )
    assert oulad_policy.recommend(**arguments).to_dict() == oulad_policy.recommend(**arguments).to_dict()


def test_cross_dataset_action_isolation(
    uci_mat_policy, uci_prediction, oulad_policy, oulad_prediction
):
    uci = uci_mat_policy.recommend(
        student_key="u",
        course_key="c",
        prediction=uci_prediction,
        g1=8,
        g2=None,
        absences=12,
        study_time=1,
        previous_failures=1,
        next_assessment_available=True,
    )
    oulad = oulad_policy.recommend(
        student_key="o",
        course_key="c",
        requested_cutoff=50,
        prediction=oulad_prediction,
        max_observation_cutoff=49,
        activity_level=4,
        inactivity_streak=14,
        assessment_progress=0.4,
        assessments_due=2,
    )
    assert all(item.action_id != "VLE_ENGAGEMENT" for item in uci.action_decisions)
    assert all(item.action_id != "ATTENDANCE_IMPROVEMENT" for item in oulad.action_decisions)


def test_cross_dataset_prediction_context_rejected(
    uci_por_policy, uci_prediction, oulad_policy
):
    import pytest

    with pytest.raises(ValueError, match="dataset mismatch"):
        uci_por_policy.recommend(
            student_key="u",
            course_key="c",
            prediction=uci_prediction,
            g1=None,
            g2=None,
            absences=2,
            study_time=3,
            previous_failures=0,
            next_assessment_available=None,
        )
    with pytest.raises(ValueError, match="OULAD prediction"):
        oulad_policy.recommend(
            student_key="o",
            course_key="c",
            requested_cutoff=50,
            prediction=uci_prediction,
            max_observation_cutoff=None,
        )


def test_no_neural_ranker(phase3_root):
    files = list((phase3_root / "src/recommend_hybrid/common").glob("*.py"))
    files += list((phase3_root / "src/recommend_hybrid/uci").glob("*.py"))
    files += list((phase3_root / "src/recommend_hybrid/oulad").glob("*.py"))
    code = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert "class HybridActionRanker" not in code
    assert "relevance_score" not in code


def test_no_expert_label_dependency(uci_mat_policy, oulad_policy):
    assert uci_mat_policy.common["expert_label_dependency"] is False
    assert oulad_policy.common["recommendation_training_required"] is False


def test_contradictory_evidence_abstains(oulad_policy, oulad_prediction):
    result = oulad_policy.recommend(
        student_key="student",
        course_key="course",
        requested_cutoff=50,
        prediction=oulad_prediction,
        max_observation_cutoff=49,
        activity_level=15,
        inactivity_streak=14,
        assessment_progress=0.5,
        assessments_due=2,
    )
    assert result.automation_status is AutomationStatus.ABSTAIN
    assert result.abstention_reasons == ("ACTIVITY_INACTIVITY_EVIDENCE_CONFLICT",)
