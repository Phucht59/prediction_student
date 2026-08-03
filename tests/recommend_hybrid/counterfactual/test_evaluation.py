from __future__ import annotations

import pytest

from src.recommend_hybrid.counterfactual.evaluation import (
    CounterfactualEvaluationRow,
    aggregate_counterfactual_metrics,
    grouped_counterfactual_metrics,
)


def _row(
    student: str,
    *,
    fold: int = 0,
    stage: str = "MIDDLE_50",
    baseline: float = 0.70,
    counterfactual: float | None = 0.60,
    action: str | None = "VLE_ENGAGEMENT",
    status: str = "COUNTERFACTUAL_SCORED",
    fallback: tuple[str, ...] = (),
):
    return CounterfactualEvaluationRow(
        student_key=student,
        course_key="AAA-2014J",
        stage=stage,
        fold=fold,
        baseline_risk=baseline,
        decision_threshold=0.65,
        status=status,
        top_action_id=action,
        top_counterfactual_risk=counterfactual,
        top_risk_reduction=(
            baseline - counterfactual if counterfactual is not None else None
        ),
        top_utility_score=0.05 if action is not None else None,
        selected_action_count=2 if action is not None else 1,
        selected_workload_minutes=120 if action is not None else 30,
        reference_profile_id="oulad_ref_test" if action is not None else None,
        fallback_reasons=fallback,
    )


def test_aggregate_metrics_report_risk_reduction_without_labels():
    rows = (
        _row("a", baseline=0.70, counterfactual=0.60),
        _row(
            "b",
            baseline=0.80,
            counterfactual=0.73,
            action="STUDY_SCHEDULE",
        ),
        _row(
            "c",
            action=None,
            counterfactual=None,
            status="POLICY_FALLBACK",
            fallback=("MISSING_MODEL_INPUTS",),
        ),
    )
    metrics = aggregate_counterfactual_metrics(rows)
    assert metrics["record_count"] == 3
    assert metrics["scored_count"] == 2
    assert metrics["scored_coverage"] == pytest.approx(2 / 3)
    assert metrics["fallback_rate"] == pytest.approx(1 / 3)
    assert metrics["success_at_0_05"] == 1.0
    assert metrics["threshold_crossing_rate"] == pytest.approx(0.5)
    assert metrics["outcome_labels_used_for_ranking"] is False
    assert metrics["claim_boundary"].endswith("NOT_CAUSAL_EFFECT")


def test_grouped_metrics_use_fold_and_stage_boundaries():
    rows = (
        _row("a", fold=0, stage="EARLY_20"),
        _row("b", fold=1, stage="EARLY_20"),
        _row("c", fold=1, stage="MIDDLE_50"),
    )
    grouped = grouped_counterfactual_metrics(rows)
    assert set(grouped) == {
        "fold_0:EARLY_20",
        "fold_1:EARLY_20",
        "fold_1:MIDDLE_50",
    }


def test_evaluation_row_rejects_inconsistent_risk_delta():
    with pytest.raises(Exception, match="inconsistent"):
        CounterfactualEvaluationRow(
            student_key="bad",
            course_key="AAA-2014J",
            stage="MIDDLE_50",
            fold=0,
            baseline_risk=0.70,
            decision_threshold=0.65,
            status="COUNTERFACTUAL_SCORED",
            top_action_id="VLE_ENGAGEMENT",
            top_counterfactual_risk=0.60,
            top_risk_reduction=0.20,
            top_utility_score=0.05,
            selected_action_count=1,
            selected_workload_minutes=90,
            reference_profile_id="oulad_ref_test",
        )
