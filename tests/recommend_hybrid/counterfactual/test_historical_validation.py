from __future__ import annotations

import pytest

from src.recommend_hybrid.counterfactual.historical_validation import (
    HistoricalTrajectoryRow,
    aggregate_historical_metrics,
)


def _row(
    student: str,
    *,
    aligned: bool | None,
    risk: float | None,
    favorable: bool,
):
    return HistoricalTrajectoryRow(
        student_key=student,
        course_key="AAA-2014J",
        stage="MIDDLE_50",
        action_id="VLE_ENGAGEMENT",
        behavior_aligned=aligned,
        next_stage_risk=risk,
        favorable_final_outcome=favorable,
    )


def test_historical_metrics_report_observational_differences_only():
    metrics = aggregate_historical_metrics(
        (
            _row("a", aligned=True, risk=0.40, favorable=True),
            _row("b", aligned=True, risk=0.50, favorable=True),
            _row("c", aligned=False, risk=0.70, favorable=False),
            _row("d", aligned=None, risk=None, favorable=False),
        )
    )
    assert metrics["record_count"] == 4
    assert metrics["behavior_evaluable_count"] == 3
    assert metrics["behavior_alignment_rate"] == pytest.approx(2 / 3)
    assert metrics["observed_next_stage_risk_difference"] == pytest.approx(
        0.25
    )
    assert metrics["observed_favorable_outcome_difference"] == 1.0
    assert metrics["used_for_action_ranking"] is False
    assert metrics["claim_boundary"].endswith("NOT_CAUSAL_EFFECT")


def test_historical_metrics_handle_missing_comparison_group():
    metrics = aggregate_historical_metrics(
        (_row("a", aligned=True, risk=0.40, favorable=True),)
    )
    assert metrics["observed_next_stage_risk_difference"] is None
    assert metrics["observed_favorable_outcome_difference"] is None
