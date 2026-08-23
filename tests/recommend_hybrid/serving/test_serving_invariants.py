"""Serving recommendation invariants. No Gemini. No future features."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.prediction.contracts import PredictionResult
from src.recommend_hybrid.serving.contracts import (
    FEATURE_COLUMNS,
    NON_INTERVENTION,
    PersistLabel,
    RouteStatus,
    map_prediction_state,
)
from src.recommend_hybrid.serving.feasibility import invalid_action, rule_label
from src.recommend_hybrid.serving.policy import attach_worklist, decision_from_row


def test_100pct_rejected():
    with pytest.raises(ValueError):
        map_prediction_state("100pct")
    assert "100pct" in NON_INTERVENTION


def test_hybrid_only_prediction_result():
    with pytest.raises(ValueError):
        PredictionResult(
            dataset="oulad",
            record_id="1",
            stage_or_endpoint="35pct",
            risk_probability=0.4,
            predicted_risk=1,
            threshold=0.3,
            model_id="xgboost",
        )


def test_feature_list_has_no_outcome():
    forbidden = {"final_result", "y", "score", "date_unregistration", "g3"}
    assert not forbidden.intersection(FEATURE_COLUMNS)


def test_rule_priority_assess_over_engage():
    row = {
        "missing_assessment_count": 2,
        "due_soon_count": 0,
        "remaining_count": 1,
        "vle_access_available": True,
        "inactivity_streak": 12,
        "active_day_rate": 0.05,
    }
    assert rule_label(row) is PersistLabel.ASSESS


def test_invalid_engage_without_vle():
    row = {"vle_access_available": False, "inactivity_streak": 20, "active_day_rate": 0.0}
    assert invalid_action(PersistLabel.ENGAGE, row) is True
    assert invalid_action(PersistLabel.COUNSEL, row) is False


def test_worklist_is_top_k_by_p():
    frame = pd.DataFrame(
        {
            "code_module": ["AAA"] * 10,
            "code_presentation": ["2013J"] * 10,
            "stage": ["EARLY_35"] * 10,
            "risk_probability": [i / 10 for i in range(10)],
            "query_id": [str(i) for i in range(10)],
        }
    )
    out = attach_worklist(frame, k_frac=0.10)
    assert int(out["in_worklist"].sum()) == 1
    assert out.loc[out["in_worklist"], "risk_probability"].iloc[0] == pytest.approx(0.9)


def test_out_of_budget_emits_counsel():
    row = pd.Series(
        {
            "student_key": "1",
            "course_key": "AAA::2013J",
            "stage": "EARLY_35",
            "in_worklist": False,
            "rank_in_cohort": 9,
            "cohort_size": 10,
            "uncertainty": 0.2,
        }
    )
    decision = decision_from_row(row, action="ASSESS", score=0.8)
    assert decision.route is RouteStatus.OUT_OF_BUDGET
    assert decision.action is PersistLabel.COUNSEL


def test_labels_module_forbids_outcome_columns():
    from src.recommend_hybrid.serving import labels as labels_mod

    source = inspect.getsource(labels_mod)
    assert "frame[\"final_result\"]" not in source
    assert "G3" not in source
