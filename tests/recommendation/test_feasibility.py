from __future__ import annotations

import pandas as pd

from src.recommendation.feasibility import build_feasibility_frame, validate_feasibility


def _state() -> pd.DataFrame:
    return pd.DataFrame([
        {"case_id": "c1", "stage": "20pct", "missing_assessments": 0, "vle_available": True, "quiz_activity": 0.0},
        {"case_id": "c2", "stage": "35pct", "missing_assessments": 2, "vle_available": True, "quiz_activity": 4.0},
    ])


def test_five_action_rows_and_deterministic_validation():
    state = _state()
    frame = build_feasibility_frame(state)
    assert len(frame) == 10
    assert validate_feasibility(frame, state) == []
    assert frame.groupby("case_id").size().to_dict() == {"c1": 5, "c2": 5}


def test_zero_quiz_activity_is_unknown_not_infeasible():
    frame = build_feasibility_frame(_state())
    row = frame[(frame.case_id == "c1") & (frame.action_id == "A5")].iloc[0]
    assert row.feasibility_status == "UNKNOWN"
    assert "ZERO" in row.reason_code


def test_assessment_recovery_without_missing_assessments_is_not_applicable():
    frame = build_feasibility_frame(_state())
    row = frame[(frame.case_id == "c1") & (frame.action_id == "A1")].iloc[0]
    assert row.feasibility_status == "INFEASIBLE"
    assert row.reason_code == "NO_MISSING_ASSESSMENTS_NOT_APPLICABLE"
