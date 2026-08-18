from __future__ import annotations

from src.recommendation.labeling import build_label_payload, validate_label_payload


def test_label_payload_is_pseudonymous_and_unlabeled():
    row = {
        "case_id": "c1", "stage": "20pct", "risk_probability": 0.4, "risk_band": "medium",
        "inactive_streak": 1.0, "active_days_ratio": 0.5, "recent_activity": 2.0,
        "activity_trend": 0.1, "assessment_completion": 0.5, "missing_assessments": 1,
        "course_progress": 0.2, "quiz_activity": 2.0, "vle_available": True,
        "student_id": "real-student",
    }
    payload = build_label_payload(
        row,
        panel="Panel A",
        feasibility_statuses={"A1": "INFEASIBLE", "A2": "FEASIBLE", "A3": "FEASIBLE", "A4": "UNKNOWN", "A5": "UNKNOWN"},
    )
    assert validate_label_payload(payload) == []
    assert "student_id" not in payload
    assert "final_result" not in payload
