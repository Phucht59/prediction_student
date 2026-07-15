from __future__ import annotations

import pandas as pd

from src.studies.oulad.splits import build_common_split_manifests


def _frame(forecast: str):
    rows = []
    targets = []
    for presentation in ["2013J", "2014J"]:
        for student in range(100 if presentation == "2013J" else 1000, (100 if presentation == "2013J" else 1000) + 80):
            record = f"{forecast}-{presentation}-{student}"
            rows.append({"record_id": record, "code_module": "AAA", "code_presentation": presentation, "id_student": student, "cutoff_day": 50, "valid_sequence_length": 8})
            targets.append({"record_id": record, "target_at_risk": student % 2})
    return pd.DataFrame(rows), pd.DataFrame(targets)


def test_future_and_grouped_outer_split_have_zero_student_overlap():
    frames = {forecast: _frame(forecast) for forecast in ["F1_EARLY", "F2_MIDDLE", "F3_LATE"]}
    support = {"historical_total_min": 50, "historical_positive_min": 10, "historical_negative_min": 10, "future_total_min": 50, "future_positive_min": 10, "future_negative_min": 10}
    manifest, future, _ = build_common_split_manifests(frames, support)
    development = manifest[manifest["role"] == "historical_development"]
    held = manifest[manifest["role"] == "future_candidate"]
    assert not set(development["id_student"]) & set(held["id_student"])
    assert set(development["outer_fold"].astype(int)) == {0, 1, 2}
    for student, group in development.groupby("id_student"):
        assert group["outer_fold"].nunique() == 1
