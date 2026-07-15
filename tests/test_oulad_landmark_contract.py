from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from src.studies.oulad.cohort import FORECASTS, materialize_landmark_cohort, presentation_sort_key, weekly_bounds
from src.studies.oulad.materialize import CHANNELS


ROOT = Path(__file__).resolve().parents[1]


def test_forecast_contract_and_weekly_half_open_bins():
    assert FORECASTS == {"F1_EARLY": 0.20, "F2_MIDDLE": 0.50, "F3_LATE": 0.80}
    assert weekly_bounds(20) == [(0, 7), (7, 14), (14, 20)]
    assert presentation_sort_key("2014B") < presentation_sort_key("2014J")


def test_landmark_active_boundary_and_target_separation():
    info = pd.DataFrame([
        {"code_module": "AAA", "code_presentation": "2014J", "id_student": 1, "num_of_prev_attempts": 0, "studied_credits": 60, "final_result": "Withdrawn"},
        {"code_module": "AAA", "code_presentation": "2014J", "id_student": 2, "num_of_prev_attempts": 0, "studied_credits": 60, "final_result": "Pass"},
    ])
    registration = pd.DataFrame([
        {"code_module": "AAA", "code_presentation": "2014J", "id_student": 1, "date_registration": -5, "date_unregistration": 50},
        {"code_module": "AAA", "code_presentation": "2014J", "id_student": 2, "date_registration": -5, "date_unregistration": 49},
    ])
    courses = pd.DataFrame([{"code_module": "AAA", "code_presentation": "2014J", "module_presentation_length": 100}])
    cohort, targets, flow = materialize_landmark_cohort(info, registration, courses, "F2_MIDDLE")
    assert cohort["id_student"].tolist() == [1]
    assert targets["target_at_risk"].tolist() == [1]
    assert "final_result" not in cohort.columns and "date_unregistration" not in cohort.columns
    assert flow["excluded_withdrawn_before_cutoff"] == 1


def test_primary_channels_exclude_target_and_sensitive_fields():
    forbidden = {"final_result", "date_unregistration", "gender", "region", "disability", "age_band", "imd_band"}
    assert not forbidden.intersection(CHANNELS)
    assert {"total_clicks", "cumulative_weighted_score", "score_missing_mask"}.issubset(CHANNELS)


def test_local_oulad_small_table_keys():
    raw = ROOT / "data" / "raw"
    info = pd.read_csv(raw / "studentInfo.csv")
    registration = pd.read_csv(raw / "studentRegistration.csv")
    courses = pd.read_csv(raw / "courses.csv")
    assert len(info) == len(registration) == 32593
    assert not info.duplicated(["code_module", "code_presentation", "id_student"]).any()
    assert not courses.duplicated(["code_module", "code_presentation"]).any()
