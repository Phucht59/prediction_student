from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.studies.common.hashing import stable_record_id


FORECASTS = {"F1_EARLY": 0.20, "F2_MIDDLE": 0.50, "F3_LATE": 0.80}
POSITIVE_RESULTS = {"Withdrawn", "Fail"}
NEGATIVE_RESULTS = {"Pass", "Distinction"}


def presentation_sort_key(value: str) -> tuple[int, int]:
    text = str(value)
    if len(text) != 5 or text[-1] not in {"B", "J"}:
        raise ValueError(f"Invalid OULAD presentation code: {value}")
    return int(text[:4]), 0 if text[-1] == "B" else 1


def materialize_landmark_cohort(
    student_info: pd.DataFrame,
    registration: pd.DataFrame,
    courses: pd.DataFrame,
    forecast_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    if forecast_id not in FORECASTS:
        raise KeyError(forecast_id)
    keys = ["code_module", "code_presentation", "id_student"]
    if student_info.duplicated(keys).any() or registration.duplicated(keys).any():
        raise ValueError("Duplicate OULAD observation key")
    merged = student_info.merge(registration, on=keys, how="inner", validate="one_to_one")
    merged = merged.merge(courses, on=["code_module", "code_presentation"], how="left", validate="many_to_one")
    if len(merged) != len(student_info) or merged["module_presentation_length"].isna().any():
        raise ValueError("Broken studentInfo/registration/courses join")
    if not set(merged["final_result"]).issubset(POSITIVE_RESULTS | NEGATIVE_RESULTS):
        raise ValueError("Unexpected final_result")
    merged["cutoff_day"] = np.floor(merged["module_presentation_length"] * FORECASTS[forecast_id]).astype(int)
    registered = merged["date_registration"].notna() & (merged["date_registration"] < merged["cutoff_day"])
    active = merged["date_unregistration"].isna() | (merged["date_unregistration"] >= merged["cutoff_day"])
    selected = merged.loc[registered & active].copy()
    selected["record_id"] = [stable_record_id(module, presentation, int(student)) for module, presentation, student in selected[keys].itertuples(index=False, name=None)]
    selected["target_at_risk"] = selected["final_result"].isin(POSITIVE_RESULTS).astype(int)
    selected["presentation_season"] = selected["code_presentation"].str[-1]
    selected["registration_lead_time"] = -pd.to_numeric(selected["date_registration"], errors="coerce")
    selected["valid_sequence_length"] = selected["cutoff_day"].map(lambda value: int(math.ceil(value / 7)))
    if selected["record_id"].duplicated().any():
        raise ValueError("Duplicate stable record_id")
    cohort_columns = keys + ["record_id", "cutoff_day", "valid_sequence_length", "code_module", "presentation_season", "num_of_prev_attempts", "studied_credits", "registration_lead_time", "module_presentation_length"]
    cohort_columns = list(dict.fromkeys(cohort_columns))
    cohort = selected[cohort_columns].reset_index(drop=True)
    targets = selected[["record_id", "target_at_risk", "final_result"]].rename(columns={"final_result": "original_final_result"}).reset_index(drop=True)
    flow = {
        "source_student_info": len(student_info),
        "joined_records": len(merged),
        "registered_before_cutoff": int(registered.sum()),
        "active_at_cutoff": int(active.sum()),
        "primary_cohort": len(selected),
        "excluded_not_registered": int((~registered).sum()),
        "excluded_withdrawn_before_cutoff": int((registered & ~active).sum()),
    }
    return cohort, targets, flow


def weekly_bounds(cutoff_day: int) -> list[tuple[int, int]]:
    return [(start, min(start + 7, cutoff_day)) for start in range(0, cutoff_day, 7)]
