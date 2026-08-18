"""Validation gates for the persisted Student Learning State."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts.state import make_case_id


RECOMMENDATION_STAGES = ("20pct", "35pct", "50pct", "75pct")
RATIO_COLUMNS = ("active_days_ratio", "assessment_completion", "course_progress")
COUNT_COLUMNS = ("inactive_streak", "missing_assessments", "quiz_activity")
BOOLEAN_COLUMNS = ("vle_available", "recommendation_eligible")


def validate_student_state(frame: pd.DataFrame, *, stages: tuple[str, ...] = RECOMMENDATION_STAGES) -> list[str]:
    errors: list[str] = []
    required = {"case_id", "dataset", "student_id", "record_id", "module", "presentation", "enrollment_identity", "stage", "outer_fold", "risk_probability", "risk_band", "prediction_source_version"}
    errors.extend(f"missing:{column}" for column in sorted(required.difference(frame.columns)))
    if errors:
        return errors
    if frame["case_id"].duplicated().any(): errors.append("case_id_not_unique")
    expected_case = [make_case_id(d, r, s) for d, r, s in zip(frame.dataset, frame.record_id, frame.stage, strict=True)]
    if frame.case_id.astype(str).tolist() != expected_case: errors.append("case_id_not_deterministic")
    if frame.student_id.isna().any() or frame.record_id.isna().any(): errors.append("identity_null")
    if frame.module.isna().any() or frame.presentation.isna().any() or frame.enrollment_identity.isna().any(): errors.append("enrollment_identity_null")
    oulad = frame.dataset.astype(str).str.casefold().eq("oulad")
    if oulad.any() and (frame.loc[oulad, "enrollment_identity"].astype(str) != frame.loc[oulad, "record_id"].astype(str)).any(): errors.append("oulad_enrollment_identity_mismatch")
    if not frame.stage.isin(stages).all(): errors.append("invalid_stage_or_final_included")
    if frame.outer_fold.isna().any() or not np.issubdtype(frame.outer_fold.dtype, np.integer): errors.append("invalid_outer_fold")
    probability = pd.to_numeric(frame.risk_probability, errors="coerce")
    if probability.isna().any() or not ((probability >= 0) & (probability <= 1)).all(): errors.append("probability_out_of_bounds")
    for column in RATIO_COLUMNS:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any() or not ((values >= 0) & (values <= 1)).all(): errors.append(f"ratio_out_of_bounds:{column}")
    for column in COUNT_COLUMNS:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any() or (values < 0).any(): errors.append(f"count_out_of_bounds:{column}")
    for column in BOOLEAN_COLUMNS:
        if column in frame and not frame[column].dropna().map(lambda value: isinstance(value, (bool, np.bool_))).all(): errors.append(f"not_boolean:{column}")
    if frame.prediction_source_version.isna().any(): errors.append("prediction_lineage_missing")
    forbidden = {"target", "final_result", "score", "date_unregistration"}.intersection(frame.columns)
    if forbidden: errors.append(f"forbidden_columns:{','.join(sorted(forbidden))}")
    return errors
