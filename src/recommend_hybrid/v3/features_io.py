"""Build RecommendationFeatures from a learner-stage or action row."""

from __future__ import annotations

import math

import pandas as pd

from .contracts import RecommendationFeatures, Stage


def _isna(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def maybe_int(value) -> int | None:
    if _isna(value):
        return None
    return int(value)


def maybe_float(value) -> float | None:
    if _isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def maybe_bool(value) -> bool | None:
    if _isna(value):
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def features_from_row(row: pd.Series) -> RecommendationFeatures:
    contraindications = row.get("contraindications", frozenset())
    if _isna(contraindications):
        frozen = frozenset()
    elif isinstance(contraindications, (set, frozenset, tuple, list)):
        frozen = frozenset(str(item) for item in contraindications)
    else:
        frozen = frozenset()
    return RecommendationFeatures(
        student_key=str(row["student_key"]),
        course_key=str(row["course_key"]),
        record_id=str(row["record_id"]),
        stage=Stage(str(row["stage"])),
        cutoff_day=int(row["cutoff_day"]),
        risk_probability=float(row["risk_probability"]),
        predicted_risk=int(row["predicted_risk"]),
        prediction_threshold=float(row["prediction_threshold"]),
        uncertainty=float(row["uncertainty"]),
        course_progress=float(row["course_progress"]),
        assessment_progress=maybe_float(row.get("assessment_progress")),
        assessments_due=maybe_int(row.get("assessments_due")),
        missing_assessment_count=maybe_int(row.get("missing_assessment_count")),
        due_soon_count=maybe_int(row.get("due_soon_count")),
        completion_rate=maybe_float(row.get("completion_rate")),
        assessment_window_open=maybe_bool(row.get("assessment_window_open")),
        time_to_deadline_days=maybe_int(row.get("time_to_deadline_days")),
        inactivity_streak=maybe_int(row.get("inactivity_streak")),
        active_day_rate=maybe_float(row.get("active_day_rate")),
        recent_activity_trend=maybe_float(row.get("recent_activity_trend")),
        regularity_score=maybe_float(row.get("regularity_score")),
        content_coverage=maybe_float(row.get("content_coverage")),
        knowledge_gap_evidence=maybe_bool(row.get("knowledge_gap_evidence")),
        quiz_activity=maybe_float(row.get("quiz_activity")),
        quiz_available=maybe_bool(row.get("quiz_available")),
        vle_access_available=maybe_bool(row.get("vle_access_available")),
        study_material_available=maybe_bool(row.get("study_material_available")),
        contraindications=frozen,
    )
