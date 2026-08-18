"""Frozen binary OULAD contract and D3 exposure-safe transformations."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import assert_binary_target, oulad_risk_target
from .common import UnifiedHybridData


OULAD_TEMPORAL_CHANNELS = (
    "activity_intensity_log1p", "active_days", "unique_sites", "unique_activity_types",
    "content_activity", "forum_activity", "quiz_activity", "assessment_related_activity",
    "weekly_submissions", "weekly_late_submissions", "week_exposure_fraction",
)
OULAD_AGGREGATE_CHANNELS = (
    "cumulative_activity", "mean_weekly_activity", "recent_activity", "recent_historical_activity_ratio",
    "activity_trend", "current_inactivity_streak", "cumulative_inactive_weeks", "days_since_last_activity",
    "assessments_due_to_date", "submitted_due_to_date", "completion_rate", "missed_due_count", "late_submission_rate",
)
COUNT_CHANNELS = ("activity_intensity_log1p", "content_activity", "forum_activity", "quiz_activity", "assessment_related_activity")
UNIQUE_CHANNELS = ("unique_sites", "unique_activity_types")
OULAD_FORBIDDEN_PREDICTORS = ("final_result", "target", "score", "date_unregistration")
OULAD_ENDPOINTS = ("20pct", "35pct", "50pct", "75pct", "FINAL-100")


def load_oulad_static_tables(raw_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the cutoff-safe static OULAD tables and construct the binary target.

    Longitudinal event tensors are intentionally supplied separately through
    ``build_oulad_array_view``. This loader owns only the static join and
    provenance-safe identifiers, so it cannot accidentally use an event or
    outcome field as a predictor.
    """
    raw_path = Path(raw_dir)
    courses = pd.read_csv(raw_path / "courses.csv")
    student_info = pd.read_csv(raw_path / "studentInfo.csv")
    student_registration = pd.read_csv(raw_path / "studentRegistration.csv")
    if "length" in courses.columns and "module_presentation_length" not in courses.columns:
        courses = courses.rename(columns={"length": "module_presentation_length"})
    merged = student_info.merge(
        student_registration,
        on=["code_module", "code_presentation", "id_student"],
        how="inner",
    ).merge(courses, on=["code_module", "code_presentation"], how="inner")
    allowed = {"Fail", "Withdrawn", "Pass", "Distinction"}
    unknown = set(merged["final_result"].dropna().unique()) - allowed
    if unknown:
        raise ValueError(f"Unknown OULAD final_result values: {sorted(unknown)}")
    merged = merged.copy()
    merged["target"] = merged["final_result"].isin({"Fail", "Withdrawn"}).astype(np.int64)
    merged["presentation_season"] = merged["code_presentation"].astype(str).str[-1]
    merged["registration_lead_time"] = -merged["date_registration"].fillna(0.0).astype(float)
    merged["group_id"] = merged["id_student"].astype(str)
    merged["record_id"] = merged.apply(
        lambda row: hashlib.sha256(
            f"oulad|{row['code_module']}|{row['code_presentation']}|{row['id_student']}".encode()
        ).hexdigest()[:24],
        axis=1,
    )
    assert_binary_target(merged["target"].to_numpy(), name="OULAD target")
    return courses, student_registration, merged


def validate_oulad_predictor_columns(columns: list[str] | tuple[str, ...] | set[str]) -> None:
    forbidden = sorted(set(columns) & set(OULAD_FORBIDDEN_PREDICTORS))
    if forbidden:
        raise ValueError(f"Forbidden OULAD predictor columns: {forbidden}")


def build_oulad_array_view(*, static: np.ndarray, temporal: np.ndarray, temporal_mask: np.ndarray, lengths: np.ndarray,
                           aggregate: np.ndarray, aggregate_available: np.ndarray, progress: np.ndarray,
                           final_result, record_id, group_id, endpoint: str) -> UnifiedHybridData:
    if endpoint not in OULAD_ENDPOINTS:
        raise ValueError(f"unknown OULAD endpoint: {endpoint}")
    target = oulad_risk_target(final_result)
    view = UnifiedHybridData(static=np.asarray(static, np.float32), temporal=np.asarray(temporal, np.float32),
                             temporal_mask=np.asarray(temporal_mask, bool), lengths=np.asarray(lengths, np.int64),
                             aggregate=np.asarray(aggregate, np.float32), aggregate_available=np.asarray(aggregate_available),
                             progress=np.asarray(progress, np.float32), target=target,
                             record_id=np.asarray(record_id).astype(str), group_id=np.asarray(group_id).astype(str),
                             metadata={"dataset": "oulad", "endpoint": endpoint, "target_rule": "Fail/Withdrawn => risk", "forbidden_predictors": list(OULAD_FORBIDDEN_PREDICTORS)})
    view.validate()
    return view


def _raw_counts(temporal: np.ndarray) -> np.ndarray:
    raw = temporal.copy()
    raw[..., OULAD_TEMPORAL_CHANNELS.index("activity_intensity_log1p")] = np.expm1(np.clip(raw[..., OULAD_TEMPORAL_CHANNELS.index("activity_intensity_log1p")], 0.0, 30.0))
    return raw


def apply_d3_variant(view: UnifiedHybridData) -> UnifiedHybridData:
    """Apply the selected D3_both_safe representation without touching labels."""
    temporal = view.temporal.copy().astype(np.float32)
    aggregate = view.aggregate.copy().astype(np.float32)
    raw = _raw_counts(view.temporal)
    ti = {name: OULAD_TEMPORAL_CHANNELS.index(name) for name in OULAD_TEMPORAL_CHANNELS}
    ai = {name: OULAD_AGGREGATE_CHANNELS.index(name) for name in OULAD_AGGREGATE_CHANNELS}
    exposure = view.temporal[..., ti["week_exposure_fraction"]]
    days = exposure * 7.0
    valid = view.temporal_mask & (days > 0)
    for name in COUNT_CHANNELS:
        idx = ti[name]
        rate = np.divide(raw[..., idx], days, out=np.zeros_like(raw[..., idx]), where=valid)
        temporal[..., idx] = np.log1p(np.maximum(rate, 0.0))
    idx = ti["active_days"]
    temporal[..., idx] = np.divide(raw[..., idx], days, out=np.zeros_like(raw[..., idx]), where=valid)
    for name in UNIQUE_CHANNELS:
        idx = ti[name]
        rate = np.divide(raw[..., idx], days, out=np.zeros_like(raw[..., idx]), where=valid)
        temporal[..., idx] = np.log1p(np.maximum(rate, 0.0))
    temporal[~view.temporal_mask] = 0.0
    total_days = days * valid
    activity = raw[..., ti["activity_intensity_log1p"]]
    daily_total = np.divide((activity * valid).sum(1), total_days.sum(1), out=np.zeros(len(view.record_id), np.float32), where=total_days.sum(1) > 0)
    last = np.maximum(view.temporal_mask.sum(1) - 1, 0)
    rows = np.arange(len(view.record_id))
    last_rate = np.divide(activity[rows, last], days[rows, last], out=np.zeros(len(rows), np.float32), where=days[rows, last] > 0)
    aggregate[:, ai["mean_weekly_activity"]] = daily_total * 7.0
    aggregate[:, ai["recent_activity"]] = last_rate * 7.0
    aggregate[:, ai["recent_historical_activity_ratio"]] = np.divide(last_rate, daily_total, out=np.zeros_like(last_rate), where=daily_total > 1e-6)
    for row in rows:
        idxs = np.flatnonzero(valid[row])
        rates = np.divide(activity[row, idxs], days[row, idxs], out=np.zeros(len(idxs), np.float32), where=days[row, idxs] > 0)
        if len(idxs) >= 2:
            aggregate[row, ai["activity_trend"]] = np.polyfit(idxs.astype(np.float32), rates, 1, w=np.sqrt(days[row, idxs]))[0]
        aggregate[row, ai["cumulative_inactive_weeks"]] = float(exposure[row, idxs][activity[row, idxs] <= 0].sum())
        streak = 0.0
        for pos in idxs[::-1]:
            if activity[row, pos] <= 0:
                streak += float(exposure[row, pos])
            else:
                break
        aggregate[row, ai["current_inactivity_streak"]] = streak
    result = UnifiedHybridData(static=view.static.copy(), temporal=temporal, temporal_mask=view.temporal_mask.copy(), lengths=view.lengths.copy(),
                               aggregate=aggregate, aggregate_available=view.aggregate_available.copy(), progress=view.progress.copy(), target=view.target.copy(),
                               record_id=view.record_id.copy(), group_id=view.group_id.copy(), metadata={**view.metadata, "data_variant": "D3_both_safe"},
                               temporal_exposure_normalized=view.temporal_exposure_normalized.copy() if view.temporal_exposure_normalized is not None else None)
    result.validate()
    return result


__all__ = ["load_oulad_static_tables", "validate_oulad_predictor_columns", "build_oulad_array_view", "apply_d3_variant", "oulad_risk_target", "OULAD_ENDPOINTS", "OULAD_FORBIDDEN_PREDICTORS"]
