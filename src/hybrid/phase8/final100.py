"""Explicit OULAD FINAL-100 endpoint construction for Phase8 development.

This module intentionally does not call ``phase7_eligible_oulad``: that helper
defines an early-warning risk set and excludes students unregistered before a
cutoff.  FINAL-100 must retain the complete student-presentation population
while ending each student's observable history at the legitimate observation
boundary.
"""
from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.hybrid.data.oulad import (
    ASSESSMENT_RELATED_TYPES,
    CONTENT_TYPES,
    FORUM_TYPES,
    QUIZ_TYPES,
)
from src.hybrid.phase7.contracts import UnifiedHybridData
from src.hybrid.phase7.data import (
    OULAD_PHASE7_AGGREGATE_CHANNELS,
    OULAD_PHASE7_TEMPORAL_CHANNELS,
    _assessment_tables,
)

FINAL_STAGE = "FINAL-100"


def save_final100_artifact(view: UnifiedHybridData, audit: dict, output_dir: str | Path) -> None:
    """Persist a deterministic additive FINAL-100 data contract."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    save_unified_view(view, out)
    (out / "metadata.json").write_text(json.dumps(view.metadata, indent=2), encoding="utf-8")
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def load_final100_artifact(output_dir: str | Path) -> UnifiedHybridData:
    return load_unified_view(output_dir)


def save_unified_view(view: UnifiedHybridData, output_dir: str | Path) -> None:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "view.npz", static=view.static, temporal=view.temporal,
                        temporal_mask=view.temporal_mask, lengths=view.lengths,
                        aggregate=view.aggregate, aggregate_available=view.aggregate_available,
                        progress=view.progress, target=view.target,
                        record_id=view.record_id.astype(str), group_id=view.group_id.astype(str))
    (out / "metadata.json").write_text(json.dumps(view.metadata, indent=2), encoding="utf-8")


def load_unified_view(output_dir: str | Path) -> UnifiedHybridData:
    out = Path(output_dir); data = np.load(out / "view.npz", allow_pickle=False)
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    view = UnifiedHybridData(static=data["static"], temporal=data["temporal"], temporal_mask=data["temporal_mask"],
                             lengths=data["lengths"], aggregate=data["aggregate"], aggregate_available=data["aggregate_available"],
                             progress=data["progress"], target=data["target"], record_id=data["record_id"].astype(str),
                             group_id=data["group_id"].astype(str), metadata=metadata)
    view.validate(); return view


def _put(temporal: np.ndarray, mask: np.ndarray, frame: pd.DataFrame, channel: str, value: str, index: dict[str, int]) -> None:
    for row in frame.itertuples():
        if 0 <= row.week < temporal.shape[1] and mask[row.student_idx, row.week]:
            temporal[row.student_idx, row.week, index[channel]] = getattr(row, value)


def _observation_end(base: pd.DataFrame) -> np.ndarray:
    """Return an availability boundary, never a predictor.

    A recorded unregistration ends observable history when present; otherwise
    history runs to the scheduled presentation end.  Negative/unusable dates
    produce an empty but valid history rather than synthetic post-outcome
    zeros.
    """
    course_end = base.module_presentation_length.to_numpy(np.int64)
    unregistered = base.date_unregistration.where(base.date_unregistration.notna(), pd.Series(course_end, index=base.index)).to_numpy(np.float64)
    unregistered = np.where(np.isfinite(unregistered), unregistered, course_end)
    return np.clip(np.minimum(course_end, unregistered).astype(np.int64), 0, course_end)


def build_oulad_final100_view(
    base_df: pd.DataFrame,
    vle_daily: pd.DataFrame,
    raw_dir: str | Path = "data/raw",
) -> tuple[pd.DataFrame, UnifiedHybridData, dict]:
    """Build the complete, registration-safe OULAD FINAL-100 view.

    The returned frame retains all target classes and IDs for auditability;
    only the arrays in ``UnifiedHybridData`` are used as model inputs.  No
    final-result, score, or unregistration column is copied into predictors.
    """
    base = base_df.copy().reset_index(drop=True)
    vle_daily = vle_daily.copy()
    required = {"record_id", "group_id", "target", "final_result", "date_registration", "date_unregistration", "module_presentation_length"}
    missing = required.difference(base.columns)
    if missing:
        raise ValueError(f"FINAL-100 missing required columns: {sorted(missing)}")
    key_columns = ("code_module", "code_presentation", "id_student")
    for frame in (base, vle_daily):
        for column in key_columns:
            if column in frame:
                frame[column] = frame[column].astype(str)
    base["student_idx"] = np.arange(len(base), dtype=np.int64)
    base["observation_start"] = np.maximum(0, base.date_registration.fillna(0).astype(np.int64))
    base["observation_end"] = _observation_end(base)
    max_t = int(math.ceil(base.module_presentation_length.max() / 7.0))
    ci = {name: i for i, name in enumerate(OULAD_PHASE7_TEMPORAL_CHANNELS)}
    n = len(base)
    temporal = np.zeros((n, max_t, len(ci)), dtype=np.float32)
    mask = np.zeros((n, max_t), dtype=bool)
    for row in base.itertuples():
        for week in range(max_t):
            start, end = week * 7, min((week + 1) * 7, int(row.observation_end))
            observed = max(0, end - max(start, int(row.observation_start)))
            if observed:
                mask[row.student_idx, week] = True
                temporal[row.student_idx, week, ci["week_exposure_fraction"]] = observed / 7.0

    lookup = base[["code_module", "code_presentation", "id_student", "student_idx", "observation_start", "observation_end"]]
    events = vle_daily.merge(lookup, on=["code_module", "code_presentation", "id_student"], how="inner")
    events = events[(events.date >= events.observation_start) & (events.date < events.observation_end)].copy()
    events["week"] = (events.date // 7).astype(int)
    if not events.empty:
        weekly = events.groupby(["student_idx", "week"]).agg(
            sum_click=("sum_click", "sum"), active_days=("date", "nunique"),
            unique_sites=("id_site", "nunique"), unique_activity_types=("activity_type", "nunique"),
        ).reset_index()
        weekly["activity_intensity_log1p"] = np.log1p(weekly.sum_click)
        for name in ("activity_intensity_log1p", "active_days", "unique_sites", "unique_activity_types"):
            _put(temporal, mask, weekly, name, name, ci)
        for name, types in (("content_activity", CONTENT_TYPES), ("forum_activity", FORUM_TYPES),
                            ("quiz_activity", QUIZ_TYPES), ("assessment_related_activity", ASSESSMENT_RELATED_TYPES)):
            grouped = events[events.activity_type.isin(types)].groupby(["student_idx", "week"], as_index=False)["sum_click"].sum()
            _put(temporal, mask, grouped, name, "sum_click", ci)

    assessments, submissions = _assessment_tables(str(raw_dir))
    for frame in (assessments, submissions):
        for column in key_columns:
            if column in frame:
                frame[column] = frame[column].astype(str)
    opportunities = base[["code_module", "code_presentation", "id_student", "student_idx", "observation_start", "observation_end"]].merge(
        assessments, on=["code_module", "code_presentation"], how="left"
    )
    opportunities = opportunities[(opportunities.date >= opportunities.observation_start) & (opportunities.date < opportunities.observation_end)]
    submitted = submissions.merge(lookup, on=["code_module", "code_presentation", "id_student"], how="inner")
    submitted = submitted[(submitted.date_submitted >= submitted.observation_start) & (submitted.date_submitted < submitted.observation_end)]
    due_keys = opportunities[["student_idx", "id_assessment"]].drop_duplicates()
    submitted = submitted.merge(due_keys, on=["student_idx", "id_assessment"], how="inner").drop_duplicates(["student_idx", "id_assessment"])
    submitted["week"] = (submitted.date_submitted // 7).astype(int)
    if not submitted.empty:
        weekly_sub = submitted.groupby(["student_idx", "week"]).agg(
            weekly_submissions=("id_assessment", "nunique"), weekly_late_submissions=("is_late", "sum")
        ).reset_index()
        _put(temporal, mask, weekly_sub, "weekly_submissions", "weekly_submissions", ci)
        _put(temporal, mask, weekly_sub, "weekly_late_submissions", "weekly_late_submissions", ci)

    activity_sum = events.groupby("student_idx")["sum_click"].sum().to_dict() if not events.empty else {}
    last_activity_day = events.groupby("student_idx")["date"].max().to_dict() if not events.empty else {}
    due_count = opportunities.groupby("student_idx")["id_assessment"].nunique().to_dict() if not opportunities.empty else {}
    submitted_count = submitted.groupby("student_idx")["id_assessment"].nunique().to_dict() if not submitted.empty else {}
    late_count = submitted.groupby("student_idx")["is_late"].sum().to_dict() if not submitted.empty else {}
    aggregate = np.zeros((n, len(OULAD_PHASE7_AGGREGATE_CHANNELS)), dtype=np.float32)
    ai = ci["activity_intensity_log1p"]
    for row in base.itertuples():
        valid = np.flatnonzero(mask[row.student_idx])
        activity = temporal[row.student_idx, valid, ai]
        raw_activity = float(activity_sum.get(row.student_idx, 0.0))
        recent = float(activity[-1]) if len(activity) else 0.0
        mean = float(activity.mean()) if len(activity) else 0.0
        inactive = activity == 0
        streak = int(np.count_nonzero(inactive[::-1].cumprod())) if inactive.size and inactive[-1] else 0
        last_day = last_activity_day.get(row.student_idx)
        days_since = float(max(0, int(row.observation_end) - 1 - last_day)) if last_day is not None else float(max(0, int(row.observation_end) - int(row.observation_start)))
        due_n, sub_n = int(due_count.get(row.student_idx, 0)), int(submitted_count.get(row.student_idx, 0))
        late = int(late_count.get(row.student_idx, 0))
        trend = float(np.polyfit(np.arange(len(activity)), activity, 1)[0]) if len(activity) >= 2 else 0.0
        aggregate[row.student_idx] = [raw_activity, mean, recent, recent / max(1e-6, mean), trend, streak,
                                      int(inactive.sum()), max(0.0, days_since), due_n, sub_n,
                                      sub_n / due_n if due_n else 0.0, due_n - sub_n, late / sub_n if sub_n else 0.0]

    view = UnifiedHybridData(
        static=np.zeros((n, 0), np.float32), temporal=temporal, temporal_mask=mask,
        lengths=mask.sum(1).astype(np.int64), aggregate=aggregate,
        aggregate_available=np.ones(n, np.int8), progress=np.ones(n, np.float32),
        target=base.target.to_numpy(np.int64), record_id=base.record_id.to_numpy(str),
        group_id=base.group_id.to_numpy(str), metadata={
            "domain": "oulad", "stage": FINAL_STAGE, "cutoff_fraction": 1.0,
            "final_endpoint": True, "temporal_channels": OULAD_PHASE7_TEMPORAL_CHANNELS,
            "aggregate_channels": OULAD_PHASE7_AGGREGATE_CHANNELS,
            "observation_boundary": "registration to min(course_end, recorded_unregistration); no post-boundary padding",
            "forbidden_predictors": ["final_result", "target", "score", "date_unregistration"],
        }
    )
    view.validate()
    exposure = temporal[:, :, ci["week_exposure_fraction"]][mask]
    labels = base.final_result.astype(str)
    sequence_frame = pd.DataFrame({"label": labels, "length": view.lengths})
    by_class = {
        str(label): {key: (float(value) if np.isfinite(value) else None) for key, value in values.items()}
        for label, values in sequence_frame.groupby("label").length.agg(["mean", "std", "min", "median", "max"]).to_dict("index").items()
    }
    short = sequence_frame.length <= 20
    short_withdrawn_rate = float((sequence_frame.loc[short, "label"] == "Withdrawn").mean()) if short.any() else 0.0
    audit = {
        "endpoint": FINAL_STAGE, "population": int(n), "base_population": int(len(base)),
        "row_loss": int(len(base) - n), "target_risk_n": int(view.target.sum()),
        "target_risk_prevalence": float(view.target.mean()),
        "class_counts": {str(k): int(v) for k, v in labels.value_counts().to_dict().items()},
        "sequence_length": {k: float(v) for k, v in {
            "mean": view.lengths.mean(), "std": view.lengths.std(), "min": view.lengths.min(),
            "median": np.median(view.lengths), "max": view.lengths.max(),
        }.items()},
        "sequence_length_by_class": by_class,
        "sequence_length_shortcut_diagnostic": {"short_history_definition": "length <= 20 weeks", "withdrawn_rate_in_short_history": short_withdrawn_rate, "flagged_as_shortcut_risk": bool(short_withdrawn_rate >= 0.95)},
        "exposure_fraction": {"n": int(len(exposure)), "mean": float(exposure.mean()) if len(exposure) else 0.0,
                              "min": float(exposure.min()) if len(exposure) else 0.0, "max": float(exposure.max()) if len(exposure) else 0.0},
        "missingness": {"date_registration": int(base.date_registration.isna().sum()), "date_unregistration": int(base.date_unregistration.isna().sum())},
        "target_classes": ["Fail", "Withdrawn", "Pass", "Distinction"],
        "forbidden_predictors": ["final_result", "target", "score", "date_unregistration"],
        "outer_test_used": False,
    }
    return base, view, audit
