"""Cutoff-safe OULAD information-state builder used by Phase 4.

This is the active, self-contained feature pipeline. It does not import
experiments.hybrid_vnext. Semantics match the frozen Phase 7/Phase 4 builder:
events satisfy observation_start <= event_time < cutoff.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import OULAD_STATES, canonical_oulad_state
from .common import UnifiedHybridData
from .oulad import (
    OULAD_AGGREGATE_CHANNELS,
    OULAD_FORBIDDEN_PREDICTORS,
    OULAD_TEMPORAL_CHANNELS,
    apply_d3_variant,
    load_oulad_static_tables,
    validate_oulad_predictor_columns,
)

CONTENT_TYPES = {"oucontent", "resource", "page", "url", "glossary", "homepage", "subpage", "dataplus"}
FORUM_TYPES = {"forumng"}
QUIZ_TYPES = {"quiz"}
ASSESSMENT_RELATED_TYPES = {"quiz", "questionnaire", "externalquiz"}

STATE_FRACTIONS = {
    "20pct": 0.20,
    "35pct": 0.35,
    "50pct": 0.50,
    "75pct": 0.75,
    "100pct": 1.0,
}


def cutoff_day(module_presentation_length: float, fraction: float) -> int:
    return max(1, int(math.floor(float(module_presentation_length) * float(fraction))))


def events_strictly_before_cutoff(event_time, observation_start, cutoff) -> bool:
    """Phase 4 rule: observation_start <= event_time < cutoff."""
    return (event_time >= observation_start) and (event_time < cutoff)


def filter_events_cutoff_safe(events: pd.DataFrame, *, time_col: str, start_col: str, cutoff_col: str) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    time = events[time_col].to_numpy()
    start = events[start_col].to_numpy()
    cutoff = events[cutoff_col].to_numpy()
    keep = (time >= start) & (time < cutoff)
    return events.loc[keep].copy()


def eligible_oulad(base_df: pd.DataFrame, cutoff_fraction: float) -> pd.DataFrame:
    """Risk-set at a cutoff. date_unregistration is eligibility-only, never a predictor."""
    base = base_df.copy()
    if "date_unregistration" not in base.columns:
        base["date_unregistration"] = np.nan
    lengths = base.groupby(["code_module", "code_presentation"])["module_presentation_length"].first()
    cutoffs = {key: cutoff_day(value, cutoff_fraction) for key, value in lengths.items()}
    base["cutoff_day"] = [cutoffs[(r.code_module, r.code_presentation)] for r in base.itertuples()]
    reg_ok = base.date_registration.isna() | (base.date_registration <= base.cutoff_day)
    unreg_ok = base.date_unregistration.isna() | (base.date_unregistration > base.cutoff_day)
    return base.loc[reg_ok & unreg_ok].copy().reset_index(drop=True)


def _assessment_tables(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    assessments = pd.read_csv(raw_dir / "assessments.csv")
    submissions = pd.read_csv(raw_dir / "studentAssessment.csv")
    submissions = submissions.merge(
        assessments[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment",
        how="inner",
    )
    submissions = submissions.loc[
        submissions["is_banked"] == 0,
        ["id_assessment", "id_student", "date_submitted", "code_module", "code_presentation", "date"],
    ].copy()
    submissions["is_late"] = (submissions["date_submitted"] > submissions["date"]).astype(np.int8)
    return assessments, submissions


def build_vle_daily(raw_dir: str | Path, *, chunk_size: int = 1_000_000) -> pd.DataFrame:
    """Aggregate raw studentVle + vle metadata to daily-site clicks."""
    raw_path = Path(raw_dir)
    vle_df = pd.read_csv(raw_path / "vle.csv")
    key_map = dict(
        zip(
            zip(vle_df["code_module"], vle_df["code_presentation"], vle_df["id_site"]),
            vle_df["activity_type"],
        )
    )
    chunks = []
    for chunk in pd.read_csv(raw_path / "studentVle.csv", chunksize=chunk_size):
        keys = list(zip(chunk["code_module"], chunk["code_presentation"], chunk["id_site"]))
        chunk = chunk.copy()
        chunk["activity_type"] = [key_map.get(k, "other") for k in keys]
        chunks.append(
            chunk.groupby(
                ["code_module", "code_presentation", "id_student", "date", "id_site", "activity_type"],
                as_index=False,
            )["sum_click"].sum()
        )
    combined = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=["code_module", "code_presentation", "id_student", "date", "id_site", "activity_type", "sum_click"]
    )
    return combined.groupby(
        ["code_module", "code_presentation", "id_student", "date", "id_site", "activity_type"],
        as_index=False,
    )["sum_click"].sum()


def build_oulad_cutoff_view(
    base_df: pd.DataFrame,
    vle_daily: pd.DataFrame,
    cutoff_fraction: float,
    raw_dir: str | Path,
    static: np.ndarray | None = None,
) -> tuple[pd.DataFrame, UnifiedHybridData, dict]:
    """Phase-4 weekly tensor + aggregates. Events use start <= t < cutoff."""
    eligible = eligible_oulad(base_df, cutoff_fraction)
    eligible["student_idx"] = np.arange(len(eligible))
    eligible["observation_start"] = np.maximum(0, eligible["date_registration"].fillna(0).astype(int))
    max_t = int(np.ceil(eligible["cutoff_day"].max() / 7)) if len(eligible) else 1
    n = len(eligible)
    ci = {name: i for i, name in enumerate(OULAD_TEMPORAL_CHANNELS)}
    temporal = np.zeros((n, max_t, len(ci)), dtype=np.float32)
    mask = np.zeros((n, max_t), dtype=bool)
    for row in eligible.itertuples():
        for week in range(max_t):
            start, end = week * 7, min((week + 1) * 7, row.cutoff_day)
            observed = max(0, end - max(start, row.observation_start))
            if observed:
                mask[row.student_idx, week] = True
                temporal[row.student_idx, week, ci["week_exposure_fraction"]] = observed / 7.0
    lookup = eligible[["code_module", "code_presentation", "id_student", "student_idx", "cutoff_day", "observation_start"]]
    events = vle_daily.merge(lookup, on=["code_module", "code_presentation", "id_student"], how="inner")
    events = filter_events_cutoff_safe(events, time_col="date", start_col="observation_start", cutoff_col="cutoff_day")
    events["week"] = (events.date // 7).astype(int)

    def put(frame: pd.DataFrame, column: str, value_col: str) -> None:
        if frame.empty:
            return
        for item in frame.itertuples():
            week = int(item.week)
            if 0 <= week < max_t and mask[item.student_idx, week]:
                temporal[item.student_idx, week, ci[column]] = getattr(item, value_col)

    if not events.empty:
        weekly = events.groupby(["student_idx", "week"]).agg(
            sum_click=("sum_click", "sum"),
            active_days=("date", "nunique"),
            unique_sites=("id_site", "nunique"),
            unique_activity_types=("activity_type", "nunique"),
        ).reset_index()
        weekly["activity_intensity_log1p"] = np.log1p(weekly.sum_click)
        for name, value in (
            ("activity_intensity_log1p", "activity_intensity_log1p"),
            ("active_days", "active_days"),
            ("unique_sites", "unique_sites"),
            ("unique_activity_types", "unique_activity_types"),
        ):
            put(weekly, name, value)
        for name, types in (
            ("content_activity", CONTENT_TYPES),
            ("forum_activity", FORUM_TYPES),
            ("quiz_activity", QUIZ_TYPES),
            ("assessment_related_activity", ASSESSMENT_RELATED_TYPES),
        ):
            grouped = events[events.activity_type.isin(types)].groupby(["student_idx", "week"], as_index=False)["sum_click"].sum()
            put(grouped, name, "sum_click")

    assessments, submissions = _assessment_tables(raw_dir)
    opportunities = eligible[["code_module", "code_presentation", "id_student", "student_idx", "cutoff_day", "observation_start"]].merge(
        assessments, on=["code_module", "code_presentation"], how="left"
    )
    opportunities = filter_events_cutoff_safe(opportunities.dropna(subset=["date"]), time_col="date", start_col="observation_start", cutoff_col="cutoff_day")
    submitted = submissions.merge(lookup, on=["code_module", "code_presentation", "id_student"], how="inner")
    submitted = filter_events_cutoff_safe(submitted, time_col="date_submitted", start_col="observation_start", cutoff_col="cutoff_day")
    if not opportunities.empty:
        due_keys = opportunities[["student_idx", "id_assessment"]].drop_duplicates()
        submitted = submitted.merge(due_keys, on=["student_idx", "id_assessment"], how="inner").drop_duplicates(["student_idx", "id_assessment"])
    if not submitted.empty:
        submitted["week"] = (submitted.date_submitted // 7).astype(int)
        weekly_sub = submitted.groupby(["student_idx", "week"]).agg(
            weekly_submissions=("id_assessment", "nunique"),
            weekly_late_submissions=("is_late", "sum"),
        ).reset_index()
        put(weekly_sub, "weekly_submissions", "weekly_submissions")
        put(weekly_sub, "weekly_late_submissions", "weekly_late_submissions")

    activity_sum = events.groupby("student_idx")["sum_click"].sum().to_dict() if not events.empty else {}
    last_activity_day = events.groupby("student_idx")["date"].max().to_dict() if not events.empty else {}
    due_count = opportunities.groupby("student_idx")["id_assessment"].nunique().to_dict() if not opportunities.empty else {}
    submitted_count = submitted.groupby("student_idx")["id_assessment"].nunique().to_dict() if not submitted.empty else {}
    late_count = submitted.groupby("student_idx")["is_late"].sum().to_dict() if not submitted.empty else {}
    aggregate = np.zeros((n, len(OULAD_AGGREGATE_CHANNELS)), dtype=np.float32)
    for row in eligible.itertuples():
        valid = np.flatnonzero(mask[row.student_idx])
        activity = temporal[row.student_idx, valid, ci["activity_intensity_log1p"]]
        raw_activity = float(activity_sum.get(row.student_idx, 0.0))
        recent = float(activity[-1]) if len(activity) else 0.0
        mean = float(activity.mean()) if len(activity) else 0.0
        inactive = activity == 0
        streak = int(np.argmax(~inactive[::-1])) if inactive.size and inactive[-1] else 0
        last_day = last_activity_day.get(row.student_idx)
        days_since = float(row.cutoff_day - 1 - last_day) if last_day is not None else float(row.cutoff_day - row.observation_start)
        due_n = int(due_count.get(row.student_idx, 0))
        sub_n = int(submitted_count.get(row.student_idx, 0))
        late = int(late_count.get(row.student_idx, 0))
        trend = float(np.polyfit(np.arange(len(activity)), activity, 1)[0]) if len(activity) >= 2 else 0.0
        aggregate[row.student_idx] = [
            raw_activity, mean, recent, recent / max(1e-6, mean), trend, streak, int(inactive.sum()),
            max(0.0, days_since), due_n, sub_n, sub_n / due_n if due_n else 0.0, due_n - sub_n, late / sub_n if sub_n else 0.0,
        ]
    static_arr = np.zeros((n, 0), np.float32) if static is None else np.asarray(static, np.float32)
    view = UnifiedHybridData(
        static=static_arr,
        temporal=temporal,
        temporal_mask=mask,
        lengths=mask.sum(1).astype(np.int64),
        aggregate=aggregate,
        aggregate_available=np.ones(n, np.int8),
        progress=np.full(n, cutoff_fraction, np.float32),
        target=eligible.target.to_numpy(np.int64),
        record_id=eligible.record_id.to_numpy(str),
        group_id=eligible.group_id.to_numpy(str),
        metadata={
            "dataset": "oulad",
            "cutoff_fraction": cutoff_fraction,
            "temporal_channels": list(OULAD_TEMPORAL_CHANNELS),
            "aggregate_channels": list(OULAD_AGGREGATE_CHANNELS),
            "cutoff_rule": "observation_start <= event_time < cutoff",
            "separate_model": False,
        },
    )
    view.validate()
    audit = {"eligible_records": n, "cutoff_strict": True, "cutoff_rule": "t < cutoff"}
    return eligible, view, audit


def build_oulad_information_state(raw_dir: str | Path, state: str, *, vle_daily: pd.DataFrame | None = None, apply_d3: bool = True) -> UnifiedHybridData:
    """Raw tables → one information state of the same OULAD Hybrid."""
    state = canonical_oulad_state(state)
    if state not in STATE_FRACTIONS:
        raise ValueError(state)
    _, _, base = load_oulad_static_tables(raw_dir)
    daily = vle_daily if vle_daily is not None else build_vle_daily(raw_dir)
    _eligible, view, _audit = build_oulad_cutoff_view(base, daily, STATE_FRACTIONS[state], raw_dir)
    if apply_d3:
        view = apply_d3_variant(view)
    return view


def assert_predictor_contract(columns) -> None:
    validate_oulad_predictor_columns(columns)
    leaked = [c for c in columns if c.lower() in {x.lower() for x in OULAD_FORBIDDEN_PREDICTORS} or c.lower() == "g3"]
    if leaked:
        raise ValueError(f"Forbidden OULAD predictor columns: {leaked}")


__all__ = [
    "STATE_FRACTIONS",
    "CONTENT_TYPES",
    "events_strictly_before_cutoff",
    "filter_events_cutoff_safe",
    "eligible_oulad",
    "build_vle_daily",
    "build_oulad_cutoff_view",
    "build_oulad_information_state",
    "assert_predictor_contract",
]
