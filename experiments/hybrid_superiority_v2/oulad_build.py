"""Cutoff-safe OULAD weekly tensors from raw CSVs. Vectorized; no kltn path."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction.data.common import UnifiedHybridData
from src.prediction.data.oulad import (
    OULAD_AGGREGATE_CHANNELS,
    OULAD_TEMPORAL_CHANNELS,
    apply_d3_variant,
    load_oulad_static_tables,
)
from src.prediction.data.oulad_features import (
    ASSESSMENT_RELATED_TYPES,
    CONTENT_TYPES,
    FORUM_TYPES,
    QUIZ_TYPES,
    STATE_FRACTIONS,
    cutoff_day,
    eligible_oulad,
    filter_events_cutoff_safe,
)

from .paths import CACHE_DIR, DATA_ROOT


def build_vle_daily(raw_dir: str | Path, *, chunk_size: int = 1_000_000, cache: Path | None = None) -> pd.DataFrame:
    cache_path = cache or (CACHE_DIR / "vle_daily.parquet")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    raw_path = Path(raw_dir)
    vle_df = pd.read_csv(raw_path / "vle.csv", usecols=["code_module", "code_presentation", "id_site", "activity_type"])
    chunks = []
    for chunk in pd.read_csv(
        raw_path / "studentVle.csv",
        chunksize=chunk_size,
        usecols=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
    ):
        merged = chunk.merge(vle_df, on=["code_module", "code_presentation", "id_site"], how="left")
        merged["activity_type"] = merged["activity_type"].fillna("other")
        chunks.append(
            merged.groupby(
                ["code_module", "code_presentation", "id_student", "date", "id_site", "activity_type"],
                as_index=False,
            )["sum_click"].sum()
        )
        del chunk, merged
    combined = pd.concat(chunks, ignore_index=True)
    del chunks
    daily = combined.groupby(
        ["code_module", "code_presentation", "id_student", "date", "id_site", "activity_type"],
        as_index=False,
    )["sum_click"].sum()
    daily.to_parquet(cache_path, index=False)
    return daily


def _scatter(temporal: np.ndarray, idx: np.ndarray, week: np.ndarray, channel: int, values: np.ndarray, mask: np.ndarray) -> None:
    ok = (week >= 0) & (week < temporal.shape[1]) & (idx >= 0) & (idx < temporal.shape[0])
    idx, week, values = idx[ok], week[ok], values[ok]
    keep = mask[idx, week]
    temporal[idx[keep], week[keep], channel] = values[keep].astype(np.float32)


def build_oulad_cutoff_view(
    base_df: pd.DataFrame,
    vle_daily: pd.DataFrame,
    cutoff_fraction: float,
    raw_dir: str | Path,
) -> tuple[pd.DataFrame, UnifiedHybridData, dict]:
    eligible = eligible_oulad(base_df, cutoff_fraction)
    n = len(eligible)
    if n == 0:
        raise RuntimeError("OULAD_EMPTY_ELIGIBLE")
    eligible = eligible.copy()
    eligible["student_idx"] = np.arange(n)
    obs_start = np.maximum(0, eligible["date_registration"].fillna(0).astype(int).to_numpy())
    cutoff = eligible["cutoff_day"].to_numpy(np.int64)
    eligible["observation_start"] = obs_start
    max_t = int(np.ceil(cutoff.max() / 7.0))
    ci = {name: i for i, name in enumerate(OULAD_TEMPORAL_CHANNELS)}
    week = np.arange(max_t)
    start = week * 7
    end = np.minimum((week + 1) * 7, cutoff[:, None])
    observed = np.maximum(0, end - np.maximum(start, obs_start[:, None]))
    mask = observed > 0
    temporal = np.zeros((n, max_t, len(ci)), dtype=np.float32)
    temporal[:, :, ci["week_exposure_fraction"]] = np.where(mask, observed / 7.0, 0.0).astype(np.float32)

    lookup = eligible[["code_module", "code_presentation", "id_student", "student_idx", "cutoff_day", "observation_start"]]
    events = vle_daily.merge(lookup, on=["code_module", "code_presentation", "id_student"], how="inner")
    events = filter_events_cutoff_safe(events, time_col="date", start_col="observation_start", cutoff_col="cutoff_day")
    if not events.empty:
        events = events.copy()
        events["week"] = (events["date"] // 7).astype(int)
        weekly = events.groupby(["student_idx", "week"], as_index=False).agg(
            sum_click=("sum_click", "sum"),
            active_days=("date", "nunique"),
            unique_sites=("id_site", "nunique"),
            unique_activity_types=("activity_type", "nunique"),
        )
        weekly["activity_intensity_log1p"] = np.log1p(weekly["sum_click"])
        _scatter(temporal, weekly.student_idx.to_numpy(), weekly.week.to_numpy(), ci["activity_intensity_log1p"], weekly.activity_intensity_log1p.to_numpy(), mask)
        _scatter(temporal, weekly.student_idx.to_numpy(), weekly.week.to_numpy(), ci["active_days"], weekly.active_days.to_numpy(), mask)
        _scatter(temporal, weekly.student_idx.to_numpy(), weekly.week.to_numpy(), ci["unique_sites"], weekly.unique_sites.to_numpy(), mask)
        _scatter(temporal, weekly.student_idx.to_numpy(), weekly.week.to_numpy(), ci["unique_activity_types"], weekly.unique_activity_types.to_numpy(), mask)
        for name, types in (
            ("content_activity", CONTENT_TYPES),
            ("forum_activity", FORUM_TYPES),
            ("quiz_activity", QUIZ_TYPES),
            ("assessment_related_activity", ASSESSMENT_RELATED_TYPES),
        ):
            sub = events[events.activity_type.isin(types)]
            if sub.empty:
                continue
            grouped = sub.groupby(["student_idx", "week"], as_index=False)["sum_click"].sum()
            _scatter(temporal, grouped.student_idx.to_numpy(), grouped.week.to_numpy(), ci[name], grouped.sum_click.to_numpy(), mask)

    assessments = pd.read_csv(Path(raw_dir) / "assessments.csv")
    submissions = pd.read_csv(Path(raw_dir) / "studentAssessment.csv")
    submissions = submissions.merge(
        assessments[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment",
        how="inner",
    )
    submissions = submissions.loc[submissions["is_banked"] == 0].copy()
    submissions["is_late"] = (submissions["date_submitted"] > submissions["date"]).astype(np.int8)

    opportunities = lookup.merge(assessments, on=["code_module", "code_presentation"], how="left")
    opportunities = filter_events_cutoff_safe(
        opportunities.dropna(subset=["date"]), time_col="date", start_col="observation_start", cutoff_col="cutoff_day"
    )
    submitted = submissions.merge(lookup, on=["code_module", "code_presentation", "id_student"], how="inner")
    submitted = filter_events_cutoff_safe(submitted, time_col="date_submitted", start_col="observation_start", cutoff_col="cutoff_day")
    if not opportunities.empty:
        due_keys = opportunities[["student_idx", "id_assessment"]].drop_duplicates()
        submitted = submitted.merge(due_keys, on=["student_idx", "id_assessment"], how="inner").drop_duplicates(["student_idx", "id_assessment"])
    if not submitted.empty:
        submitted = submitted.copy()
        submitted["week"] = (submitted.date_submitted // 7).astype(int)
        weekly_sub = submitted.groupby(["student_idx", "week"], as_index=False).agg(
            weekly_submissions=("id_assessment", "nunique"),
            weekly_late_submissions=("is_late", "sum"),
        )
        _scatter(temporal, weekly_sub.student_idx.to_numpy(), weekly_sub.week.to_numpy(), ci["weekly_submissions"], weekly_sub.weekly_submissions.to_numpy(), mask)
        _scatter(temporal, weekly_sub.student_idx.to_numpy(), weekly_sub.week.to_numpy(), ci["weekly_late_submissions"], weekly_sub.weekly_late_submissions.to_numpy(), mask)

    temporal[~mask] = 0.0
    activity_sum = events.groupby("student_idx")["sum_click"].sum() if not events.empty else pd.Series(dtype=float)
    last_activity_day = events.groupby("student_idx")["date"].max() if not events.empty else pd.Series(dtype=float)
    due_count = opportunities.groupby("student_idx")["id_assessment"].nunique() if not opportunities.empty else pd.Series(dtype=float)
    submitted_count = submitted.groupby("student_idx")["id_assessment"].nunique() if not submitted.empty else pd.Series(dtype=float)
    late_count = submitted.groupby("student_idx")["is_late"].sum() if not submitted.empty else pd.Series(dtype=float)

    aggregate = np.zeros((n, len(OULAD_AGGREGATE_CHANNELS)), dtype=np.float32)
    intensity = temporal[:, :, ci["activity_intensity_log1p"]]
    for i in range(n):
        valid = np.flatnonzero(mask[i])
        activity = intensity[i, valid]
        raw_activity = float(activity_sum.get(i, 0.0))
        recent = float(activity[-1]) if len(activity) else 0.0
        mean = float(activity.mean()) if len(activity) else 0.0
        inactive = activity == 0
        streak = int(np.argmax(~inactive[::-1])) if inactive.size and inactive[-1] else 0
        last_day = last_activity_day.get(i) if len(last_activity_day) else None
        days_since = float(cutoff[i] - 1 - last_day) if last_day is not None and np.isfinite(last_day) else float(cutoff[i] - obs_start[i])
        due_n = int(due_count.get(i, 0)) if len(due_count) else 0
        sub_n = int(submitted_count.get(i, 0)) if len(submitted_count) else 0
        late = int(late_count.get(i, 0)) if len(late_count) else 0
        trend = float(np.polyfit(np.arange(len(activity)), activity, 1)[0]) if len(activity) >= 2 else 0.0
        aggregate[i] = [
            raw_activity,
            mean,
            recent,
            recent / max(1e-6, mean),
            trend,
            streak,
            int(inactive.sum()) if inactive.size else 0,
            max(0.0, days_since),
            due_n,
            sub_n,
            sub_n / due_n if due_n else 0.0,
            due_n - sub_n,
            late / sub_n if sub_n else 0.0,
        ]

    view = UnifiedHybridData(
        static=np.zeros((n, 0), np.float32),
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
    audit = {"eligible_records": n, "cutoff_strict": True, "max_t": max_t, "prevalence": float(eligible.target.mean())}
    return eligible, view, audit


def augment_temporal_deltas(view: UnifiedHybridData) -> UnifiedHybridData:
    """Append first-difference of activity intensity. Prefix-only, mask-safe."""
    from src.prediction.data.oulad import OULAD_TEMPORAL_CHANNELS

    idx = list(OULAD_TEMPORAL_CHANNELS).index("activity_intensity_log1p")
    values = view.temporal[:, :, idx]
    delta = np.zeros_like(values)
    delta[:, 1:] = values[:, 1:] - values[:, :-1]
    delta[:, 0] = 0.0
    delta[~view.temporal_mask] = 0.0
    temporal = np.concatenate([view.temporal, delta[:, :, None]], axis=-1)
    metadata = dict(view.metadata)
    metadata["temporal_channels"] = list(metadata.get("temporal_channels", OULAD_TEMPORAL_CHANNELS)) + ["activity_intensity_delta"]
    out = UnifiedHybridData(
        static=view.static,
        temporal=temporal.astype(np.float32),
        temporal_mask=view.temporal_mask,
        lengths=view.lengths,
        aggregate=view.aggregate,
        aggregate_available=view.aggregate_available,
        progress=view.progress,
        target=view.target,
        record_id=view.record_id,
        group_id=view.group_id,
        metadata=metadata,
    )
    out.validate()
    return out
