"""Cutoff-safe Phase 7 feature construction, independent of frozen loaders."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .contracts import UnifiedHybridData
from src.hybrid.data.oulad import ASSESSMENT_RELATED_TYPES, CONTENT_TYPES, FORUM_TYPES, QUIZ_TYPES

UCI_PHASE7_AGGREGATE_CHANNELS = ["latest_grade", "running_mean", "observed_grade_fraction", "delta_grade", "delta_available"]
OULAD_PHASE7_TEMPORAL_CHANNELS = [
    "activity_intensity_log1p", "active_days", "unique_sites", "unique_activity_types",
    "content_activity", "forum_activity", "quiz_activity", "assessment_related_activity",
    "weekly_submissions", "weekly_late_submissions", "week_exposure_fraction",
]
OULAD_PHASE7_AGGREGATE_CHANNELS = [
    "cumulative_activity", "mean_weekly_activity", "recent_activity", "recent_historical_activity_ratio",
    "activity_trend", "current_inactivity_streak", "cumulative_inactive_weeks", "days_since_last_activity",
    "assessments_due_to_date", "submitted_due_to_date", "completion_rate", "missed_due_count", "late_submission_rate",
]


def phase7_eligible_oulad(base_df: pd.DataFrame, cutoff_fraction: float) -> pd.DataFrame:
    """Historical operational risk-set: unregistration is eligibility-only, never a feature."""
    base = base_df.copy()
    if "date_unregistration" not in base:
        base["date_unregistration"] = np.nan  # synthetic/adapter compatibility: no withdrawal recorded
    lengths = base.groupby(["code_module", "code_presentation"])["module_presentation_length"].first()
    cutoffs = {key: max(1, int(math.floor(value * cutoff_fraction))) for key, value in lengths.items()}
    base["cutoff_day"] = [cutoffs[(r.code_module, r.code_presentation)] for r in base.itertuples()]
    reg_ok = base.date_registration.isna() | (base.date_registration <= base.cutoff_day)
    unreg_ok = base.date_unregistration.isna() | (base.date_unregistration > base.cutoff_day)
    return base.loc[reg_ok & unreg_ok].copy().reset_index(drop=True)


def _empty_static(n: int, static: np.ndarray | None) -> np.ndarray:
    return np.zeros((n, 0), dtype=np.float32) if static is None else np.asarray(static, dtype=np.float32)


def build_uci_phase7_view(uci_df: pd.DataFrame, stage: str, static: np.ndarray | None = None) -> UnifiedHybridData:
    """Build UCI S0/S1/S2 without converting grade summaries into timesteps."""
    if stage not in {"S0", "S1", "S2"}:
        raise ValueError("stage must be S0, S1, or S2")
    n = len(uci_df)
    temporal = np.zeros((n, 2, 1), dtype=np.float32)
    mask = np.zeros((n, 2), dtype=bool)
    aggregate = np.zeros((n, len(UCI_PHASE7_AGGREGATE_CHANNELS)), dtype=np.float32)
    available = np.zeros(n, dtype=np.int8)
    g1 = uci_df["G1"].to_numpy(np.float32) / 20.0
    g2 = uci_df["G2"].to_numpy(np.float32) / 20.0
    if stage in {"S1", "S2"}:
        temporal[:, 0, 0] = g1; mask[:, 0] = True
        aggregate[:, 0] = g1; aggregate[:, 1] = g1; aggregate[:, 2] = 0.5
        available[:] = 1
    if stage == "S2":
        temporal[:, 1, 0] = g2; mask[:, 1] = True
        aggregate[:, 0] = g2; aggregate[:, 1] = (g1 + g2) / 2.0
        aggregate[:, 2] = 1.0; aggregate[:, 3] = g2 - g1; aggregate[:, 4] = 1.0
    view = UnifiedHybridData(
        static=_empty_static(n, static), temporal=temporal, temporal_mask=mask, lengths=mask.sum(1).astype(np.int64),
        aggregate=aggregate, aggregate_available=available, progress=np.full(n, {"S0": 0., "S1": .5, "S2": 1.}[stage], np.float32),
        target=uci_df["target"].to_numpy(np.int64), record_id=uci_df["record_id"].to_numpy(str),
        group_id=uci_df["global_student_group"].to_numpy(str), metadata={"domain": "uci", "stage": stage, "aggregate_channels": UCI_PHASE7_AGGREGATE_CHANNELS},
    )
    view.validate(); return view


def _assessment_tables(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    assessments = pd.read_csv(f"{raw_dir}/assessments.csv")
    submissions = pd.read_csv(f"{raw_dir}/studentAssessment.csv")
    submissions = submissions.merge(assessments[["id_assessment", "code_module", "code_presentation", "date"]], on="id_assessment", how="inner")
    # is_banked is an eligibility/status flag, not a predictor. score is intentionally discarded.
    submissions = submissions.loc[submissions["is_banked"] == 0, ["id_assessment", "id_student", "date_submitted", "code_module", "code_presentation", "date"]].copy()
    submissions["is_late"] = (submissions["date_submitted"] > submissions["date"]).astype(np.int8)
    return assessments, submissions


def build_oulad_phase7_view(base_df: pd.DataFrame, vle_daily: pd.DataFrame, cutoff_fraction: float, raw_dir: str = "data/raw", static: np.ndarray | None = None) -> tuple[pd.DataFrame, UnifiedHybridData, dict]:
    """Build compact weekly OULAD histories with registration-aware exposure.

    A week is valid only for its observed overlap with [registration, cutoff).
    Thus pre-registration weeks are padding, never observed zero activity.
    """
    eligible = phase7_eligible_oulad(base_df, cutoff_fraction)
    eligible["student_idx"] = np.arange(len(eligible))
    # Course start is day 0. Missing registration means course start.
    eligible["observation_start"] = np.maximum(0, eligible["date_registration"].fillna(0).astype(int))
    max_t = int(np.ceil(eligible["cutoff_day"].max() / 7))
    n = len(eligible); ci = {name: i for i, name in enumerate(OULAD_PHASE7_TEMPORAL_CHANNELS)}
    temporal = np.zeros((n, max_t, len(ci)), dtype=np.float32); mask = np.zeros((n, max_t), dtype=bool)
    mask_counts = {"pre_registration": 0, "post_cutoff": 0, "padding": 0, "partial_exposure": 0}
    for r in eligible.itertuples():
        for w in range(max_t):
            start, end = w * 7, min((w + 1) * 7, r.cutoff_day)
            observed = max(0, end - max(start, r.observation_start))
            if observed:
                mask[r.student_idx, w] = True
                temporal[r.student_idx, w, ci["week_exposure_fraction"]] = observed / 7.0
                if observed < 7: mask_counts["partial_exposure"] += 1
            elif start >= r.cutoff_day:
                mask_counts["padding"] += 1
            elif end <= r.observation_start:
                mask_counts["pre_registration"] += 1
            else:
                # The only remaining case is a zero-width final interval; it is not observable.
                mask_counts["post_cutoff"] += 1
    lookup = eligible[["code_module", "code_presentation", "id_student", "student_idx", "cutoff_day", "observation_start"]]
    events = vle_daily.merge(lookup, on=["code_module", "code_presentation", "id_student"], how="inner")
    events = events[(events.date >= events.observation_start) & (events.date < events.cutoff_day)].copy()
    events["week"] = (events.date // 7).astype(int)
    if not events.empty:
        def put(frame: pd.DataFrame, column: str, values: str) -> None:
            for r in frame.itertuples():
                if mask[r.student_idx, r.week]: temporal[r.student_idx, r.week, ci[column]] = getattr(r, values)
        weekly = events.groupby(["student_idx", "week"]).agg(sum_click=("sum_click", "sum"), active_days=("date", "nunique"), unique_sites=("id_site", "nunique"), unique_activity_types=("activity_type", "nunique")).reset_index()
        weekly["activity_intensity_log1p"] = np.log1p(weekly.sum_click)
        for name, value in (("activity_intensity_log1p", "activity_intensity_log1p"), ("active_days", "active_days"), ("unique_sites", "unique_sites"), ("unique_activity_types", "unique_activity_types")): put(weekly, name, value)
        for name, types in (("content_activity", CONTENT_TYPES), ("forum_activity", FORUM_TYPES), ("quiz_activity", QUIZ_TYPES), ("assessment_related_activity", ASSESSMENT_RELATED_TYPES)):
            grouped = events[events.activity_type.isin(types)].groupby(["student_idx", "week"], as_index=False)["sum_click"].sum()
            put(grouped, name, "sum_click")
    assessments, submissions = _assessment_tables(raw_dir)
    opportunities = eligible[["code_module", "code_presentation", "id_student", "student_idx", "cutoff_day", "observation_start"]].merge(assessments, on=["code_module", "code_presentation"], how="left")
    opportunities = opportunities[(opportunities.date >= opportunities.observation_start) & (opportunities.date < opportunities.cutoff_day)]
    submitted = submissions.merge(lookup, on=["code_module", "code_presentation", "id_student"], how="inner")
    submitted = submitted[(submitted.date_submitted >= submitted.observation_start) & (submitted.date_submitted < submitted.cutoff_day)]
    due_keys = opportunities[["student_idx", "id_assessment"]].drop_duplicates()
    submitted = submitted.merge(due_keys, on=["student_idx", "id_assessment"], how="inner").drop_duplicates(["student_idx", "id_assessment"])
    submitted["week"] = (submitted.date_submitted // 7).astype(int)
    if not submitted.empty:
        weekly_sub = submitted.groupby(["student_idx", "week"]).agg(weekly_submissions=("id_assessment", "nunique"), weekly_late_submissions=("is_late", "sum")).reset_index()
        put(weekly_sub, "weekly_submissions", "weekly_submissions"); put(weekly_sub, "weekly_late_submissions", "weekly_late_submissions")
    activity_sum = events.groupby("student_idx")["sum_click"].sum().to_dict() if not events.empty else {}
    last_activity_day = events.groupby("student_idx")["date"].max().to_dict() if not events.empty else {}
    due_count = opportunities.groupby("student_idx")["id_assessment"].nunique().to_dict() if not opportunities.empty else {}
    submitted_count = submitted.groupby("student_idx")["id_assessment"].nunique().to_dict() if not submitted.empty else {}
    late_count = submitted.groupby("student_idx")["is_late"].sum().to_dict() if not submitted.empty else {}
    aggregate = np.zeros((n, len(OULAD_PHASE7_AGGREGATE_CHANNELS)), dtype=np.float32)
    for r in eligible.itertuples():
        valid = np.flatnonzero(mask[r.student_idx]); activity = temporal[r.student_idx, valid, ci["activity_intensity_log1p"]]
        raw_activity = float(activity_sum.get(r.student_idx, 0.0))
        recent = float(activity[-1]) if len(activity) else 0.; mean = float(activity.mean()) if len(activity) else 0.
        inactive = activity == 0; streak = int(np.argmax(~inactive[::-1])) if inactive.size and inactive[-1] else 0
        last_day = last_activity_day.get(r.student_idx)
        days_since = float(r.cutoff_day - 1 - last_day) if last_day is not None else float(r.cutoff_day - r.observation_start)
        due_n, sub_n = int(due_count.get(r.student_idx, 0)), int(submitted_count.get(r.student_idx, 0)); late = int(late_count.get(r.student_idx, 0))
        trend = float(np.polyfit(np.arange(len(activity)), activity, 1)[0]) if len(activity) >= 2 else 0.
        aggregate[r.student_idx] = [raw_activity, mean, recent, recent / max(1e-6, mean), trend, streak, int(inactive.sum()), max(0., days_since), due_n, sub_n, sub_n / due_n if due_n else 0., due_n - sub_n, late / sub_n if sub_n else 0.]
    normalized = temporal.copy(); exposure = temporal[:, :, ci["week_exposure_fraction"]]
    for name in OULAD_PHASE7_TEMPORAL_CHANNELS[:-1]:
        normalized[:, :, ci[name]] = np.divide(temporal[:, :, ci[name]], exposure, out=np.zeros_like(exposure), where=exposure > 0)
    view = UnifiedHybridData(static=_empty_static(n, static), temporal=temporal, temporal_mask=mask, lengths=mask.sum(1).astype(np.int64), aggregate=aggregate, aggregate_available=np.ones(n, np.int8), progress=np.full(n, cutoff_fraction, np.float32), target=eligible.target.to_numpy(np.int64), record_id=eligible.record_id.to_numpy(str), group_id=eligible.group_id.to_numpy(str), metadata={"domain": "oulad", "cutoff_fraction": cutoff_fraction, "temporal_channels": OULAD_PHASE7_TEMPORAL_CHANNELS, "aggregate_channels": OULAD_PHASE7_AGGREGATE_CHANNELS, "exposure_normalized_variant": "prepared_for_later_ablation_not_model_input"}, temporal_exposure_normalized=normalized)
    view.validate()
    audit = {"eligible_records": n, "risk_prevalence": float(view.target.mean()), "mask_counts": mask_counts, "assessment_opportunity_source": "assessments.csv deadline metadata", "cutoff_strict": True}
    return eligible, view, audit


def build_phase7_baseline_frame(static_frame: pd.DataFrame, view: UnifiedHybridData) -> pd.DataFrame:
    """Fair tabular view: allowed static + aggregate + deterministic temporal summaries only."""
    if len(static_frame) != len(view.record_id):
        raise ValueError("static_frame must align one-to-one with Phase7 view")
    static = static_frame.copy(); static["record_id"] = static.record_id.astype(str)
    static = static.set_index("record_id").loc[view.record_id.astype(str)].reset_index()
    out = static.drop(columns=[c for c in ("target", "group_id", "final_result", "date_unregistration", "score") if c in static], errors="ignore")
    for i, name in enumerate(view.metadata.get("aggregate_channels", [])):
        out[f"aggregate__{name}"] = view.aggregate[:, i]
    for i, name in enumerate(view.metadata.get("temporal_channels", ["raw_grade_normalized"])):
        values = view.temporal[:, :, i]; valid = view.temporal_mask
        out[f"temporal__{name}__last"] = np.asarray([row[np.flatnonzero(m)[-1]] if m.any() else 0. for row, m in zip(values, valid)], np.float32)
        out[f"temporal__{name}__mean"] = np.asarray([row[m].mean() if m.any() else 0. for row, m in zip(values, valid)], np.float32)
        out[f"temporal__{name}__max"] = np.asarray([row[m].max() if m.any() else 0. for row, m in zip(values, valid)], np.float32)
    out["progress"] = view.progress; out["target"] = view.target; out["group_id"] = view.group_id
    return out
