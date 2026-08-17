"""OULAD longitudinal weekly feature extraction, cutoff-safe aggregation, and view builders."""

from __future__ import annotations

import math
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.hybrid.contracts import HybridDataView
from src.hybrid.data.common import make_deterministic_id

# Conservative explicit VLE activity type sets
CONTENT_TYPES: set[str] = {
    "oucontent",
    "resource",
    "page",
    "url",
    "glossary",
    "homepage",
    "subpage",
    "dataplus",
}

FORUM_TYPES: set[str] = {
    "forumng",
}

QUIZ_TYPES: set[str] = {
    "quiz",
}

ASSESSMENT_RELATED_TYPES: set[str] = {
    "quiz",
    "questionnaire",
    "externalquiz",
}

OULAD_SENSITIVE_CONTEXT: list[str] = [
    "gender",
    "region",
    "disability",
    "age_band",
    "imd_band",
]

OULAD_CATEGORICAL_CONTEXT: list[str] = [
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
    "code_module",
    "presentation_season",
]

OULAD_NUMERIC_CONTEXT: list[str] = [
    "num_of_prev_attempts",
    "studied_credits",
    "registration_lead_time",
    "module_presentation_length",
]

OULAD_FORBIDDEN_PREDICTORS: list[str] = [
    "final_result",
    "date_unregistration",
    "score",  # Assessment score forbidden
]

OULAD_TEMPORAL_CHANNELS: list[str] = [
    # 1. Base weekly features (12)
    "total_clicks",
    "active_days",
    "unique_sites",
    "unique_activity_types",
    "content_clicks",
    "forum_clicks",
    "quiz_clicks",
    "assessment_related_clicks",
    "submitted_assessment_count",
    "late_submission_count",
    "days_since_last_vle_activity",
    "cumulative_inactive_weeks",
    # 2. Count transforms (5)
    "log1p_total_clicks",
    "log1p_active_days",
    "log1p_unique_sites",
    "log1p_assessment_related_clicks",
    "log1p_submitted_assessment_count",
    # 3. First-order deltas (8)
    "delta_total_clicks",
    "delta_active_days",
    "delta_unique_sites",
    "delta_content_clicks",
    "delta_forum_clicks",
    "delta_quiz_clicks",
    "delta_assessment_related_clicks",
    "delta_submitted_assessment_count",
    # 4. Rolling two-week (4)
    "rolling2_total_clicks",
    "rolling2_active_days",
    "rolling2_assessment_related_clicks",
    "rolling2_submission_count",
    # 5. Behavior states (3)
    "current_inactivity_streak",
    "activity_resumed_indicator",
    "new_inactivity_indicator",
    # 6. Activity composition shares (4)
    "content_share",
    "forum_share",
    "quiz_share",
    "assessment_share",
    # 7. Submission rate to date (1)
    "late_submission_rate_to_date",
]


def load_oulad_static_tables(
    raw_dir: str | Path = "data/raw",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load courses, studentInfo, studentRegistration and build base student presentation records."""
    raw_path = Path(raw_dir)
    courses = pd.read_csv(raw_path / "courses.csv")
    student_info = pd.read_csv(raw_path / "studentInfo.csv")
    student_reg = pd.read_csv(raw_path / "studentRegistration.csv")

    if "length" in courses.columns and "module_presentation_length" not in courses.columns:
        courses = courses.rename(columns={"length": "module_presentation_length"})

    merged = pd.merge(
        student_info,
        student_reg,
        on=["code_module", "code_presentation", "id_student"],
        how="inner",
    )
    merged = pd.merge(
        merged,
        courses,
        on=["code_module", "code_presentation"],
        how="inner",
    )

    valid_results = {"Fail", "Withdrawn", "Pass", "Distinction"}
    unknown = set(merged["final_result"].unique()) - valid_results
    if unknown:
        raise ValueError(f"Unknown final_result values found: {unknown}")

    merged["target"] = merged["final_result"].isin({"Fail", "Withdrawn"}).astype(np.int64)
    merged["presentation_season"] = merged["code_presentation"].astype(str).str[-1]
    merged["registration_lead_time"] = -merged["date_registration"].fillna(0.0).astype(np.float64)
    merged["group_id"] = merged["id_student"].astype(str)
    merged["record_id"] = merged.apply(
        lambda r: make_deterministic_id(
            "oulad", r["code_module"], r["code_presentation"], r["id_student"], length=24
        ),
        axis=1,
    )

    return courses, student_reg, merged


def build_compact_vle_daily(
    raw_dir: str | Path = "data/raw",
    cache_dir: str | Path = "artifacts/hybrid/phase1/runtime",
    chunk_size: int = 1_000_000,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Read studentVle in chunked passes, merge full VLE metadata key, and aggregate daily-site clicks."""
    raw_path = Path(raw_dir)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    compact_vle_file = cache_path / "compact_vle_daily_site.parquet"
    metadata_file = cache_path / "compact_vle_daily_site.metadata.json"

    def _sha(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _expected_metadata() -> dict[str, str | int]:
        try:
            phase1_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            phase1_commit = "unknown"
        manifest = Path("artifacts/hybrid/phase1/oulad_temporal_channels.json")
        return {"cache_schema_version": 1, "protocol_data_version": 3,
                "raw_studentVle_sha256": _sha(raw_path / "studentVle.csv"),
                "raw_vle_sha256": _sha(raw_path / "vle.csv"),
                "temporal_channel_manifest_sha256": _sha(manifest) if manifest.is_file() else "missing",
                "phase1_commit_sha": phase1_commit}

    # Remove stale cache if present
    old_cache = cache_path / "compact_vle_daily.parquet"
    if old_cache.is_file():
        try:
            old_cache.unlink()
        except Exception:
            pass

    expected_metadata = _expected_metadata()
    if compact_vle_file.is_file() and metadata_file.is_file() and not force_rebuild:
        cached_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if all(cached_metadata.get(k) == v for k, v in expected_metadata.items()) and cached_metadata.get("cache_file_sha256") == _sha(compact_vle_file):
            print(f"Loading fingerprint-matched compact VLE daily-site from {compact_vle_file}")
            return pd.read_parquet(compact_vle_file)

    print("Building compact VLE daily-site representation from raw studentVle.csv...")
    vle_df = pd.read_csv(raw_path / "vle.csv")
    # Merge on full key: (code_module, code_presentation, id_site)
    vle_key_map = dict(
        zip(
            zip(vle_df["code_module"], vle_df["code_presentation"], vle_df["id_site"]),
            vle_df["activity_type"],
        )
    )

    chunk_aggregates: list[pd.DataFrame] = []
    vle_path = raw_path / "studentVle.csv"

    for chunk in pd.read_csv(vle_path, chunksize=chunk_size):
        keys = list(zip(chunk["code_module"], chunk["code_presentation"], chunk["id_site"]))
        chunk["activity_type"] = [vle_key_map.get(k, "other") for k in keys]

        agg = (
            chunk.groupby(
                ["code_module", "code_presentation", "id_student", "date", "id_site", "activity_type"],
                as_index=False,
            )["sum_click"]
            .sum()
        )
        chunk_aggregates.append(agg)

    combined_agg = pd.concat(chunk_aggregates, ignore_index=True)
    final_agg = (
        combined_agg.groupby(
            ["code_module", "code_presentation", "id_student", "date", "id_site", "activity_type"],
            as_index=False,
        )["sum_click"]
        .sum()
    )

    final_agg.to_parquet(compact_vle_file, index=False)
    expected_metadata.update({"cache_file_sha256": _sha(compact_vle_file), "row_count": int(len(final_agg))})
    metadata_file.write_text(json.dumps(expected_metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved compact VLE daily-site ({len(final_agg)} rows) to {compact_vle_file}")
    return final_agg


def load_assessment_events(
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Load studentAssessment joined with assessment metadata (filtering banked items)."""
    raw_path = Path(raw_dir)
    assessments = pd.read_csv(raw_path / "assessments.csv")
    student_assess = pd.read_csv(raw_path / "studentAssessment.csv")

    merged = pd.merge(
        student_assess,
        assessments[["id_assessment", "code_module", "code_presentation", "date", "assessment_type", "weight"]],
        on="id_assessment",
        how="inner",
        suffixes=("", "_due"),
    )

    unbanked = merged[merged["is_banked"] == 0].copy()
    unbanked = unbanked[unbanked["date_submitted"].notna()]
    unbanked["is_late"] = (unbanked["date_submitted"] > unbanked["date"]).astype(np.int64)
    return unbanked


def compute_weekly_features_at_cutoff(
    base_df: pd.DataFrame,
    vle_daily: pd.DataFrame,
    assessments_df: pd.DataFrame,
    cutoff_fraction: float,
    *,
    include_pre_end_withdrawals: bool = False,
) -> tuple[pd.DataFrame, HybridDataView, dict[str, Any]]:
    """Build cutoff-filtered longitudinal weekly tensor for eligible students with exact weekly uniqueness and recency."""
    pres_lengths = base_df.groupby(["code_module", "code_presentation"])["module_presentation_length"].first()
    cutoff_days = {
        k: max(1, int(math.floor(length * cutoff_fraction)))
        for k, length in pres_lengths.items()
    }

    base = base_df.copy()
    if "target" not in base.columns and "final_result" in base.columns:
        base["target"] = base["final_result"].isin({"Fail", "Withdrawn"}).astype(np.int64)
    base["cutoff_day"] = base.apply(
        lambda r: cutoff_days[(r["code_module"], r["code_presentation"])], axis=1
    )

    # 1. Vectorized Eligibility filtering
    reg_ok = base["date_registration"].isna() | (base["date_registration"] <= base["cutoff_day"])
    unreg_ok = (pd.Series(True, index=base.index) if include_pre_end_withdrawals else
                base["date_unregistration"].isna() | (base["date_unregistration"] > base["cutoff_day"]))

    excluded_late_reg = int((~reg_ok).sum())
    excluded_early_unreg = int((reg_ok & (~unreg_ok)).sum())

    eligible_df = base[reg_ok & unreg_ok].copy().reset_index(drop=True)
    n_eligible = len(eligible_df)
    if n_eligible == 0:
        raise ValueError(f"No eligible records at cutoff fraction {cutoff_fraction}")

    eligible_df["student_idx"] = np.arange(n_eligible)
    eligible_lookup = eligible_df[["code_module", "code_presentation", "id_student", "student_idx", "cutoff_day"]]

    # 2. Weekly slot counts
    student_num_weeks = (
        eligible_df["cutoff_day"].apply(lambda d: int(math.floor((d - 1) / 7)) + 1).to_numpy(dtype=np.int64)
    )
    max_t = int(student_num_weeks.max())
    n_channels = len(OULAD_TEMPORAL_CHANNELS)
    channel_idx = {name: idx for idx, name in enumerate(OULAD_TEMPORAL_CHANNELS)}

    temporal = np.zeros((n_eligible, max_t, n_channels), dtype=np.float32)
    mask = np.zeros((n_eligible, max_t), dtype=bool)

    for i in range(n_eligible):
        mask[i, : student_num_weeks[i]] = True

    # 3. Cutoff-Filtered VLE Events
    vle_joined = pd.merge(
        vle_daily,
        eligible_lookup,
        on=["code_module", "code_presentation", "id_student"],
        how="inner",
    )
    # Strict event filter: date >= 0 and date < cutoff_day
    vle_valid = vle_joined[(vle_joined["date"] >= 0) & (vle_joined["date"] < vle_joined["cutoff_day"])].copy()
    vle_valid["week"] = (vle_valid["date"] // 7).astype(np.int64)

    # Dictionary for exact recency tracking per student: student_idx -> list of active event dates
    student_active_dates: dict[int, list[int]] = {}

    if not vle_valid.empty:
        # A. Exact total_clicks, active_days, unique_sites, unique_activity_types per (student_idx, week)
        vle_weekly_exact = (
            vle_valid.groupby(["student_idx", "week"])
            .agg(
                total_clicks=("sum_click", "sum"),
                active_days=("date", "nunique"),
                unique_sites=("id_site", "nunique"),
                unique_activity_types=("activity_type", "nunique"),
            )
            .reset_index()
        )

        s_idx = vle_weekly_exact["student_idx"].to_numpy(dtype=np.int64)
        w_idx = vle_weekly_exact["week"].to_numpy(dtype=np.int64)
        valid_pos = (w_idx < max_t) & (w_idx < student_num_weeks[s_idx])
        s_idx_v = s_idx[valid_pos]
        w_idx_v = w_idx[valid_pos]

        temporal[s_idx_v, w_idx_v, channel_idx["total_clicks"]] = vle_weekly_exact["total_clicks"].to_numpy()[valid_pos]
        temporal[s_idx_v, w_idx_v, channel_idx["active_days"]] = vle_weekly_exact["active_days"].to_numpy()[valid_pos]
        temporal[s_idx_v, w_idx_v, channel_idx["unique_sites"]] = vle_weekly_exact["unique_sites"].to_numpy()[valid_pos]
        temporal[s_idx_v, w_idx_v, channel_idx["unique_activity_types"]] = vle_weekly_exact["unique_activity_types"].to_numpy()[valid_pos]

        # B. Exact category clicks
        # Content clicks
        content_mask = vle_valid["activity_type"].isin(CONTENT_TYPES)
        if content_mask.any():
            c_agg = vle_valid[content_mask].groupby(["student_idx", "week"])["sum_click"].sum().reset_index()
            s_c = c_agg["student_idx"].to_numpy(dtype=np.int64)
            w_c = c_agg["week"].to_numpy(dtype=np.int64)
            val_c = (w_c < max_t) & (w_c < student_num_weeks[s_c])
            temporal[s_c[val_c], w_c[val_c], channel_idx["content_clicks"]] = c_agg["sum_click"].to_numpy()[val_c]

        # Forum clicks
        forum_mask = vle_valid["activity_type"].isin(FORUM_TYPES)
        if forum_mask.any():
            f_agg = vle_valid[forum_mask].groupby(["student_idx", "week"])["sum_click"].sum().reset_index()
            s_f = f_agg["student_idx"].to_numpy(dtype=np.int64)
            w_f = f_agg["week"].to_numpy(dtype=np.int64)
            val_f = (w_f < max_t) & (w_f < student_num_weeks[s_f])
            temporal[s_f[val_f], w_f[val_f], channel_idx["forum_clicks"]] = f_agg["sum_click"].to_numpy()[val_f]

        # Quiz clicks
        quiz_mask = vle_valid["activity_type"].isin(QUIZ_TYPES)
        if quiz_mask.any():
            q_agg = vle_valid[quiz_mask].groupby(["student_idx", "week"])["sum_click"].sum().reset_index()
            s_q = q_agg["student_idx"].to_numpy(dtype=np.int64)
            w_q = q_agg["week"].to_numpy(dtype=np.int64)
            val_q = (w_q < max_t) & (w_q < student_num_weeks[s_q])
            temporal[s_q[val_q], w_q[val_q], channel_idx["quiz_clicks"]] = q_agg["sum_click"].to_numpy()[val_q]

        # Assessment related clicks (includes quiz, questionnaire, externalquiz)
        ass_mask = vle_valid["activity_type"].isin(ASSESSMENT_RELATED_TYPES)
        if ass_mask.any():
            a_agg = vle_valid[ass_mask].groupby(["student_idx", "week"])["sum_click"].sum().reset_index()
            s_a = a_agg["student_idx"].to_numpy(dtype=np.int64)
            w_a = a_agg["week"].to_numpy(dtype=np.int64)
            val_a = (w_a < max_t) & (w_a < student_num_weeks[s_a])
            temporal[s_a[val_a], w_a[val_a], channel_idx["assessment_related_clicks"]] = a_agg["sum_click"].to_numpy()[val_a]

        # Collect unique active dates per student for exact days_since_last_vle_activity
        date_groups = vle_valid.groupby("student_idx")["date"].unique()
        for s_i, dates in date_groups.items():
            student_active_dates[s_i] = sorted(dates)

    # 4. Assessment Submissions Aggregation
    assess_joined = pd.merge(
        assessments_df,
        eligible_lookup,
        on=["code_module", "code_presentation", "id_student"],
        how="inner",
    )
    assess_valid = assess_joined[
        (assess_joined["date_submitted"] >= 0) & (assess_joined["date_submitted"] < assess_joined["cutoff_day"])
    ].copy()
    assess_valid["week"] = (assess_valid["date_submitted"] // 7).astype(np.int64)

    if not assess_valid.empty:
        assess_totals = (
            assess_valid.groupby(["student_idx", "week"])
            .agg(
                submitted_count=("id_assessment", "count"),
                late_count=("is_late", "sum"),
            )
            .reset_index()
        )

        s_a = assess_totals["student_idx"].to_numpy(dtype=np.int64)
        w_a = assess_totals["week"].to_numpy(dtype=np.int64)
        valid_pos_a = (w_a < max_t) & (w_a < student_num_weeks[s_a])
        s_av = s_a[valid_pos_a]
        w_av = w_a[valid_pos_a]

        temporal[s_av, w_av, channel_idx["submitted_assessment_count"]] = assess_totals["submitted_count"].to_numpy()[valid_pos_a]
        temporal[s_av, w_av, channel_idx["late_submission_count"]] = assess_totals["late_count"].to_numpy()[valid_pos_a]

    # 5. Derived historical features (Transforms, Deltas, Rolling2, Behavior States, Recency)
    # Log1p
    temporal[:, :, channel_idx["log1p_total_clicks"]] = np.log1p(temporal[:, :, channel_idx["total_clicks"]])
    temporal[:, :, channel_idx["log1p_active_days"]] = np.log1p(temporal[:, :, channel_idx["active_days"]])
    temporal[:, :, channel_idx["log1p_unique_sites"]] = np.log1p(temporal[:, :, channel_idx["unique_sites"]])
    temporal[:, :, channel_idx["log1p_assessment_related_clicks"]] = np.log1p(temporal[:, :, channel_idx["assessment_related_clicks"]])
    temporal[:, :, channel_idx["log1p_submitted_assessment_count"]] = np.log1p(temporal[:, :, channel_idx["submitted_assessment_count"]])

    # Deltas
    for feat in [
        "total_clicks",
        "active_days",
        "unique_sites",
        "content_clicks",
        "forum_clicks",
        "quiz_clicks",
        "assessment_related_clicks",
        "submitted_assessment_count",
    ]:
        delta_feat = f"delta_{feat}"
        orig = temporal[:, :, channel_idx[feat]]
        diff = np.zeros_like(orig)
        diff[:, 1:] = orig[:, 1:] - orig[:, :-1]
        temporal[:, :, channel_idx[delta_feat]] = diff

    # Rolling 2-week
    for feat in [
        ("total_clicks", "rolling2_total_clicks"),
        ("active_days", "rolling2_active_days"),
        ("assessment_related_clicks", "rolling2_assessment_related_clicks"),
        ("submitted_assessment_count", "rolling2_submission_count"),
    ]:
        orig = temporal[:, :, channel_idx[feat[0]]]
        r2 = np.zeros_like(orig)
        r2[:, 0] = orig[:, 0]
        r2[:, 1:] = (orig[:, 1:] + orig[:, :-1]) / 2.0
        temporal[:, :, channel_idx[feat[1]]] = r2

    # Behavior states & shares
    tot_clicks_arr = temporal[:, :, channel_idx["total_clicks"]]
    tot_safe = np.maximum(1.0, tot_clicks_arr)
    temporal[:, :, channel_idx["content_share"]] = temporal[:, :, channel_idx["content_clicks"]] / tot_safe
    temporal[:, :, channel_idx["forum_share"]] = temporal[:, :, channel_idx["forum_clicks"]] / tot_safe
    temporal[:, :, channel_idx["quiz_share"]] = temporal[:, :, channel_idx["quiz_clicks"]] / tot_safe
    temporal[:, :, channel_idx["assessment_share"]] = temporal[:, :, channel_idx["assessment_related_clicks"]] / tot_safe

    is_active = tot_clicks_arr > 0
    resumed = np.zeros_like(is_active, dtype=np.float32)
    new_inact = np.zeros_like(is_active, dtype=np.float32)
    resumed[:, 1:] = (is_active[:, 1:] & (~is_active[:, :-1])).astype(np.float32)
    new_inact[:, 1:] = ((~is_active[:, 1:]) & is_active[:, :-1]).astype(np.float32)

    temporal[:, :, channel_idx["activity_resumed_indicator"]] = resumed
    temporal[:, :, channel_idx["new_inactivity_indicator"]] = new_inact

    # Cumulative late submission rate
    cum_subs = np.cumsum(temporal[:, :, channel_idx["submitted_assessment_count"]], axis=1)
    cum_late = np.cumsum(temporal[:, :, channel_idx["late_submission_count"]], axis=1)
    temporal[:, :, channel_idx["late_submission_rate_to_date"]] = cum_late / np.maximum(1.0, cum_subs)

    # Exact days_since_last_vle_activity, current_inactivity_streak, cumulative_inactive_weeks
    days_since_arr = np.zeros((n_eligible, max_t), dtype=np.float32)
    streak_arr = np.zeros((n_eligible, max_t), dtype=np.float32)
    cum_inactive_arr = np.zeros((n_eligible, max_t), dtype=np.float32)

    student_cutoff_days = eligible_df["cutoff_day"].to_numpy(dtype=np.int64)

    for i in range(n_eligible):
        n_w = student_num_weeks[i]
        c_day = student_cutoff_days[i]
        active_dates = student_active_dates.get(i, [])

        curr_streak = 0
        tot_inactive = 0

        for w in range(n_w):
            # Inactivity streak & cumulative inactive weeks
            if tot_clicks_arr[i, w] == 0:
                curr_streak += 1
                tot_inactive += 1
            else:
                curr_streak = 0

            streak_arr[i, w] = curr_streak
            cum_inactive_arr[i, w] = tot_inactive

            # Exact days since last VLE activity
            week_end_exclusive = min((w + 1) * 7, c_day)
            # Find latest active date strictly before week_end_exclusive
            prior_dates = [d for d in active_dates if d < week_end_exclusive]
            if prior_dates:
                last_act = prior_dates[-1]
                days_since_arr[i, w] = float(max(0, week_end_exclusive - 1 - last_act))
            else:
                # No activity in course up to week_end_exclusive
                days_since_arr[i, w] = float(week_end_exclusive)

    temporal[:, :, channel_idx["days_since_last_vle_activity"]] = days_since_arr
    temporal[:, :, channel_idx["current_inactivity_streak"]] = streak_arr
    temporal[:, :, channel_idx["cumulative_inactive_weeks"]] = cum_inactive_arr

    # Ensure all padded positions are strictly zero
    padded = ~mask
    for c in range(n_channels):
        temporal[:, :, c][padded] = 0.0

    lengths = np.sum(mask, axis=1).astype(np.int64)
    record_ids = eligible_df["record_id"].to_numpy(dtype=str)
    group_ids = eligible_df["group_id"].to_numpy(dtype=str)
    targets = eligible_df["target"].to_numpy(dtype=np.int64)

    view = HybridDataView(
        record_id=record_ids,
        group_id=group_ids,
        target=targets,
        temporal=temporal,
        mask=mask,
        lengths=lengths,
        context=None,
        metadata={
            "cutoff_fraction": cutoff_fraction,
            "domain": "oulad",
            "channels": OULAD_TEMPORAL_CHANNELS,
        },
    )
    view.validate()

    audit_summary = {
        "cutoff_fraction": float(cutoff_fraction),
        "total_base_records": int(len(base_df)),
        "eligible_records": int(n_eligible),
        "excluded_late_registration": int(excluded_late_reg),
        "excluded_early_unregistration": int(excluded_early_unreg),
        "risk_records": int(targets.sum()),
        "risk_prevalence": float(targets.mean()),
        "unique_students": int(len(set(group_ids))),
        "max_sequence_length": int(max_t),
        "temporal_channels_count": int(n_channels),
    }

    return eligible_df, view, audit_summary


def build_oulad_final_diagnostic_view(
    base_df: pd.DataFrame,
    vle_daily: pd.DataFrame,
    assessments_df: pd.DataFrame,
) -> tuple[pd.DataFrame, HybridDataView, dict[str, Any]]:
    """Course-end supplemental view retaining pre-end withdrawals.

    This is a post-course diagnostic, not an early-warning cutoff. Post-withdrawal
    inactivity can encode outcome-adjacent information and must not guide tuning.
    """
    eligible, view, audit = compute_weekly_features_at_cutoff(
        base_df, vle_daily, assessments_df, 1.0, include_pre_end_withdrawals=True
    )
    metadata = dict(view.metadata)
    metadata.update({"evaluation_scope": "POST_COURSE_FINAL_DIAGNOSTIC", "early_warning_claim_allowed": False})
    final_view = HybridDataView(view.record_id, view.group_id, view.target, view.temporal,
                                view.mask, view.lengths, view.context, metadata)
    audit.update({"evaluation_scope": "POST_COURSE_FINAL_DIAGNOSTIC", "early_warning_claim_allowed": False,
                  "pre_end_withdrawals_retained": True})
    return eligible, final_view, audit
