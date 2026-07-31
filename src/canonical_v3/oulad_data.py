"""Build the strict canonical OULAD FINAL view without changing old evidence."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.pipelines import oulad

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "artifacts" / "canonical_v3" / "runtime" / "canonical_bundle.joblib"
CANONICAL_STAGES = (
    "E1_EARLY_20PCT",
    "E2_EARLY_35PCT",
    "M1_MIDDLE_50PCT",
    "L1_LATE_75PCT",
    "FINAL",
)
OLD_TO_CANONICAL = {
    "E1_EARLY_20PCT": "E1_EARLY_20PCT",
    "E2_EARLY_35PCT": "E2_EARLY_35PCT",
    "M1_MIDDLE_FROZEN": "M1_MIDDLE_50PCT",
    "L1_LATE_75PCT": "L1_LATE_75PCT",
}


def _final_stage(base: pd.DataFrame) -> tuple[oulad.StageData, pd.DataFrame]:
    frame = base.copy()
    frame["FINAL"] = frame.module_presentation_length.astype(int) - 14
    cutoff = frame["FINAL"].astype(int)
    registered = frame.date_registration.notna() & (frame.date_registration < cutoff)
    unregistered = frame.date_unregistration.notna() & (frame.date_unregistration < cutoff)
    frame = frame.loc[registered & ~unregistered].copy().reset_index(drop=True)
    frame["cutoff_day"] = frame["FINAL"].astype(int)
    frame["progress_fraction"] = frame.cutoff_day / frame.module_presentation_length
    frame["observed_week_count"] = np.ceil(frame.cutoff_day / 7).astype(int)
    frame["weeks_remaining"] = np.ceil(
        (frame.module_presentation_length - frame.cutoff_day) / 7
    ).astype(int)
    frame["assessment_available_fraction"] = 0.0

    previous_stages = oulad.STAGES
    try:
        oulad.STAGES = ("FINAL",)
        weekly, submissions = oulad._raw_weekly({"FINAL": frame})
    finally:
        oulad.STAGES = previous_stages

    lengths = frame.observed_week_count.to_numpy(dtype=int)
    max_weeks = int(lengths.max())
    mask = np.arange(max_weeks)[None, :] < lengths[:, None]
    sequence = np.zeros((len(frame), max_weeks, len(oulad.BASE_CHANNELS)), dtype=np.float32)
    weekly_groups = {
        str(key): value.set_index("week")
        for key, value in weekly["FINAL"].groupby("base_record_id", sort=False)
    }
    submission_groups = {
        str(key): value.set_index("week")
        for key, value in submissions["FINAL"].groupby("base_record_id", sort=False)
    }
    channel = {name: index for index, name in enumerate(oulad.BASE_CHANNELS)}
    for row_index, row in frame.iterrows():
        record = str(row.base_record_id)
        rows = weekly_groups.get(record)
        submitted_rows = submission_groups.get(record)
        last_activity: float | None = None
        inactive = 0
        for week in range(lengths[row_index]):
            if rows is not None and week in rows.index:
                observed = rows.loc[week]
                if isinstance(observed, pd.DataFrame):
                    observed = observed.iloc[0]
                for name in (
                    "total_clicks",
                    "content_clicks",
                    "forum_clicks",
                    "quiz_clicks",
                    "assessment_related_clicks",
                    "active_days",
                    "unique_sites",
                    "unique_activity_types",
                ):
                    sequence[row_index, week, channel[name]] = float(observed[name])
                last_activity = float(observed["last_vle_day"])
            if submitted_rows is not None and week in submitted_rows.index:
                observed = submitted_rows.loc[week]
                if isinstance(observed, pd.DataFrame):
                    observed = observed.iloc[0]
                sequence[row_index, week, channel["submitted_assessment_count"]] = float(
                    observed["submitted_assessment_count"]
                )
                sequence[row_index, week, channel["late_submission_count"]] = float(
                    observed["late_submission_count"]
                )
            if sequence[row_index, week, channel["total_clicks"]] == 0:
                inactive += 1
            day = min((week + 1) * 7, int(row.cutoff_day))
            sequence[row_index, week, channel["days_since_last_vle_activity"]] = float(
                day if last_activity is None else max(0, day - 1 - last_activity)
            )
            sequence[row_index, week, channel["weeks_without_activity"]] = inactive
            sequence[row_index, week, channel["score_missing_mask"]] = 1.0

    full_sequence = oulad._dynamic(sequence, mask)
    aggregate = oulad._aggregate(sequence, lengths)
    aggregate = np.column_stack(
        [aggregate, frame.loc[:, oulad.CONTEXT_COLUMNS].to_numpy(dtype=np.float32)]
    )
    keep = list(
        dict.fromkeys(
            [
                "base_record_id",
                "id_student",
                "code_module",
                "code_presentation",
                "outer_fold",
                "target",
                "outcome_aux",
                "date_unregistration",
                "cutoff_day",
                "module_presentation_length",
                *oulad.STATIC_COLUMNS,
                *oulad.CONTEXT_COLUMNS,
            ]
        )
    )
    data = oulad.StageData(
        "FINAL",
        frame.loc[:, keep].reset_index(drop=True),
        full_sequence,
        lengths,
        mask,
        aggregate.astype(np.float32),
    )
    data.validate()
    cutoff_frame = frame.loc[:, ["base_record_id", "FINAL"]].copy()
    return data, cutoff_frame


def build_canonical_bundle(*, force: bool = False) -> oulad.Bundle:
    if CACHE.is_file() and not force:
        return joblib.load(CACHE)
    old = oulad._build_bundle()
    final, final_cutoff = _final_stage(old.base)
    stages: dict[str, oulad.StageData] = {}
    for old_name, canonical_name in OLD_TO_CANONICAL.items():
        source = old.stages[old_name]
        stages[canonical_name] = oulad.StageData(
            canonical_name,
            source.frame.copy(),
            source.sequence.copy(),
            source.lengths.copy(),
            source.mask.copy(),
            source.aggregate.copy(),
        )
    stages["FINAL"] = final
    cutoff = old.cutoff.rename(columns={"M1_MIDDLE_FROZEN": "M1_MIDDLE_50PCT"}).merge(
        final_cutoff, on="base_record_id", validate="one_to_one"
    )
    bundle = oulad.Bundle(stages, old.base.copy(), cutoff)
    for stage in CANONICAL_STAGES:
        bundle.stages[stage].validate()
    order = cutoff.loc[:, list(CANONICAL_STAGES)].to_numpy(dtype=int)
    if not np.all(order[:, :-1] <= order[:, 1:]):
        raise RuntimeError("canonical OULAD information windows are not monotonic")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, CACHE, compress=3)
    return bundle


def stage_rows(bundle: oulad.Bundle, base_ids: set[str]) -> tuple:
    previous = oulad.STAGES
    try:
        oulad.STAGES = CANONICAL_STAGES
        return oulad._stage_rows(bundle, base_ids)
    finally:
        oulad.STAGES = previous


def single_stage_rows(
    bundle: oulad.Bundle, stage: str, base_ids: set[str]
) -> tuple:
    data = bundle.stages[stage]
    indices = np.flatnonzero(data.frame.base_record_id.isin(base_ids).to_numpy())
    frame = data.frame.iloc[indices].copy().reset_index(drop=True)
    frame["prediction_stage"] = stage
    return (
        frame,
        data.sequence[indices],
        data.lengths[indices],
        data.mask[indices],
        data.aggregate[indices],
        frame.target.to_numpy(dtype=np.float32),
        np.ones(len(indices), dtype=np.float32),
    )

