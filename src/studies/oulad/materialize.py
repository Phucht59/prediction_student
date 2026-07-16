from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.studies.common.hashing import semantic_sha256, sha256_file
from src.studies.oulad.cohort import FORECASTS, materialize_landmark_cohort


CHANNELS = [
    "total_clicks", "active_days", "unique_sites", "unique_activity_types",
    "content_clicks", "forum_clicks", "quiz_clicks", "assessment_related_clicks",
    "submitted_assessment_count", "late_submission_count", "available_score_count",
    "cumulative_mean_score", "cumulative_weighted_score", "days_since_last_vle_activity",
    "weeks_without_activity", "score_missing_mask",
]
CONTENT_TYPES = {"oucontent", "resource", "page", "url", "glossary", "homepage", "subpage", "dataplus"}
FORUM_TYPES = {"forumng"}
QUIZ_TYPES = {"quiz"}
ASSESSMENT_TYPES = {"quiz", "questionnaire", "externalquiz"}


def _weekly_vle(raw_root: Path, courses: pd.DataFrame, vle: pd.DataFrame, chunksize: int = 750_000) -> dict[str, pd.DataFrame]:
    cutoffs = courses.copy()
    for forecast_id, fraction in FORECASTS.items():
        cutoffs[forecast_id] = np.floor(cutoffs["module_presentation_length"] * fraction).astype(int)
    activity = vle[["code_module", "code_presentation", "id_site", "activity_type"]].drop_duplicates()
    if activity.duplicated(["code_module", "code_presentation", "id_site"]).any():
        raise ValueError("VLE site key is not unique on module/presentation/site")
    totals: dict[str, list[pd.DataFrame]] = {forecast_id: [] for forecast_id in FORECASTS}
    days: dict[str, list[pd.DataFrame]] = {forecast_id: [] for forecast_id in FORECASTS}
    sites: dict[str, list[pd.DataFrame]] = {forecast_id: [] for forecast_id in FORECASTS}
    activities: dict[str, list[pd.DataFrame]] = {forecast_id: [] for forecast_id in FORECASTS}
    usecols = ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]
    for chunk in pd.read_csv(raw_root / "studentVle.csv", usecols=usecols, chunksize=chunksize):
        for forecast_id in FORECASTS:
            prepared = _prepare_vle_chunk(chunk, cutoffs, activity, forecast_id)
            keys = ["code_module", "code_presentation", "id_student", "week"]
            totals[forecast_id].append(prepared.groupby(keys, as_index=False).agg(total_clicks=("sum_click", "sum"), content_clicks=("content_clicks", "sum"), forum_clicks=("forum_clicks", "sum"), quiz_clicks=("quiz_clicks", "sum"), assessment_related_clicks=("assessment_related_clicks", "sum"), last_vle_date=("date", "max")))
            days[forecast_id].append(prepared[keys + ["date"]].drop_duplicates())
            sites[forecast_id].append(prepared[keys + ["id_site"]].drop_duplicates())
            activities[forecast_id].append(prepared[keys + ["activity_type"]].drop_duplicates())
    result = {}
    keys = ["code_module", "code_presentation", "id_student", "week"]
    for forecast_id in FORECASTS:
        weekly = pd.concat(totals[forecast_id], ignore_index=True).groupby(keys, as_index=False).agg(total_clicks=("total_clicks", "sum"), content_clicks=("content_clicks", "sum"), forum_clicks=("forum_clicks", "sum"), quiz_clicks=("quiz_clicks", "sum"), assessment_related_clicks=("assessment_related_clicks", "sum"), last_vle_date=("last_vle_date", "max"))
        active_days = pd.concat(days[forecast_id], ignore_index=True).drop_duplicates().groupby(keys, as_index=False).size().rename(columns={"size": "active_days"})
        unique_sites = pd.concat(sites[forecast_id], ignore_index=True).drop_duplicates().groupby(keys, as_index=False).size().rename(columns={"size": "unique_sites"})
        unique_activities = pd.concat(activities[forecast_id], ignore_index=True).drop_duplicates().groupby(keys, as_index=False).size().rename(columns={"size": "unique_activity_types"})
        weekly = weekly.merge(active_days, on=keys, validate="one_to_one").merge(unique_sites, on=keys, validate="one_to_one").merge(unique_activities, on=keys, validate="one_to_one")
        if weekly.duplicated(keys).any():
            raise RuntimeError("Weekly VLE aggregation is not unique")
        result[forecast_id] = weekly
    return result


def _prepare_vle_chunk(chunk: pd.DataFrame, cutoffs: pd.DataFrame, activity: pd.DataFrame, forecast_id: str) -> pd.DataFrame:
    chunk = chunk.merge(cutoffs[["code_module", "code_presentation", forecast_id]], on=["code_module", "code_presentation"], how="left", validate="many_to_one")
    chunk = chunk[(chunk["date"] >= 0) & (chunk["date"] < chunk[forecast_id])]
    chunk = chunk.merge(activity, on=["code_module", "code_presentation", "id_site"], how="left", validate="many_to_one")
    chunk["week"] = (chunk["date"] // 7).astype(int)
    for name, types in [("content_clicks", CONTENT_TYPES), ("forum_clicks", FORUM_TYPES), ("quiz_clicks", QUIZ_TYPES), ("assessment_related_clicks", ASSESSMENT_TYPES)]:
        chunk[name] = np.where(chunk["activity_type"].isin(types), chunk["sum_click"], 0)
    return chunk


def _assessment_events(raw_root: Path, courses: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    assessments = pd.read_csv(raw_root / "assessments.csv")
    submissions = pd.read_csv(raw_root / "studentAssessment.csv")
    joined = submissions.merge(assessments, on="id_assessment", how="left", validate="many_to_one")
    if joined["code_module"].isna().any():
        raise ValueError("Submission has no assessment parent")
    cutoffs = courses.copy()
    for forecast_id, fraction in FORECASTS.items():
        cutoffs[forecast_id] = np.floor(cutoffs["module_presentation_length"] * fraction).astype(int)
    joined = joined.merge(cutoffs[["code_module", "code_presentation", *FORECASTS]], on=["code_module", "code_presentation"], how="left", validate="many_to_one")
    joined = joined[(joined["is_banked"] == 0) & joined["date"].notna()].copy()
    keys = ["code_module", "code_presentation", "id_student", "week"]
    result = {}
    for forecast_id in FORECASTS:
        submitted = joined[(joined["date_submitted"] >= 0) & (joined["date_submitted"] < joined[forecast_id])].copy()
        submitted["week"] = (submitted["date_submitted"] // 7).astype(int)
        submitted["late"] = (submitted["date_submitted"] > submitted["date"]).astype(int)
        submission_weekly = submitted.groupby(keys, as_index=False).agg(submitted_assessment_count=("id_assessment", "count"), late_submission_count=("late", "sum"))
        available = joined[joined["score"].notna()].copy()
        available["availability_day"] = available[["date_submitted", "date"]].max(axis=1)
        available = available[(available["date_submitted"] < available[forecast_id]) & (available["date"] < available[forecast_id]) & (available["availability_day"] >= 0)]
        available["week"] = (available["availability_day"] // 7).astype(int)
        available["weighted_score_numerator"] = available["score"] * available["weight"]
        score_weekly = available.groupby(keys, as_index=False).agg(available_score_count=("score", "count"), score_sum=("score", "sum"), weighted_score_numerator=("weighted_score_numerator", "sum"), weight_sum=("weight", "sum"))
        result[forecast_id] = (submission_weekly, score_weekly)
    return result


def _aggregate_features(sequence: np.ndarray, valid_lengths: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_rows: list[dict[str, float]] = []
    flat_rows: list[dict[str, float]] = []
    for row_index, length in enumerate(valid_lengths.astype(int)):
        values = sequence[row_index, :length]
        aggregate: dict[str, float] = {}
        flattened: dict[str, float] = {}
        x = np.arange(length, dtype=float)
        for channel_index, channel in enumerate(CHANNELS):
            channel_values = values[:, channel_index].astype(float)
            half = max(1, length // 2)
            slope = 0.0 if length < 2 else float(np.polyfit(x, channel_values, 1)[0])
            aggregate.update({
                f"{channel}__sum": float(channel_values.sum()), f"{channel}__mean": float(channel_values.mean()),
                f"{channel}__std": float(channel_values.std()), f"{channel}__min": float(channel_values.min()),
                f"{channel}__max": float(channel_values.max()), f"{channel}__last": float(channel_values[-1]),
                f"{channel}__slope": slope, f"{channel}__recent_2_week_mean": float(channel_values[-2:].mean()),
                f"{channel}__first_half_mean": float(channel_values[:half].mean()), f"{channel}__second_half_mean": float(channel_values[half:].mean()) if half < length else float(channel_values[-1]),
            })
            for week in range(sequence.shape[1]):
                flattened[f"week_{week + 1:02d}__{channel}"] = float(sequence[row_index, week, channel_index])
        for week in range(sequence.shape[1]):
            flattened[f"week_{week + 1:02d}__valid"] = float(week < length)
        aggregate["inactive_week_count"] = float((values[:, CHANNELS.index("total_clicks")] == 0).sum())
        flat_rows.append(flattened)
        aggregate_rows.append(aggregate)
    return pd.DataFrame(aggregate_rows), pd.DataFrame(flat_rows)


def rebuild_derived_from_sequences(processed_root: Path) -> dict[str, object]:
    rebuilt = {}
    for forecast_id in FORECASTS:
        archive = np.load(processed_root / "sequences" / f"{forecast_id}.npz", allow_pickle=True)
        sequence = archive["sequence"]
        record_ids = archive["record_ids"]
        valid_lengths = archive["valid_lengths"]
        aggregate, flat = _aggregate_features(sequence, valid_lengths)
        aggregate.insert(0, "record_id", record_ids); flat.insert(0, "record_id", record_ids)
        aggregate_path = processed_root / "aggregated" / f"{forecast_id}.parquet"
        flat_path = processed_root / "flat" / f"{forecast_id}.parquet"
        aggregate.to_parquet(aggregate_path, index=False); flat.to_parquet(flat_path, index=False)
        manifest_path = processed_root / "manifests" / f"{forecast_id}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["checksums"]["aggregated"] = sha256_file(aggregate_path)
        manifest["checksums"]["flat"] = sha256_file(flat_path)
        manifest["flat_padding_indicator"] = True
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        rebuilt[forecast_id] = {"aggregated_columns": len(aggregate.columns), "flat_columns": len(flat.columns), "flat_sha256": manifest["checksums"]["flat"]}
    return {"status": "PASS", "rebuilt": rebuilt}


def materialize_all(raw_root: Path, processed_root: Path, protocol: dict) -> dict[str, object]:
    courses = pd.read_csv(raw_root / "courses.csv")
    info = pd.read_csv(raw_root / "studentInfo.csv")
    registration = pd.read_csv(raw_root / "studentRegistration.csv")
    vle = pd.read_csv(raw_root / "vle.csv")
    weekly_vle = _weekly_vle(raw_root, courses, vle)
    assessment_events = _assessment_events(raw_root, courses)
    for child in ["normalized", "cohorts", "targets", "sequences", "aggregated", "flat", "manifests"]:
        (processed_root / child).mkdir(parents=True, exist_ok=True)
    for forecast_id in FORECASTS:
        weekly_vle[forecast_id].to_parquet(processed_root / "normalized" / f"weekly_vle_{forecast_id}.parquet", index=False)
        assessment_events[forecast_id][0].to_parquet(processed_root / "normalized" / f"weekly_submissions_{forecast_id}.parquet", index=False)
        assessment_events[forecast_id][1].to_parquet(processed_root / "normalized" / f"weekly_scores_{forecast_id}.parquet", index=False)

    manifests = {}
    flow_rows = []
    for forecast_id in FORECASTS:
        cohort, targets, flow = materialize_landmark_cohort(info, registration, courses, forecast_id)
        max_weeks = int(cohort["valid_sequence_length"].max())
        sequence = np.zeros((len(cohort), max_weeks, len(CHANNELS)), dtype=np.float32)
        record_lookup = {(row.code_module, row.code_presentation, int(row.id_student)): index for index, row in cohort.iterrows()}
        cutoff_lookup = cohort["cutoff_day"].to_numpy(int)

        forecast_vle = weekly_vle[forecast_id]
        submission_weekly, score_weekly = assessment_events[forecast_id]
        vle_rows = forecast_vle.merge(cohort[["code_module", "code_presentation", "id_student", "record_id"]], on=["code_module", "code_presentation", "id_student"], how="inner")
        for row in vle_rows.itertuples(index=False):
            index = record_lookup[(row.code_module, row.code_presentation, int(row.id_student))]
            week = int(row.week)
            if week >= int(cohort.iloc[index]["valid_sequence_length"]): continue
            for channel in ["total_clicks", "active_days", "unique_sites", "unique_activity_types", "content_clicks", "forum_clicks", "quiz_clicks", "assessment_related_clicks"]:
                sequence[index, week, CHANNELS.index(channel)] = float(getattr(row, channel))

        for table, names in [(submission_weekly, ["submitted_assessment_count", "late_submission_count"]), (score_weekly, ["available_score_count", "score_sum", "weighted_score_numerator", "weight_sum"])]:
            rows = table.merge(cohort[["code_module", "code_presentation", "id_student"]], on=["code_module", "code_presentation", "id_student"], how="inner")
            for row in rows.itertuples(index=False):
                index = record_lookup[(row.code_module, row.code_presentation, int(row.id_student))]
                week = int(row.week)
                if week >= int(cohort.iloc[index]["valid_sequence_length"]): continue
                if "submitted_assessment_count" in names:
                    sequence[index, week, CHANNELS.index("submitted_assessment_count")] = float(row.submitted_assessment_count)
                    sequence[index, week, CHANNELS.index("late_submission_count")] = float(row.late_submission_count)

        score_lookup = score_weekly.set_index(["code_module", "code_presentation", "id_student", "week"])
        last_lookup = forecast_vle.set_index(["code_module", "code_presentation", "id_student", "week"])["last_vle_date"]
        for index, row in cohort.iterrows():
            count = score_sum = weighted_num = weight_sum = 0.0
            last_activity = None
            inactive = 0
            for week in range(int(row["valid_sequence_length"])):
                key = (row["code_module"], row["code_presentation"], int(row["id_student"]), week)
                if key in score_lookup.index:
                    score_row = score_lookup.loc[key]
                    if isinstance(score_row, pd.DataFrame): score_row = score_row.iloc[0]
                    count += float(score_row["available_score_count"]); score_sum += float(score_row["score_sum"])
                    weighted_num += float(score_row["weighted_score_numerator"]); weight_sum += float(score_row["weight_sum"])
                sequence[index, week, CHANNELS.index("available_score_count")] = count
                sequence[index, week, CHANNELS.index("cumulative_mean_score")] = score_sum / count if count else 0.0
                sequence[index, week, CHANNELS.index("cumulative_weighted_score")] = weighted_num / weight_sum if weight_sum else 0.0
                sequence[index, week, CHANNELS.index("score_missing_mask")] = 0.0 if count else 1.0
                if key in last_lookup.index: last_activity = float(last_lookup.loc[key])
                if sequence[index, week, CHANNELS.index("total_clicks")] == 0: inactive += 1
                week_end = min((week + 1) * 7, int(row["cutoff_day"]))
                sequence[index, week, CHANNELS.index("days_since_last_vle_activity")] = float(week_end if last_activity is None else max(0, week_end - 1 - last_activity))
                sequence[index, week, CHANNELS.index("weeks_without_activity")] = float(inactive)

        aggregate, flat = _aggregate_features(sequence, cohort["valid_sequence_length"].to_numpy())
        aggregate.insert(0, "record_id", cohort["record_id"].to_numpy())
        flat.insert(0, "record_id", cohort["record_id"].to_numpy())
        cohort_path = processed_root / "cohorts" / f"{forecast_id}.parquet"
        target_path = processed_root / "targets" / f"{forecast_id}.parquet"
        sequence_path = processed_root / "sequences" / f"{forecast_id}.npz"
        aggregate_path = processed_root / "aggregated" / f"{forecast_id}.parquet"
        flat_path = processed_root / "flat" / f"{forecast_id}.parquet"
        cohort.to_parquet(cohort_path, index=False); targets.to_parquet(target_path, index=False)
        np.savez_compressed(sequence_path, record_ids=cohort["record_id"].astype(str).to_numpy(dtype="U64"), sequence=sequence, valid_lengths=cohort["valid_sequence_length"].to_numpy(int), padding_mask=np.arange(max_weeks)[None, :] < cohort["valid_sequence_length"].to_numpy(int)[:, None], channel_order=np.array(CHANNELS))
        aggregate.to_parquet(aggregate_path, index=False); flat.to_parquet(flat_path, index=False)
        target_hash = semantic_sha256(targets.sort_values("record_id").to_dict("records"))
        feature_contract_hash = semantic_sha256({"channels": CHANNELS, "forecast": forecast_id, "cutoff": "date < cutoff_day", "static": protocol["study_c"]["static_features"]})
        manifest = {"forecast_id": forecast_id, "forecast_fraction": FORECASTS[forecast_id], "cutoff_rule": "floor(length*fraction); events 0<=date<cutoff", "row_count": len(cohort), "cohort_hash": semantic_sha256(sorted(cohort["record_id"].tolist())), "target_hash": target_hash, "feature_contract_hash": feature_contract_hash, "source_hashes": {key: value["sha256"] for key, value in protocol["sources"].items() if key.startswith("oulad")}, "sequence_length": max_weeks, "channel_order": CHANNELS, "channel_order_hash": semantic_sha256(CHANNELS), "dtype": "float32", "padding_policy": "right_zero_padding_with_boolean_mask", "missing_policy": "score_zero_plus_missing_mask; train-only imputation downstream", "checksums": {"cohort": sha256_file(cohort_path), "target": sha256_file(target_path), "sequence": sha256_file(sequence_path), "aggregated": sha256_file(aggregate_path), "flat": sha256_file(flat_path)}}
        (processed_root / "manifests" / f"{forecast_id}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifests[forecast_id] = manifest
        flow_rows.append({"forecast_id": forecast_id, **flow, "at_risk": int(targets["target_at_risk"].sum()), "not_at_risk": int((1 - targets["target_at_risk"]).sum())})
    pd.DataFrame(flow_rows).to_csv(processed_root / "cohort_flow.csv", index=False)
    return {"status": "PASS", "manifests": manifests, "cohort_flow": flow_rows, "channels": CHANNELS}
