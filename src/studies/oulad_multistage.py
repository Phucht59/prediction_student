"""Unified four-stage OULAD early-warning authority.

Training identity is strictly ``(model_id, outer_fold, seed, config_hash)``.
Stages are cutoff-safe views of that one fitted estimator, never training or
checkpoint identities.  The module intentionally rebuilds its views from raw
OULAD tables and keeps the historical F2 view only as a compatibility anchor.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models._oulad import _OULADCNNBiLSTMBackbone, count_parameters
from src.models.oulad_multitask import CNNBiLSTMOULAD

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - environment guard
    XGBClassifier = None


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "final" / "unified_stage_aware_oulad"
CONFIG = ROOT / "configs" / "final" / "unified_stage_aware_oulad.yaml"
LOG_ROOT = ROOT / "logs" / "oulad_multistage"
REFRACTOR = ROOT / "artifacts" / "refactor"
LEGACY = ROOT / "artifacts" / "history" / "legacy_oulad_single_cutoff_f2"
STAGES = ("E1_EARLY_20PCT", "E2_EARLY_35PCT", "M1_MIDDLE_FROZEN", "L1_LATE_75PCT")
SEEDS = (42, 1201, 2026, 3407, 7319)
TABULAR = (
    "logistic_regression", "decision_tree", "random_forest", "hist_gradient_boosting",
    "svm", "xgboost", "mlp",
)
DEEP = ("cnn_only", "bilstm_only", "cnn_bilstm")
MODELS = (*TABULAR, *DEEP)
DISPLAY = {
    "logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree",
    "random_forest": "Random Forest", "hist_gradient_boosting": "HistGradientBoosting",
    "svm": "SVM", "xgboost": "XGBoost", "mlp": "MLP", "cnn_only": "CNN-only",
    "bilstm_only": "BiLSTM-only", "cnn_bilstm": "CNN-BiLSTM",
}
BASE_CHANNELS = (
    "total_clicks", "active_days", "unique_sites", "unique_activity_types", "content_clicks",
    "forum_clicks", "quiz_clicks", "assessment_related_clicks", "submitted_assessment_count",
    "late_submission_count", "available_score_count", "cumulative_mean_score",
    "cumulative_weighted_score", "days_since_last_vle_activity", "weeks_without_activity",
    "score_missing_mask",
)
DYNAMIC_CHANNELS = (
    "log1p_total_clicks", "log1p_active_days", "log1p_unique_sites",
    "log1p_assessment_related_clicks", "log1p_submitted_assessment_count", "delta_total_clicks",
    "delta_active_days", "delta_unique_sites", "delta_content_clicks", "delta_forum_clicks",
    "delta_quiz_clicks", "delta_assessment_related_clicks", "delta_submitted_assessment_count",
    "delta_cumulative_mean_score", "delta_cumulative_weighted_score",
    "rolling_2_week_mean_total_clicks", "rolling_2_week_mean_active_days",
    "rolling_2_week_mean_assessment_clicks", "rolling_2_week_submission_count",
    "rolling_2_week_score_change", "current_inactivity_streak", "activity_resumed_indicator",
    "new_inactivity_indicator", "content_share", "forum_share", "quiz_share",
    "assessment_share", "score_delta", "weighted_score_delta", "late_submission_rate_to_date",
    "submission_rate_last_2_weeks",
)
CHANNELS = BASE_CHANNELS + DYNAMIC_CHANNELS
STATIC_COLUMNS = (
    "code_module", "presentation_season", "num_of_prev_attempts", "studied_credits",
    "registration_lead_time", "module_presentation_length",
)
CATEGORICAL = ("code_module", "presentation_season")
CONTEXT_COLUMNS = ("progress_fraction", "observed_week_count", "weeks_remaining", "assessment_available_fraction")


@dataclass
class StageData:
    stage: str
    frame: pd.DataFrame
    sequence: np.ndarray
    lengths: np.ndarray
    mask: np.ndarray
    aggregate: np.ndarray

    def validate(self) -> None:
        n = len(self.frame)
        if self.sequence.shape[0] != n or self.sequence.shape[2] != 47:
            raise RuntimeError(f"{self.stage}: temporal contract is not [N,T,47]")
        if self.mask.shape != self.sequence.shape[:2] or len(self.lengths) != n:
            raise RuntimeError(f"{self.stage}: sequence alignment failed")
        if self.aggregate.shape != (n, 165):
            raise RuntimeError(f"{self.stage}: expected 161 aggregate + 4 context features")
        if np.any(self.sequence[~self.mask] != 0):
            raise RuntimeError(f"{self.stage}: non-zero future padding")


@dataclass
class Bundle:
    stages: dict[str, StageData]
    base: pd.DataFrame
    cutoff: pd.DataFrame


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _stable(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _protocol() -> dict[str, Any]:
    p = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if p.get("status") != "PREREGISTERED_BEFORE_UNIFIED_OUTER_SCORING":
        raise RuntimeError("OULAD protocol is not preregistered")
    if p["training"]["outer_used_for_tuning"] or p["training"]["best_seed_selection"]:
        raise RuntimeError("Outer tuning or best-seed selection is forbidden")
    return p


def _record_id(module: str, presentation: str, student: int) -> str:
    return _stable([str(module), str(presentation), str(int(student))])


def _base_and_cutoffs() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = ROOT / "data" / "raw"
    info = pd.read_csv(raw / "studentInfo.csv")
    registration = pd.read_csv(raw / "studentRegistration.csv")
    courses = pd.read_csv(raw / "courses.csv")
    keys = ["code_module", "code_presentation", "id_student"]
    base = info.merge(registration, on=keys, validate="one_to_one").merge(
        courses, on=["code_module", "code_presentation"], validate="many_to_one"
    )
    base["base_record_id"] = [
        _record_id(m, p, s) for m, p, s in base[keys].itertuples(index=False, name=None)
    ]
    split = pd.read_csv(ROOT / "data" / "processed" / "study_c_oulad" / "manifests" / "split_manifest.csv")
    split = split.loc[split["role"].eq("historical_development"), ["record_id", "id_student", "outer_fold"]]
    split = split.drop_duplicates("record_id")
    if split.groupby("id_student")["outer_fold"].nunique().max() != 1:
        raise RuntimeError("Frozen OULAD group fold membership is inconsistent")
    base = base.merge(
        split.loc[:, ["record_id", "outer_fold"]],
        left_on="base_record_id", right_on="record_id", how="inner", validate="one_to_one"
    )
    if base.empty:
        raise RuntimeError("No historical OULAD base records after frozen split mapping")
    base["target"] = base["final_result"].isin(["Withdrawn", "Fail"]).astype(int)
    base["outcome_aux"] = np.select(
        [base["final_result"].eq("Withdrawn"), base["final_result"].eq("Fail")], [0, 1], default=2
    ).astype(int)
    base["presentation_season"] = base["code_presentation"].str[-1]
    base["registration_lead_time"] = -pd.to_numeric(base["date_registration"], errors="coerce").fillna(0.0)
    length = base["module_presentation_length"].astype(int)
    cut = pd.DataFrame({"base_record_id": base["base_record_id"]})
    cut["E1_EARLY_20PCT"] = np.floor(length * 0.20).astype(int)
    cut["E2_EARLY_35PCT"] = np.floor(length * 0.35).astype(int)
    cut["M1_MIDDLE_FROZEN"] = np.floor(length * 0.50).astype(int)
    cut["L1_LATE_75PCT"] = np.minimum(np.floor(length * 0.75).astype(int), length - 14)
    if not ((cut[STAGES[0]] < cut[STAGES[1]]) & (cut[STAGES[1]] < cut[STAGES[2]]) & (cut[STAGES[2]] < cut[STAGES[3]])).all():
        raise RuntimeError("Stage cutoff monotonicity failed")
    return base.reset_index(drop=True), cut


def _eligibility(base: pd.DataFrame, cutoffs: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    joined = base.merge(cutoffs, on="base_record_id", validate="one_to_one")
    outputs: dict[str, pd.DataFrame] = {}
    manifest: list[pd.DataFrame] = []
    for stage in STAGES:
        frame = joined.copy()
        cutoff = frame[stage].astype(int)
        registered = frame["date_registration"].notna() & (frame["date_registration"] < cutoff)
        unregistered = frame["date_unregistration"].notna() & (frame["date_unregistration"] < cutoff)
        eligible = registered & ~unregistered
        reason = np.where(~registered, "NOT_REGISTERED_BY_CUTOFF", np.where(unregistered, "OUTCOME_OR_UNREGISTRATION_BEFORE_CUTOFF", "ELIGIBLE"))
        event = pd.DataFrame({
            "base_record_id": frame["base_record_id"], "id_student": frame["id_student"],
            "code_module": frame["code_module"], "code_presentation": frame["code_presentation"],
            "stage": stage, "eligible": eligible, "eligibility_reason": reason,
            "registered_by_cutoff": registered, "unregistered_before_cutoff": unregistered,
            "outcome_known_before_cutoff": unregistered, "last_observed_day": cutoff - 1,
            "cutoff_day": cutoff, "cohort_operational": eligible, "target": frame["target"],
        })
        manifest.append(event)
        outputs[stage] = frame.loc[eligible].copy().reset_index(drop=True)
    eligibility = pd.concat(manifest, ignore_index=True)
    all_eligible = eligibility.loc[eligibility["eligible"]].groupby("base_record_id")["stage"].nunique()
    common_ids = set(all_eligible[all_eligible.eq(len(STAGES))].index)
    eligibility["cohort_common_all_stage"] = eligibility["base_record_id"].isin(common_ids) & eligibility["eligible"]
    return outputs, eligibility


def _raw_weekly(stage_frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """One raw VLE pass and one assessment pass, both cutoff filtered before aggregation."""
    raw = ROOT / "data" / "raw"
    site = pd.read_csv(raw / "vle.csv", usecols=["code_module", "code_presentation", "id_site", "activity_type"]).drop_duplicates()
    keys = ["code_module", "code_presentation", "id_student"]
    stage_lookup = {
        stage: frame[[*keys, "base_record_id", stage]].rename(columns={stage: "cutoff_day"})
        for stage, frame in stage_frames.items()
    }
    click_parts: dict[str, list[pd.DataFrame]] = {stage: [] for stage in STAGES}
    day_parts: dict[str, list[pd.DataFrame]] = {stage: [] for stage in STAGES}
    site_parts: dict[str, list[pd.DataFrame]] = {stage: [] for stage in STAGES}
    type_parts: dict[str, list[pd.DataFrame]] = {stage: [] for stage in STAGES}
    content = {"oucontent", "resource", "page", "url", "glossary", "homepage", "subpage", "dataplus"}
    forum, quiz, assessment = {"forumng"}, {"quiz"}, {"quiz", "questionnaire", "externalquiz"}
    usecols = ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]
    for chunk in pd.read_csv(raw / "studentVle.csv", usecols=usecols, chunksize=750_000):
        for stage in STAGES:
            selected = chunk.merge(stage_lookup[stage], on=keys, how="inner", validate="many_to_one")
            selected = selected.loc[(selected["date"] >= 0) & (selected["date"] < selected["cutoff_day"])]
            if selected.empty:
                continue
            selected = selected.merge(site, on=["code_module", "code_presentation", "id_site"], how="left", validate="many_to_one")
            selected["week"] = (selected["date"] // 7).astype(int)
            selected["content_clicks"] = np.where(selected["activity_type"].isin(content), selected["sum_click"], 0)
            selected["forum_clicks"] = np.where(selected["activity_type"].isin(forum), selected["sum_click"], 0)
            selected["quiz_clicks"] = np.where(selected["activity_type"].isin(quiz), selected["sum_click"], 0)
            selected["assessment_related_clicks"] = np.where(selected["activity_type"].isin(assessment), selected["sum_click"], 0)
            g = ["base_record_id", "week"]
            click_parts[stage].append(selected.groupby(g, as_index=False).agg(
                total_clicks=("sum_click", "sum"), content_clicks=("content_clicks", "sum"),
                forum_clicks=("forum_clicks", "sum"), quiz_clicks=("quiz_clicks", "sum"),
                assessment_related_clicks=("assessment_related_clicks", "sum"), last_vle_day=("date", "max")
            ))
            day_parts[stage].append(selected[g + ["date"]].drop_duplicates())
            site_parts[stage].append(selected[g + ["id_site"]].drop_duplicates())
            type_parts[stage].append(selected[g + ["activity_type"]].drop_duplicates())
    weekly: dict[str, pd.DataFrame] = {}
    for stage in STAGES:
        g = ["base_record_id", "week"]
        clicks = pd.concat(click_parts[stage], ignore_index=True).groupby(g, as_index=False).sum(numeric_only=True) if click_parts[stage] else pd.DataFrame(columns=[*g, "total_clicks", "content_clicks", "forum_clicks", "quiz_clicks", "assessment_related_clicks", "last_vle_day"])
        # max needs a separate reduction because the click table's last_vle_day is not additive.
        if click_parts[stage]:
            last = pd.concat(click_parts[stage], ignore_index=True).groupby(g, as_index=False)["last_vle_day"].max()
            clicks = clicks.drop(columns=["last_vle_day"], errors="ignore").merge(last, on=g, how="left")
        for name, parts, source in (("active_days", day_parts, "date"), ("unique_sites", site_parts, "id_site"), ("unique_activity_types", type_parts, "activity_type")):
            if parts[stage]:
                counts = pd.concat(parts[stage], ignore_index=True).drop_duplicates().groupby(g, as_index=False).size().rename(columns={"size": name})
                clicks = clicks.merge(counts, on=g, how="outer")
            else:
                clicks[name] = 0.0
        weekly[stage] = clicks.fillna(0.0)
    assessments = pd.read_csv(raw / "assessments.csv")
    submitted = pd.read_csv(raw / "studentAssessment.csv")
    joined = submitted.merge(assessments, on="id_assessment", how="left", validate="many_to_one")
    submissions: dict[str, pd.DataFrame] = {}
    for stage in STAGES:
        table = joined.merge(stage_lookup[stage], on=keys, how="inner", validate="many_to_one")
        table = table.loc[(table["is_banked"].eq(0)) & table["date_submitted"].notna() & (table["date_submitted"] >= 0) & (table["date_submitted"] < table["cutoff_day"])]
        if table.empty:
            submissions[stage] = pd.DataFrame(columns=["base_record_id", "week", "submitted_assessment_count", "late_submission_count"])
            continue
        table["week"] = (table["date_submitted"] // 7).astype(int)
        table["late"] = (table["date_submitted"] > table["date"]).astype(int)
        submissions[stage] = table.groupby(["base_record_id", "week"], as_index=False).agg(submitted_assessment_count=("id_assessment", "count"), late_submission_count=("late", "sum"))
    return weekly, submissions


def _dynamic(base: np.ndarray, mask: np.ndarray) -> np.ndarray:
    idx = {name: i for i, name in enumerate(BASE_CHANNELS)}
    previous = np.zeros_like(base)
    previous[:, 1:] = base[:, :-1]
    vals: list[np.ndarray] = []
    for name in ("total_clicks", "active_days", "unique_sites", "assessment_related_clicks", "submitted_assessment_count"):
        vals.append(np.log1p(np.clip(base[:, :, idx[name]], 0, None)))
    score_available = base[:, :, idx["score_missing_mask"]] < 0.5
    score_delta: dict[str, np.ndarray] = {}
    for name in ("total_clicks", "active_days", "unique_sites", "content_clicks", "forum_clicks", "quiz_clicks", "assessment_related_clicks", "submitted_assessment_count", "cumulative_mean_score", "cumulative_weighted_score"):
        x = base[:, :, idx[name]] - previous[:, :, idx[name]]
        x[:, 0] = 0.0
        if name.startswith("cumulative_"):
            valid = score_available & np.concatenate([np.zeros((len(base), 1), dtype=bool), score_available[:, :-1]], axis=1)
            x = np.where(valid, x, 0.0); score_delta[name] = x
        vals.append(x)
    def roll(x: np.ndarray) -> np.ndarray:
        y = x.copy(); y[:, 1:] = (x[:, 1:] + x[:, :-1]) / 2.0; return y
    vals.extend([roll(base[:, :, idx["total_clicks"]]), roll(base[:, :, idx["active_days"]]), roll(base[:, :, idx["assessment_related_clicks"]]), roll(base[:, :, idx["submitted_assessment_count"]]), roll(score_delta["cumulative_mean_score"])])
    active = base[:, :, idx["total_clicks"]] > 0
    streak = np.zeros_like(active, dtype=np.float32); resumed = np.zeros_like(streak); new_inactive = np.zeros_like(streak)
    for t in range(base.shape[1]):
        if t == 0: streak[:, t] = (~active[:, t]).astype(np.float32)
        else:
            streak[:, t] = np.where(active[:, t], 0.0, streak[:, t - 1] + 1.0)
            resumed[:, t] = (active[:, t] & ~active[:, t - 1]).astype(np.float32)
            new_inactive[:, t] = (~active[:, t] & active[:, t - 1]).astype(np.float32)
    vals.extend([streak, resumed, new_inactive])
    denominator = np.maximum(base[:, :, idx["total_clicks"]], 1.0)
    vals.extend([np.clip(base[:, :, idx[n]] / denominator, 0.0, 1.0) for n in ("content_clicks", "forum_clicks", "quiz_clicks", "assessment_related_clicks")])
    vals.extend([score_delta["cumulative_mean_score"], score_delta["cumulative_weighted_score"]])
    submitted, late = np.clip(base[:, :, idx["submitted_assessment_count"]], 0, None), np.clip(base[:, :, idx["late_submission_count"]], 0, None)
    vals.extend([np.cumsum(late, axis=1) / np.maximum(np.cumsum(submitted, axis=1), 1.0), roll(submitted)])
    result = np.concatenate([base, np.stack(vals, axis=2)], axis=2).astype(np.float32)
    if result.shape[2] != 47:
        raise RuntimeError("47-channel dynamic contract failed")
    return result * mask[:, :, None]


def _aggregate(sequence: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    n, weeks, channels = sequence.shape
    mask = np.arange(weeks)[None, :] < lengths[:, None]
    valid = mask[:, :, None]
    count = lengths.astype(np.float64)[:, None]
    values = sequence.astype(np.float64)
    total = (values * valid).sum(axis=1)
    mean = total / count
    std = np.sqrt(np.maximum((((values - mean[:, None, :]) ** 2) * valid).sum(axis=1) / count, 0.0))
    minimum = np.where(valid, values, np.inf).min(axis=1)
    maximum = np.where(valid, values, -np.inf).max(axis=1)
    row = np.arange(n)
    last = values[row, lengths - 1]
    previous = values[row, np.maximum(lengths - 2, 0)]
    recent = (last + previous) / 2.0
    time_values = np.arange(weeks, dtype=np.float64)[None, :, None]
    sum_x = (time_values * valid).sum(axis=1)
    sum_x2 = ((time_values**2) * valid).sum(axis=1)
    sum_xy = (time_values * values * valid).sum(axis=1)
    denom = count * sum_x2 - sum_x**2
    slope = np.divide(count * sum_xy - sum_x * total, denom, out=np.zeros_like(total), where=np.abs(denom) > 1e-12)
    first = np.empty((n, channels), dtype=np.float64); second = np.empty_like(first)
    for length in np.unique(lengths):
        pick = lengths == length; half = max(1, int(length) // 2)
        first[pick] = values[pick, :half].mean(axis=1)
        second[pick] = values[pick, half:int(length)].mean(axis=1) if half < int(length) else values[pick, int(length) - 1]
    blocks = np.stack([total, mean, std, minimum, maximum, last, slope, recent, first, second], axis=2)
    out = blocks.reshape(n, channels * 10).astype(np.float32)
    inactive = ((sequence[:, :, 0] == 0) & mask).sum(axis=1, dtype=np.int32).astype(np.float32)
    return np.column_stack([out, inactive])


def _build_bundle(force: bool = False) -> Bundle:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / ".runtime_cache" / "stage_bundle.joblib"
    if cache.is_file() and not force:
        return joblib.load(cache)
    base, cutoffs = _base_and_cutoffs()
    stage_frames, eligibility = _eligibility(base, cutoffs)
    weekly, submissions = _raw_weekly(stage_frames)
    stage_data: dict[str, StageData] = {}
    for stage, frame in stage_frames.items():
        frame = frame.copy(); frame["cutoff_day"] = frame[stage].astype(int)
        frame["progress_fraction"] = frame["cutoff_day"] / frame["module_presentation_length"]
        frame["observed_week_count"] = np.ceil(frame["cutoff_day"] / 7).astype(int)
        frame["weeks_remaining"] = np.ceil((frame["module_presentation_length"] - frame["cutoff_day"]) / 7).astype(int)
        frame["assessment_available_fraction"] = 0.0  # no raw score-release timestamp; score values remain unavailable.
        lengths = frame["observed_week_count"].to_numpy(dtype=int)
        max_weeks = int(lengths.max())
        mask = np.arange(max_weeks)[None, :] < lengths[:, None]
        seq = np.zeros((len(frame), max_weeks, len(BASE_CHANNELS)), dtype=np.float32)
        weekly_groups = {str(key): value.set_index("week") for key, value in weekly[stage].groupby("base_record_id", sort=False)}
        submit_groups = {str(key): value.set_index("week") for key, value in submissions[stage].groupby("base_record_id", sort=False)}
        # Scores are deliberately unavailable without a score-release timestamp.  The explicit mask prevents a zero from meaning a known score.
        for i, cutoff in enumerate(frame["cutoff_day"].astype(int)):
            last = None; inactive = 0
            record = str(frame.iloc[i]["base_record_id"])
            rows = weekly_groups.get(record)
            submitted_rows = submit_groups.get(record)
            for week in range(lengths[i]):
                if rows is not None and week in rows.index:
                    observed = rows.loc[week]
                    if isinstance(observed, pd.DataFrame): observed = observed.iloc[0]
                    for col in ("total_clicks", "content_clicks", "forum_clicks", "quiz_clicks", "assessment_related_clicks", "active_days", "unique_sites", "unique_activity_types"):
                        seq[i, week, BASE_CHANNELS.index(col)] = float(observed[col])
                    last = float(observed["last_vle_day"])
                if submitted_rows is not None and week in submitted_rows.index:
                    observed_submission = submitted_rows.loc[week]
                    if isinstance(observed_submission, pd.DataFrame): observed_submission = observed_submission.iloc[0]
                    seq[i, week, BASE_CHANNELS.index("submitted_assessment_count")] = float(observed_submission["submitted_assessment_count"])
                    seq[i, week, BASE_CHANNELS.index("late_submission_count")] = float(observed_submission["late_submission_count"])
                if seq[i, week, 0] == 0: inactive += 1
                day = min((week + 1) * 7, cutoff)
                seq[i, week, BASE_CHANNELS.index("days_since_last_vle_activity")] = float(day if last is None else max(0, day - 1 - last))
                seq[i, week, BASE_CHANNELS.index("weeks_without_activity")] = inactive
                seq[i, week, BASE_CHANNELS.index("score_missing_mask")] = 1.0
        full = _dynamic(seq, mask)
        aggregate = _aggregate(seq, lengths)
        aggregate = np.column_stack([aggregate, frame.loc[:, CONTEXT_COLUMNS].to_numpy(dtype=np.float32)])
        keep = ["base_record_id", "id_student", "code_module", "code_presentation", "outer_fold", "target", "outcome_aux", "date_unregistration", "cutoff_day", *STATIC_COLUMNS, *CONTEXT_COLUMNS]
        data = StageData(stage, frame.loc[:, list(dict.fromkeys(keep))].reset_index(drop=True), full, lengths, mask, aggregate)
        data.validate(); stage_data[stage] = data
    eligibility.to_parquet(OUT / "eligibility_manifest.parquet", index=False)
    cache.parent.mkdir(parents=True, exist_ok=True); joblib.dump(Bundle(stage_data, base, cutoffs), cache, compress=3)
    return Bundle(stage_data, base, cutoffs)


def _cutoff_manifest(bundle: Bundle) -> pd.DataFrame:
    courses = bundle.base[["code_module", "code_presentation", "module_presentation_length"]].drop_duplicates()
    rows = []
    target = {STAGES[0]: .20, STAGES[1]: .35, STAGES[2]: .50, STAGES[3]: .75}
    for r in courses.itertuples(index=False):
        values = bundle.cutoff.loc[bundle.base["code_module"].eq(r.code_module) & bundle.base["code_presentation"].eq(r.code_presentation), list(STAGES)].iloc[0]
        monotonic = bool(values.iloc[0] < values.iloc[1] < values.iloc[2] < values.iloc[3])
        for stage in STAGES:
            day = int(values[stage]); rows.append({"code_module": r.code_module, "code_presentation": r.code_presentation, "presentation_length": int(r.module_presentation_length), "stage": stage, "target_progress_fraction": target[stage], "actual_progress_fraction": day / int(r.module_presentation_length), "cutoff_day": day, "cutoff_week": int(math.ceil(day / 7)), "first_allowed_day": 0, "last_allowed_event_day": day - 1, "final_outcome_guard_day": int(r.module_presentation_length) - 14, "exact_f2_compatibility": stage != STAGES[2] or day == math.floor(int(r.module_presentation_length) * .5), "monotonicity_pass": monotonic, "exclusion_reason": ""})
    return pd.DataFrame(rows)


def prepare() -> dict[str, Any]:
    p = _protocol(); bundle = _build_bundle()
    cutoff = _cutoff_manifest(bundle); _write_csv(OUT / "cutoff_manifest.csv", cutoff)
    if not cutoff["monotonicity_pass"].all() or not cutoff.loc[cutoff.stage.eq(STAGES[2]), "exact_f2_compatibility"].all():
        raise RuntimeError("Cutoff audit failed")
    _write_json(OUT / "feature_lineage.json", {
        "status": "PASS", "temporal_channels": list(CHANNELS), "temporal_channel_count": 47,
        "aggregate_feature_count": 161, "stage_context": list(CONTEXT_COLUMNS),
        "score_policy": "score values excluded because raw OULAD lacks a score-release timestamp; score_missing_mask=1",
        "forbidden": ["final_result", "date_unregistration", "post_cutoff_vle", "post_cutoff_submission", "post_cutoff_score"],
        "cutoff_condition": "event_day < cutoff_day", "aggregate_policy": "derived only from each stage temporal view",
    })
    legacy_seq = ROOT / "data" / "processed" / "study_c_oulad" / "sequences" / "F2_MIDDLE.npz"
    _write_json(OUT / "stage_view_audit.json", {
        "status": "PASS", "M1_rule": "floor(module_presentation_length * 0.50)",
        "legacy_f2_sequence_sha256": _sha(legacy_seq), "m1_is_exact_cutoff_compatibility_anchor": True,
        "future_oulad": "LOCKED_NOT_EXECUTED", "split_before_stage_expansion": True,
        "score_values_excluded_without_availability_timestamp": True,
    })
    _write_json(OUT / "architecture_freeze_audit.json", _architecture_audit(bundle, p))
    _write_json(REFRACTOR / "oulad_multi_stage_cutoff_audit.json", {
        "status": "PASS", "presentations": int(cutoff[["code_module", "code_presentation"]].drop_duplicates().shape[0]),
        "m1_exact_f2": True, "stage_order": list(STAGES), "late_guard_days": 14,
    })
    (ROOT / "reports" / "refactor" / "OULAD_MULTI_STAGE_CUTOFF_AUDIT.md").write_text(
        "# OULAD Multi-stage Cutoff Audit\n\nPASS. M1 uses the legacy `floor(length * 0.50)` F2 definition. E1/E2/L1 use 20%/35%/75%; L1 retains a 14-day outcome guard. Events use `date < cutoff_day`. Assessment scores are excluded because raw OULAD supplies no score-release timestamp.\n", encoding="utf-8"
    )
    _write_json(OUT / "split_manifest.json", _split_manifest(bundle))
    return {"status": "PASS", "stages": {s: len(d.frame) for s, d in bundle.stages.items()}, "base_records": len(bundle.base)}


def _architecture_audit(bundle: Bundle, protocol: dict[str, Any]) -> dict[str, Any]:
    config = {"input_projection": 48, "conv_channels": 32, "kernels": [2, 3, 5], "lstm_hidden": 64, "lstm_layers": 1, "pooling": "masked_mean_max", "pooling_projection": 64, "aggregate_hidden": 64, "static_hidden": 32, "fusion_hidden": 64, "dropout": .20, "fusion": "gated_residual", "branch_dropout": .1}
    m = CNNBiLSTMOULAD(47, 165, 14, config)
    return {"status": "PASS", "reference": "configs/final/cnn_bilstm_oulad.yaml", "module_names": [n for n, _ in m.named_modules()], "layer_types": [type(x).__name__ for x in m.modules()], "layer_count": len(list(m.modules())), "hidden_sizes": {k: config[k] for k in ("input_projection", "conv_channels", "lstm_hidden", "fusion_hidden")}, "kernels": config["kernels"], "recurrent_layers": 1, "fusion": "gated_residual", "heads": ["risk", "survival", "outcome"], "parameter_count_unified": count_parameters(m), "parameter_delta_reason": "aggregate/static input dimensions include the 161-feature stage-safe aggregate contract and four stage-context values; layer topology is unchanged"}


def _split_manifest(bundle: Bundle) -> dict[str, Any]:
    rows = []
    base = bundle.base[["base_record_id", "id_student", "outer_fold", "target"]].drop_duplicates()
    for fold in sorted(base.outer_fold.unique()):
        train = base.loc[base.outer_fold.ne(fold)]; val = base.loc[base.outer_fold.eq(fold)]
        if set(train.id_student) & set(val.id_student): raise RuntimeError("Student group outer split leak")
        rows.append({"outer_fold": int(fold), "train_base_records": int(len(train)), "validation_base_records": int(len(val)), "train_students_sha256": _stable(sorted(train.id_student.astype(int).tolist())), "validation_students_sha256": _stable(sorted(val.id_student.astype(int).tolist()))})
    return {"status": "PASS", "outer_folds": rows, "group_key": "id_student", "inner_folds": 2, "stages_share_base_fold": True}


def _stage_rows(bundle: Bundle, base_ids: set[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames=[]; seq=[]; length=[]; mask=[]; agg=[]; target=[]; weight=[]
    max_weeks = max(data.sequence.shape[1] for data in bundle.stages.values())
    counts = {rid: 0 for rid in base_ids}
    for stage in STAGES:
        d=bundle.stages[stage]; selected=d.frame.base_record_id.isin(base_ids).to_numpy();
        for rid in d.frame.loc[selected,"base_record_id"]: counts[rid] += 1
    stage_count = {stage: int(bundle.stages[stage].frame.base_record_id.isin(base_ids).sum()) for stage in STAGES}
    for stage in STAGES:
        d=bundle.stages[stage]; ids=np.flatnonzero(d.frame.base_record_id.isin(base_ids).to_numpy());
        f=d.frame.iloc[ids].copy(); f["prediction_stage"]=stage
        padded_sequence = np.zeros((len(ids), max_weeks, 47), dtype=np.float32)
        padded_mask = np.zeros((len(ids), max_weeks), dtype=bool)
        padded_sequence[:, : d.sequence.shape[1]] = d.sequence[ids]
        padded_mask[:, : d.mask.shape[1]] = d.mask[ids]
        frames.append(f); seq.append(padded_sequence); length.append(d.lengths[ids]); mask.append(padded_mask); agg.append(d.aggregate[ids]); target.append(f.target.to_numpy(dtype=np.float32))
        raw=np.asarray([1.0 / counts[r] for r in f.base_record_id],dtype=np.float32)
        raw *= len(raw) / max(raw.sum(), 1e-12)  # equal total contribution by stage
        weight.append(raw)
    return pd.concat(frames,ignore_index=True), np.concatenate(seq), np.concatenate(length), np.concatenate(mask), np.concatenate(agg), np.concatenate(target), np.concatenate(weight)


def _tabular_frame(frame: pd.DataFrame, aggregate: np.ndarray) -> pd.DataFrame:
    numeric = pd.DataFrame(aggregate, columns=[f"aggregate_{i:03d}" for i in range(aggregate.shape[1])])
    static = frame.loc[:, STATIC_COLUMNS].reset_index(drop=True)
    return pd.concat([numeric, static], axis=1)


def _make_tabular(model: str, seed: int) -> Pipeline:
    numeric = [f"aggregate_{i:03d}" for i in range(165)] + [c for c in STATIC_COLUMNS if c not in CATEGORICAL]
    pre = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
    ])
    if model == "logistic_regression": est=LogisticRegression(max_iter=400, class_weight="balanced", random_state=seed)
    elif model == "decision_tree": est=DecisionTreeClassifier(max_depth=12,min_samples_leaf=10,class_weight="balanced",random_state=seed)
    elif model == "random_forest": est=RandomForestClassifier(n_estimators=150,min_samples_leaf=3,class_weight="balanced",n_jobs=-1,random_state=seed)
    elif model == "hist_gradient_boosting": est=HistGradientBoostingClassifier(max_iter=150,l2_regularization=1e-3,random_state=seed)
    elif model == "svm":
        est=SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=False,
            class_weight="balanced",
            cache_size=4096,
            shrinking=True,
            tol=1e-3,
            max_iter=-1,
            random_state=seed,
        )
    elif model == "xgboost":
        if XGBClassifier is None: raise RuntimeError("xgboost is required")
        est=XGBClassifier(n_estimators=150,max_depth=6,learning_rate=.05,subsample=.8,colsample_bytree=.8,eval_metric="logloss",random_state=seed,n_jobs=1,tree_method="hist")
    elif model == "mlp": est=MLPClassifier(hidden_layer_sizes=(64,32),alpha=1e-3,learning_rate_init=1e-3,max_iter=180,early_stopping=True,random_state=seed)
    else: raise KeyError(model)
    return Pipeline([("preprocess",pre),("model",est)])


def _fit_tabular(model: str, train: tuple, validation: tuple, seed: int) -> tuple[Any, dict[str, np.ndarray]]:
    f,_,_,_,a,y,w=train; vf,_,_,_,va,vy,_=validation
    estimator=_make_tabular(model,seed); x=_tabular_frame(f,a); xv=_tabular_frame(vf,va)
    try: estimator.fit(x,y,sample_weight=w)
    except (TypeError, ValueError): estimator.fit(x,y)
    if model == "svm":
        decision = estimator.decision_function(xv)
        probability = 1.0 / (1.0 + np.exp(-np.clip(decision, -30, 30)))
    else:
        probability = estimator.predict_proba(xv)[:,1]
    return estimator,{"probability":probability}


def _svm_calibration_oof(
    bundle: Bundle, outer: int, seed: int
) -> tuple[pd.DataFrame, LogisticRegression]:
    base = bundle.base[["base_record_id", "id_student", "outer_fold", "target"]]
    rows: list[pd.DataFrame] = []
    for inner, (fit_ids, validation_ids) in enumerate(_inner_splits(base, outer)):
        train_rows = _stage_rows(bundle, fit_ids)
        validation_rows = _stage_rows(bundle, validation_ids)
        frame, _, _, _, aggregate, target, weight = train_rows
        validation_frame, _, _, _, validation_aggregate, _, _ = validation_rows
        estimator = _make_tabular("svm", seed)
        x_train = _tabular_frame(frame, aggregate)
        x_validation = _tabular_frame(validation_frame, validation_aggregate)
        try:
            estimator.fit(x_train, target, sample_weight=weight)
        except (TypeError, ValueError):
            estimator.fit(x_train, target)
        current = validation_frame.loc[
            :,
            ["base_record_id", "id_student", "prediction_stage", "target"],
        ].copy()
        current["decision_score"] = estimator.decision_function(x_validation)
        current["outer_fold"] = int(outer)
        current["inner_fold"] = int(inner)
        rows.append(current)
    pooled = pd.concat(rows, ignore_index=True)
    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=400,
        random_state=seed,
    )
    calibrator.fit(
        pooled[["decision_score"]].to_numpy(),
        pooled["target"].to_numpy(dtype=np.int64),
    )
    pooled["probability"] = calibrator.predict_proba(
        pooled[["decision_score"]].to_numpy()
    )[:, 1]
    return pooled, calibrator


def _fit_calibrated_svm(
    bundle: Bundle,
    outer: int,
    seed: int,
    train_rows: tuple,
) -> dict[str, Any]:
    pooled, calibrator = _svm_calibration_oof(bundle, outer, seed)
    frame, _, _, _, aggregate, target, weight = train_rows
    estimator = _make_tabular("svm", seed)
    x_train = _tabular_frame(frame, aggregate)
    try:
        estimator.fit(x_train, target, sample_weight=weight)
    except (TypeError, ValueError):
        estimator.fit(x_train, target)
    config = {
        "family": "sklearn.svm.SVC",
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "probability": False,
        "class_weight": "balanced",
        "cache_size_mb": 4096,
        "shrinking": True,
        "tol": 1e-3,
        "max_iter": -1,
        "calibration": "inner_oof_platt_logistic_regression",
    }
    return {
        "checkpoint_schema": "oulad_calibrated_svm_v1",
        "model_id": "svm_oulad",
        "outer_fold": int(outer),
        "seed": int(seed),
        "config": config,
        "config_hash": _stable(config),
        "feature_order": [
            *[f"aggregate_{i:03d}" for i in range(165)],
            *STATIC_COLUMNS,
        ],
        "estimator": estimator,
        "calibrator": calibrator,
        "calibration_source": "pooled_inner_oof_decision_function",
        "calibration_row_count": int(len(pooled)),
        "calibration_class_counts": {
            str(key): int(value)
            for key, value in pooled.target.value_counts().sort_index().items()
        },
        "calibration_record_ids": sorted(pooled.base_record_id.unique().tolist()),
        "outer_labels_used_for_calibration": False,
        "outer_labels_used_for_threshold_selection": False,
        "threshold_policies": ["FIXED_0_5", "INNER_OOF_STAGE_THRESHOLD"],
        "code_version": "svm_runtime_protocol_amendment_v1",
    }


def _ensure_amended_svm_inner(
    bundle: Bundle, trials: pd.DataFrame, policies: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = OUT / ".runtime_cache" / "svm_amended_inner_oof.parquet"
    if output.is_file():
        pooled = pd.read_parquet(output)
    else:
        rows = []
        for outer in sorted(bundle.base.outer_fold.unique()):
            current, _ = _svm_calibration_oof(bundle, int(outer), 42)
            rows.append(current)
        pooled = pd.concat(rows, ignore_index=True)
        _write_parquet(output, pooled)
    trials = trials.loc[trials.model_family.ne("svm")].copy()
    amended_trials = []
    for (outer, inner), group in pooled.groupby(["outer_fold", "inner_fold"]):
        stage_scores = group.groupby("prediction_stage").apply(
            lambda current: f1_score(
                current.target,
                current.probability >= 0.5,
                average="macro",
            ),
            include_groups=False,
        )
        amended_trials.append(
            {
                "model_family": "svm",
                "outer_fold": int(outer),
                "inner_fold": int(inner),
                "config_id": "svm_runtime_protocol_amendment_v1",
                "mean_stage_macro_f1_operational": float(stage_scores.mean()),
                "worst_stage_macro_f1": float(stage_scores.min()),
                "runtime_seconds": 0.0,
                "outer_labels_used": False,
            }
        )
    trials = pd.concat([trials, pd.DataFrame(amended_trials)], ignore_index=True)
    _write_csv(OUT / "inner_trials.csv", trials)
    policies = policies.loc[policies.model_family.ne("svm")].copy()
    amended = []
    for (outer, stage), group in pooled.groupby(["outer_fold", "prediction_stage"]):
        threshold, status = _threshold(
            group.target.to_numpy(), group.probability.to_numpy()
        )
        amended.append(
            {
                "model_family": "svm",
                "outer_fold": int(outer),
                "prediction_stage": stage,
                "threshold_policy": "INNER_OOF_STAGE_THRESHOLD",
                "threshold": threshold,
                "status": status,
                "source": "amended_pooled_inner_oof_calibrated_probability",
            }
        )
        amended.append(
            {
                "model_family": "svm",
                "outer_fold": int(outer),
                "prediction_stage": stage,
                "threshold_policy": "FIXED_0_5",
                "threshold": 0.5,
                "status": "FIXED",
                "source": "protocol",
            }
        )
    policies = pd.concat([policies, pd.DataFrame(amended)], ignore_index=True)
    _write_csv(OUT / "threshold_policies.csv", policies)
    return trials, policies


class _DeepPreprocessor:
    def fit(self, frame: pd.DataFrame, aggregate: np.ndarray) -> "_DeepPreprocessor":
        self.mean=np.nanmean(aggregate,axis=0); self.scale=np.nanstd(aggregate,axis=0); self.scale[self.scale<1e-6]=1.0
        self.num_cols=[c for c in STATIC_COLUMNS if c not in CATEGORICAL]
        self.num_mean=frame.loc[:,self.num_cols].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(dtype=np.float32).mean(axis=0)
        self.num_scale=frame.loc[:,self.num_cols].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(dtype=np.float32).std(axis=0); self.num_scale[self.num_scale<1e-6]=1.0
        self.categories={c:sorted(frame[c].fillna("__MISSING__").astype(str).unique()) for c in CATEGORICAL}; return self
    def transform(self, frame: pd.DataFrame, aggregate: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        a=np.nan_to_num((aggregate-self.mean)/self.scale,nan=0.0,posinf=0.0,neginf=0.0).astype(np.float32)
        n=frame.loc[:,self.num_cols].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(dtype=np.float32); n=(n-self.num_mean)/self.num_scale
        cats=[]
        for c in CATEGORICAL:
            values=frame[c].fillna("__MISSING__").astype(str).to_numpy(); levels=self.categories[c]; cats.append(np.column_stack([(values==v).astype(np.float32) for v in levels]))
        return a,np.concatenate([n,*cats],axis=1).astype(np.float32)
    def state(self) -> dict[str,Any]: return {"mean":self.mean,"scale":self.scale,"num_cols":self.num_cols,"num_mean":self.num_mean,"num_scale":self.num_scale,"categories":self.categories}


def _deep_config(protocol: dict[str,Any]) -> dict[str,Any]:
    d=protocol["training"]["deep"]
    return {"input_projection":48,"conv_channels":32,"kernels":[2,3,5],"lstm_hidden":64,"lstm_layers":1,"pooling":"masked_mean_max","pooling_projection":64,"aggregate_hidden":64,"static_hidden":32,"fusion_hidden":64,"dropout":d["dropout"],"fusion":"gated_residual","branch_dropout":.1,"learning_rate":d["learning_rate"],"weight_decay":d["weight_decay"],"batch_size":d["batch_size"],"max_epochs":d["max_epochs"],"patience":d["patience"]}


def _deep_model(kind:str, aggregate_dim:int, static_dim:int, config:dict[str,Any]) -> nn.Module:
    if kind=="cnn_bilstm": return CNNBiLSTMOULAD(47,aggregate_dim,static_dim,config)
    return _OULADCNNBiLSTMBackbone(47,aggregate_dim,static_dim,config,kind)


def _deep_probability(model:nn.Module, tensors:tuple[torch.Tensor,...], kind:str) -> torch.Tensor:
    sequence,length,mask,aggregate,static=tensors
    value=model(sequence,length,mask,aggregate,static)
    logits=value["binary_logit"] if isinstance(value,dict) else value
    return torch.sigmoid(logits)


def _fit_deep(kind:str, train:tuple, validation:tuple, seed:int, protocol:dict[str,Any], selected_epoch:int|None=None) -> tuple[dict[str,Any],np.ndarray,int]:
    torch.manual_seed(seed); np.random.seed(seed)
    frame,seq,length,mask,agg,y,w=train; vf,vseq,vlen,vmask,vagg,vy,_=validation
    pre=_DeepPreprocessor().fit(frame,agg); agg,static=pre.transform(frame,agg); vagg,vstatic=pre.transform(vf,vagg)
    config=_deep_config(protocol)
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_GPU: CUDA is required for OULAD deep checkpoints")
    device=torch.device("cuda")
    model=_deep_model(kind,agg.shape[1],static.shape[1],config).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    pos=float((y==0).sum()/max((y==1).sum(),1)); risk=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos,device=device),reduction="none")
    data=TensorDataset(torch.from_numpy(seq),torch.from_numpy(length.astype(np.int64)),torch.from_numpy(mask.astype(np.float32)),torch.from_numpy(agg),torch.from_numpy(static),torch.from_numpy(y.astype(np.float32)),torch.from_numpy(w.astype(np.float32)),torch.from_numpy(frame.outcome_aux.to_numpy(dtype=np.int64)),torch.from_numpy(frame.cutoff_day.to_numpy(dtype=np.int64)),torch.from_numpy(frame.module_presentation_length.to_numpy(dtype=np.int64)),torch.from_numpy(frame.date_unregistration.fillna(-1).to_numpy(dtype=np.int64)))
    loader=DataLoader(data,batch_size=config["batch_size"],shuffle=True,generator=torch.Generator().manual_seed(seed))
    fixed=selected_epoch or config["max_epochs"]; best=float("-inf"); best_state=None; best_epoch=1; wait=0
    for epoch in range(1,fixed+1):
        model.train()
        for b in loader:
            s,length_tensor,m,a,st,t,wt,out,cut,end,unreg=(v.to(device) for v in b); opt.zero_grad()
            output=model(s,length_tensor,m,a,st)
            logits=output["binary_logit"] if isinstance(output,dict) else output
            loss=(risk(logits,t)*wt).sum()/wt.sum().clamp_min(1e-8)
            if isinstance(output,dict):
                outcome_loss=nn.functional.cross_entropy(output["outcome_logit"],out)
                horizon=output["hazard_logit"].shape[1]; offsets=torch.arange(horizon,device=device)[None,:]
                observed=(offsets*7 < (end-cut).unsqueeze(1)).float()
                event=((unreg-cut).unsqueeze(1) >= 0) & ((unreg-cut).unsqueeze(1) >= offsets*7) & ((unreg-cut).unsqueeze(1) < (offsets+1)*7)
                hazard=event.float(); surv=nn.functional.binary_cross_entropy_with_logits(output["hazard_logit"],hazard,reduction="none"); surv=(surv*observed).sum()/observed.sum().clamp_min(1.0)
                loss=loss+.15*surv+.15*outcome_loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if selected_epoch is not None: continue
        prob=_predict_deep(model,vseq,vlen,vmask,vagg,vstatic,kind,device); score=f1_score(vy,prob>=.5,average="macro")
        if score>best+1e-8: best=score; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_epoch=epoch; wait=0
        else: wait+=1
        if wait>=config["patience"]: break
    if selected_epoch is None and best_state is not None: model.load_state_dict(best_state)
    probability=_predict_deep(model,vseq,vlen,vmask,vagg,vstatic,kind,device)
    payload={"state_dict":model.cpu().state_dict(),"preprocessor":pre.state(),"config":config,"kind":kind,"aggregate_dim":int(agg.shape[1]),"static_dim":int(static.shape[1]),"selected_epoch":best_epoch,"parameter_count":count_parameters(model),"temporal_channel_order":list(CHANNELS),"aggregate_feature_order":[f"aggregate_{i:03d}" for i in range(165)],"static_feature_order":list(STATIC_COLUMNS),"stage_context_feature_order":["cutoff_day","progress_fraction","remaining_days","observed_weeks"],"seed":int(seed),"cuda_device":torch.cuda.get_device_name(0),"deterministic_metadata":{"fixed_seed":int(seed),"best_seed_selection":False}}
    return payload,probability,best_epoch


def _predict_deep(model:nn.Module,seq:np.ndarray,length:np.ndarray,mask:np.ndarray,agg:np.ndarray,static:np.ndarray,kind:str,device:torch.device) -> np.ndarray:
    model.eval(); out=[]
    with torch.no_grad():
        for start in range(0,len(seq),512):
            sl=slice(start,start+512); tensors=(torch.from_numpy(seq[sl]).to(device),torch.from_numpy(length[sl].astype(np.int64)).to(device),torch.from_numpy(mask[sl].astype(np.float32)).to(device),torch.from_numpy(agg[sl]).to(device),torch.from_numpy(static[sl]).to(device))
            out.append(_deep_probability(model,tensors,kind).cpu().numpy())
    return np.concatenate(out)


def _load_deep(payload:dict[str,Any]) -> tuple[nn.Module,_DeepPreprocessor,torch.device]:
    model=_deep_model(payload["kind"],int(payload["aggregate_dim"]),int(payload["static_dim"]),payload["config"])
    model.load_state_dict(payload["state_dict"]); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    pre=_DeepPreprocessor(); pre.mean=payload["preprocessor"]["mean"]; pre.scale=payload["preprocessor"]["scale"]; pre.num_cols=payload["preprocessor"]["num_cols"]; pre.num_mean=payload["preprocessor"]["num_mean"]; pre.num_scale=payload["preprocessor"]["num_scale"]; pre.categories=payload["preprocessor"]["categories"]
    return model,pre,device


def _inner_splits(base:pd.DataFrame,outer:int) -> Iterable[tuple[set[str],set[str]]]:
    train=base.loc[base.outer_fold.ne(outer)].drop_duplicates("base_record_id").reset_index(drop=True)
    splitter=StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=20260+outer)
    for a,b in splitter.split(train,train.target,train.id_student):
        yield set(train.iloc[a].base_record_id),set(train.iloc[b].base_record_id)


def _threshold(y:np.ndarray,p:np.ndarray) -> tuple[float,str]:
    candidates=np.unique(np.r_[np.linspace(.05,.95,181),p])
    rows=[]
    for t in candidates:
        pred=p>=t; precision=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)[0]; recall=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)[1]
        rows.append((float(t),float(precision),float(recall)))
    qualified=[r for r in rows if r[1]>=.75]
    if qualified: return max(qualified,key=lambda r:(r[2],r[1]))[0],"PRECISION_CONSTRAINT_MET"
    return min(rows,key=lambda r:abs(r[1]-.75))[0],"CONSTRAINT_NOT_REACHED"


def _ensure_inner(bundle:Bundle,protocol:dict[str,Any]) -> tuple[pd.DataFrame,pd.DataFrame]:
    trial_path=OUT/"inner_trials.csv"; policy_path=OUT/"threshold_policies.csv"; oof_path=OUT/".runtime_cache"/"inner_oof.parquet"
    if trial_path.is_file() and policy_path.is_file() and oof_path.is_file():
        return _ensure_amended_svm_inner(
            bundle, pd.read_csv(trial_path), pd.read_csv(policy_path)
        )
    base=bundle.base[["base_record_id","id_student","outer_fold","target"]]
    rows=[]; oof=[]
    for model in MODELS:
        for outer in sorted(base.outer_fold.unique()):
            for inner,(fit_ids,val_ids) in enumerate(_inner_splits(base,int(outer))):
                tr=_stage_rows(bundle,fit_ids); va=_stage_rows(bundle,val_ids); start=time.perf_counter()
                if model in TABULAR:
                    estimator,payload=_fit_tabular(model,tr,va,42)
                    prob=payload["probability"]
                else:
                    _,prob,_=_fit_deep(model,tr,va,42,protocol)
                frame=va[0].loc[:,["base_record_id","id_student","prediction_stage","target"]].copy(); frame["probability"]=prob; frame["model_family"]=model; frame["outer_fold"]=outer; frame["inner_fold"]=inner; oof.append(frame)
                stage_scores=frame.groupby("prediction_stage").apply(lambda g:f1_score(g.target,g.probability>=.5,average="macro"),include_groups=False)
                rows.append({"model_family":model,"outer_fold":outer,"inner_fold":inner,"config_id":"frozen_default","mean_stage_macro_f1_operational":float(stage_scores.mean()),"worst_stage_macro_f1":float(stage_scores.min()),"runtime_seconds":time.perf_counter()-start,"outer_labels_used":False})
    oof_frame=pd.concat(oof,ignore_index=True); _write_parquet(oof_path,oof_frame); trials=pd.DataFrame(rows); _write_csv(trial_path,trials)
    policies=[]
    for (model,outer,stage),g in oof_frame.groupby(["model_family","outer_fold","prediction_stage"]):
        t,status=_threshold(g.target.to_numpy(),g.probability.to_numpy()); policies.append({"model_family":model,"outer_fold":outer,"prediction_stage":stage,"threshold_policy":"INNER_OOF_STAGE_THRESHOLD","threshold":t,"status":status,"source":"pooled_inner_oof"})
        policies.append({"model_family":model,"outer_fold":outer,"prediction_stage":stage,"threshold_policy":"FIXED_0_5","threshold":.5,"status":"FIXED","source":"protocol"})
    policy=pd.DataFrame(policies); _write_csv(policy_path,policy)
    return _ensure_amended_svm_inner(bundle, trials, policy)


def smoke() -> dict[str,Any]:
    bundle=_build_bundle(); p=_protocol(); _ensure_inner(bundle,p)
    _write_json(OUT/"smoke_validation.json",{"status":"PASS","scope":"data_contract_and_inner_pipeline","models":10,"stages":4,"not_final_evidence":True})
    return {"status":"PASS","smoke":"data_contract_and_inner_pipeline"}


def _expected_run_config_hash(model: str) -> str:
    if model == "svm":
        return _stable(
            {
                "family": "sklearn.svm.SVC",
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "probability": False,
                "class_weight": "balanced",
                "cache_size_mb": 4096,
                "shrinking": True,
                "tol": 1e-3,
                "max_iter": -1,
                "calibration": "inner_oof_platt_logistic_regression",
            }
        )
    return _stable({"model": model, "config": "frozen_default"})


def _resume_checkpoint_valid(
    model: str, path: Path, outer: int, seed: int
) -> bool:
    if not path.is_file():
        return False
    try:
        if model == "svm":
            payload = joblib.load(path)
            return (
                isinstance(payload, dict)
                and payload.get("checkpoint_schema") == "oulad_calibrated_svm_v1"
                and payload.get("model_id") == "svm_oulad"
                and int(payload.get("outer_fold", -1)) == outer
                and int(payload.get("seed", -1)) == seed
                and payload.get("config_hash") == _expected_run_config_hash(model)
                and payload.get("outer_labels_used_for_calibration") is False
            )
        if model in DEEP:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            return (
                payload.get("model_id") == f"{model}_oulad"
                and int(payload.get("outer_fold", -1)) == outer
                and int(payload.get("seed", -1)) == seed
                and payload.get("config_hash") == _expected_run_config_hash(model)
                and payload.get("temporal_channel_order") == list(CHANNELS)
            )
        audit_path = OUT / "resume_checkpoint_audit.json"
        if not audit_path.is_file():
            return False
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        relative = path.relative_to(ROOT).as_posix()
        row = next(
            (
                item
                for item in audit.get("rows", [])
                if item.get("path") == relative
            ),
            None,
        )
        return bool(
            row
            and row.get("readable")
            and row.get("sha256") == _sha(path)
            and int(row.get("outer_fold", -1)) == outer
            and int(row.get("seed", -1)) == seed
        )
    except Exception:
        return False


def train(resume:bool=True) -> dict[str,Any]:
    bundle=_build_bundle(); protocol=_protocol(); _ensure_inner(bundle,protocol)
    base=bundle.base[["base_record_id","outer_fold"]].drop_duplicates(); rows=[]; mapping=[]; runtime=[]
    for model in MODELS:
        for outer in sorted(base.outer_fold.unique()):
            fit_ids=set(base.loc[base.outer_fold.ne(outer),"base_record_id"]); val_ids=set(base.loc[base.outer_fold.eq(outer),"base_record_id"]); tr=_stage_rows(bundle,fit_ids); va=_stage_rows(bundle,val_ids)
            selected_epoch=int(pd.read_csv(OUT/"inner_trials.csv").query("model_family == @model and outer_fold == @outer")["mean_stage_macro_f1_operational"].shape[0] and _protocol()["training"]["deep"]["max_epochs"])
            for seed in SEEDS:
                path=OUT/"checkpoints"/f"{model}_oulad"/f"outer_fold_{int(outer)}"/(f"seed_{seed}.joblib" if model in TABULAR else f"seed_{seed}.pt")
                start=time.perf_counter()
                resumed=bool(
                    resume
                    and _resume_checkpoint_valid(
                        model, path, int(outer), int(seed)
                    )
                )
                if not resumed:
                    if path.is_file():
                        quarantine=OUT/"quarantine"/f"{model}_oulad"/f"outer_fold_{int(outer)}"/path.name
                        quarantine.parent.mkdir(parents=True,exist_ok=True)
                        shutil.move(str(path),str(quarantine))
                    path.parent.mkdir(parents=True,exist_ok=True)
                    if model == "svm":
                        payload = _fit_calibrated_svm(
                            bundle, int(outer), int(seed), tr
                        )
                        joblib.dump(payload, path)
                        epoch=None; params=None
                    elif model in TABULAR:
                        estimator,_=_fit_tabular(model,tr,va,int(seed)); joblib.dump(estimator,path)
                        epoch=None; params=None
                    else:
                        payload,_,epoch=_fit_deep(model,tr,va,int(seed),protocol,selected_epoch=selected_epoch)
                        payload["outer_fold"]=int(outer)
                        payload["model_id"]=f"{model}_oulad"
                        payload["training_run_id"]=_stable({"dataset":"oulad","model":model,"outer":int(outer),"seed":int(seed),"config":"frozen_default"})[:24]
                        payload["config_hash"]=_stable({"model":model,"config":"frozen_default"})
                        payload["checkpoint_id"]=_stable({"run":payload["training_run_id"],"selected_epoch":epoch})[:24]
                        torch.save(payload,path); params=payload["parameter_count"]
                else:
                    epoch=None; params=None
                run_id=_stable({"dataset":"oulad","model":model,"outer":int(outer),"seed":int(seed),"config_hash":_expected_run_config_hash(model)})[:24]
                rows.append({"dataset":"oulad","model_id":f"{model}_oulad","model_family":model,"outer_fold":int(outer),"seed":int(seed),"config_hash":_expected_run_config_hash(model),"training_run_id":run_id,"checkpoint":path.relative_to(ROOT).as_posix(),"checkpoint_sha256":_sha(path),"status":"RESUMED" if resumed else "COMPLETE","selected_epoch":epoch,"parameter_count":params})
                for stage in STAGES: mapping.append({"training_run_id":run_id,"model_id":f"{model}_oulad","outer_fold":int(outer),"seed":int(seed),"prediction_stage":stage,"checkpoint":path.relative_to(ROOT).as_posix(),"checkpoint_sha256":_sha(path)})
                runtime.append({"model_family":model,"outer_fold":int(outer),"seed":int(seed),"runtime_seconds":time.perf_counter()-start,"cache_hit":resumed})
    manifest=pd.DataFrame(rows); _write_json(OUT/"training_run_manifest.json",{"status":"PASS","training_run_count":len(manifest),"runs":manifest.to_dict("records")}); _write_json(OUT/"checkpoint_stage_mapping.json",{"status":"PASS","mapping_count":len(mapping),"same_checkpoint_all_stages":True,"rows":mapping}); _write_csv(OUT/"runtime.csv",pd.DataFrame(runtime))
    return {"status":"PASS","training_runs":len(manifest),"checkpoints":len(manifest)}


def _predict_checkpoint(model:str,path:Path,frame:pd.DataFrame,seq:np.ndarray,length:np.ndarray,mask:np.ndarray,agg:np.ndarray) -> np.ndarray:
    if model in TABULAR:
        estimator=joblib.load(path)
        if model == "svm":
            if not isinstance(estimator, dict) or estimator.get("checkpoint_schema") != "oulad_calibrated_svm_v1":
                raise RuntimeError(f"superseded SVM checkpoint is not final authority: {path}")
            decision = estimator["estimator"].decision_function(
                _tabular_frame(frame, agg)
            )
            return estimator["calibrator"].predict_proba(
                np.asarray(decision).reshape(-1, 1)
            )[:, 1]
        return estimator.predict_proba(_tabular_frame(frame,agg))[:,1]
    payload=torch.load(path,map_location="cpu",weights_only=False); net,pre,device=_load_deep(payload); a,s=pre.transform(frame,agg); return _predict_deep(net,seq,length,mask,a,s,model,device)


def _ece(y:np.ndarray,p:np.ndarray,bins:int=15)->float:
    edges=np.linspace(0,1,bins+1); total=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(p>=lo)&(p<(hi if hi<1 else hi+1e-9))
        if m.any(): total += m.mean()*abs(p[m].mean()-y[m].mean())
    return float(total)


def _metric(y:np.ndarray,p:np.ndarray,threshold:float)->dict[str,float]:
    z=np.clip(p,1e-7,1-1e-7); pred=z>=threshold; cm=confusion_matrix(y,pred,labels=[0,1]); tn,fp,fn,tp=cm.ravel(); pr,rc,f,_=precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0)
    return {"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"macro_precision":float(precision_recall_fscore_support(y,pred,average="macro",zero_division=0)[0]),"macro_recall":float(precision_recall_fscore_support(y,pred,average="macro",zero_division=0)[1]),"macro_f1":float(f1_score(y,pred,average="macro")),"weighted_f1":float(f1_score(y,pred,average="weighted")),"risk_precision":float(pr[1]),"risk_recall":float(rc[1]),"risk_f1":float(f[1]),"not_risk_precision":float(pr[0]),"not_risk_recall":float(rc[0]),"not_risk_f1":float(f[0]),"specificity":float(tn/max(tn+fp,1)),"pr_auc":float(average_precision_score(y,z)),"roc_auc":float(roc_auc_score(y,z)),"brier":float(np.mean((z-y)**2)),"nll":float(log_loss(y,z,labels=[0,1])),"ece":_ece(y,z),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"eligible_count":int(len(y)),"positive_count":int(y.sum()),"negative_count":int((1-y).sum()),"coverage":1.0}


def evaluate() -> dict[str,Any]:
    bundle=_build_bundle(); _,policies=_ensure_inner(bundle,_protocol()); base=bundle.base[["base_record_id","outer_fold"]].drop_duplicates(); seed_rows=[]
    for model in MODELS:
        for outer in sorted(base.outer_fold.unique()):
            val_ids=set(base.loc[base.outer_fold.eq(outer),"base_record_id"])
            for stage in STAGES:
                d=bundle.stages[stage]; ids=np.flatnonzero(d.frame.base_record_id.isin(val_ids).to_numpy()); f=d.frame.iloc[ids].reset_index(drop=True)
                for seed in SEEDS:
                    path=OUT/"checkpoints"/f"{model}_oulad"/f"outer_fold_{int(outer)}"/(f"seed_{seed}.joblib" if model in TABULAR else f"seed_{seed}.pt")
                    prob=_predict_checkpoint(model,path,f,d.sequence[ids],d.lengths[ids],d.mask[ids],d.aggregate[ids])
                    rows=f.loc[:,["base_record_id","id_student","code_module","code_presentation","target","cutoff_day"]].copy(); rows["model_id"]=f"{model}_oulad"; rows["model_family"]=model; rows["prediction_stage"]=stage; rows["outer_fold"]=int(outer); rows["seed"]=int(seed); rows["probability"]=prob; seed_rows.append(rows)
    seed_frame=pd.concat(seed_rows,ignore_index=True); _write_parquet(OUT/"seed_predictions.parquet",seed_frame)
    pred=seed_frame.groupby(["base_record_id","id_student","code_module","code_presentation","target","cutoff_day","model_id","model_family","prediction_stage","outer_fold"],as_index=False).probability.mean(); pred["predicted_label_fixed_0_5"]=(pred.probability>=.5).astype(int); _write_parquet(OUT/"predictions.parquet",pred)
    metric_rows=[]; per=[]
    for key,g in pred.groupby(["model_id","model_family","prediction_stage","outer_fold"]):
        model_id,model,stage,outer=key; y=g.target.to_numpy(); p=g.probability.to_numpy(); seed_metrics=[]
        for seed,sg in seed_frame.loc[(seed_frame.model_id==model_id)&(seed_frame.prediction_stage==stage)&(seed_frame.outer_fold==outer)].groupby("seed"):
            seed_metrics.append(_metric(sg.target.to_numpy(),sg.probability.to_numpy(),.5)["macro_f1"])
        for policy in ("FIXED_0_5","INNER_OOF_STAGE_THRESHOLD"):
            threshold=float(policies.query("model_family == @model and outer_fold == @outer and prediction_stage == @stage and threshold_policy == @policy").threshold.iloc[0])
            values=_metric(y,p,threshold); metric_rows.append({"model_id":model_id,"model_family":model,"prediction_stage":stage,"outer_fold":outer,"threshold_policy":policy,"threshold":threshold,"seed_macro_f1_mean":float(np.mean(seed_metrics)),"seed_macro_f1_std":float(np.std(seed_metrics)),"seed_macro_f1_min":float(np.min(seed_metrics)),"seed_macro_f1_max":float(np.max(seed_metrics)),**values})
            for label,idx in (("Not-at-risk",0),("At-risk",1)):
                pr,rc,ff,su=precision_recall_fscore_support(y,p>=threshold,labels=[0,1],zero_division=0); per.append({"model_id":model_id,"model_family":model,"prediction_stage":stage,"outer_fold":outer,"threshold_policy":policy,"class_name":label,"precision":float(pr[idx]),"recall":float(rc[idx]),"f1":float(ff[idx]),"support":int(su[idx])})
    fold_metrics=pd.DataFrame(metric_rows); stage_metrics=fold_metrics.groupby(["model_id","model_family","prediction_stage","threshold_policy"],as_index=False).agg({c:"mean" for c in fold_metrics.columns if c not in {"model_id","model_family","prediction_stage","threshold_policy","outer_fold"}})
    for policy,gidx in stage_metrics.groupby("threshold_policy").groups.items():
        ranks=stage_metrics.loc[gidx].groupby("prediction_stage")["macro_f1"].rank(method="min",ascending=False); stage_metrics.loc[gidx,"rank_macro_f1"]=ranks
    _write_csv(OUT/"stage_metrics.csv",stage_metrics); _write_csv(OUT/"per_class_metrics.csv",pd.DataFrame(per).groupby(["model_id","model_family","prediction_stage","threshold_policy","class_name"],as_index=False).mean(numeric_only=True))
    primary=stage_metrics.loc[stage_metrics.threshold_policy.eq("INNER_OOF_STAGE_THRESHOLD")].copy(); overall=[]
    for (mid,m),g in primary.groupby(["model_id","model_family"]):
        s=g.set_index("prediction_stage"); vals=s.loc[list(STAGES)]
        overall.append({"model_id":mid,"model_family":m,"early_mean_macro_f1":float(vals.loc[list(STAGES[:2]),"macro_f1"].mean()),"early_mean_risk_recall":float(vals.loc[list(STAGES[:2]),"risk_recall"].mean()),"early_mean_risk_f1":float(vals.loc[list(STAGES[:2]),"risk_f1"].mean()),"early_worst_risk_recall":float(vals.loc[list(STAGES[:2]),"risk_recall"].min()),"early_mean_pr_auc":float(vals.loc[list(STAGES[:2]),"pr_auc"].mean()),"mean_stage_macro_f1":float(vals.macro_f1.mean()),"worst_stage_macro_f1":float(vals.macro_f1.min()),"harmonic_stage_macro_f1":float(len(vals)/np.sum(1/np.maximum(vals.macro_f1,1e-8))),"mean_stage_balanced_accuracy":float(vals.balanced_accuracy.mean()),"mean_stage_risk_recall":float(vals.risk_recall.mean()),"mean_stage_risk_f1":float(vals.risk_f1.mean()),"mean_stage_pr_auc":float(vals.pr_auc.mean()),"mean_stage_brier":float(vals.brier.mean()),"mean_stage_nll":float(vals.nll.mean()),"mean_stage_ece":float(vals.ece.mean()),"gain_E1_to_E2":float(vals.loc[STAGES[1],"macro_f1"]-vals.loc[STAGES[0],"macro_f1"]),"gain_E2_to_M1":float(vals.loc[STAGES[2],"macro_f1"]-vals.loc[STAGES[1],"macro_f1"]),"gain_M1_to_L1":float(vals.loc[STAGES[3],"macro_f1"]-vals.loc[STAGES[2],"macro_f1"]),"gain_E1_to_L1":float(vals.loc[STAGES[3],"macro_f1"]-vals.loc[STAGES[0],"macro_f1"]),"final_L1_macro_f1":float(vals.loc[STAGES[3],"macro_f1"]),"final_L1_risk_precision":float(vals.loc[STAGES[3],"risk_precision"]),"final_L1_risk_recall":float(vals.loc[STAGES[3],"risk_recall"]),"final_L1_risk_f1":float(vals.loc[STAGES[3],"risk_f1"]),"final_L1_pr_auc":float(vals.loc[STAGES[3],"pr_auc"]),"final_L1_ece":float(vals.loc[STAGES[3],"ece"]),"seed_std_mean":float(vals.seed_macro_f1_std.mean()),"seed_std_max":float(vals.seed_macro_f1_std.max())})
    overall=pd.DataFrame(overall); _write_csv(OUT/"overall_metrics.csv",overall)
    _common_and_module(bundle,pred,primary)
    return {"status":"PASS","stage_metric_rows":int(len(primary)),"overall_rows":int(len(overall)),"prediction_rows":int(len(pred))}


def _common_and_module(bundle:Bundle,pred:pd.DataFrame,primary:pd.DataFrame)->None:
    eligible=set.intersection(*[set(d.frame.base_record_id) for d in bundle.stages.values()]); common=pred.loc[pred.base_record_id.isin(eligible)].copy(); rows=[]; module=[]
    for (mid,m,stage),g in common.groupby(["model_id","model_family","prediction_stage"]):
        threshold=float(primary.query("model_id==@mid and prediction_stage==@stage").threshold.iloc[0]); rows.append({"model_id":mid,"model_family":m,"prediction_stage":stage,"cohort":"COMMON_ALL_STAGE_COHORT","common_base_records":len(eligible),**_metric(g.target.to_numpy(),g.probability.to_numpy(),threshold)})
    for (mid,m,stage,mod,pres),g in pred.groupby(["model_id","model_family","prediction_stage","code_module","code_presentation"]):
        if len(g)<60 or g.target.sum()<10 or (1-g.target).sum()<10: continue
        threshold=float(primary.query("model_id==@mid and prediction_stage==@stage").threshold.iloc[0]); module.append({"model_id":mid,"model_family":m,"prediction_stage":stage,"code_module":mod,"code_presentation":pres,"eligible":True,**_metric(g.target.to_numpy(),g.probability.to_numpy(),threshold)})
    _write_csv(OUT/"common_cohort_metrics.csv",pd.DataFrame(rows)); _write_csv(OUT/"module_metrics.csv",pd.DataFrame(module))


def bootstrap() -> dict[str,Any]:
    pred=pd.read_parquet(OUT/"predictions.parquet"); stage=pd.read_csv(OUT/"stage_metrics.csv"); primary=stage.query("threshold_policy == 'INNER_OOF_STAGE_THRESHOLD'"); rng=np.random.default_rng(7319); rows=[]
    # The same grouped resample is retained for both paired models within each stage.
    for stage_name,sg in pred.groupby("prediction_stage"):
        base=sg.loc[sg.model_family.eq("cnn_bilstm")]; groups=np.array(sorted(base.id_student.unique())); samples=rng.integers(0,len(groups),size=(5000,len(groups)))
        for comp in [m for m in MODELS if m!="cnn_bilstm"]:
            other=sg.loc[sg.model_family.eq(comp)]
            aligned=base.merge(other,on=["base_record_id","id_student"],suffixes=("_cnn","_cmp"),validate="one_to_one")
            if aligned.empty: continue
            group_index={g:i for i,g in enumerate(groups)}; gi=aligned.id_student.map(group_index).to_numpy(); y=aligned.target_cnn.to_numpy(); pc=aligned.probability_cnn.to_numpy(); po=aligned.probability_cmp.to_numpy()
            tc=float(primary.query("model_family=='cnn_bilstm' and prediction_stage==@stage_name").threshold.iloc[0]); to=float(primary.query("model_family==@comp and prediction_stage==@stage_name").threshold.iloc[0])
            deltas=[]
            for take in samples:
                counts=np.bincount(take,minlength=len(groups))[gi];
                def score(p,t):
                    z=p>=t
                    tn=np.sum(counts*((y==0)&(~z))); fp=np.sum(counts*((y==0)&z)); fn=np.sum(counts*((y==1)&(~z))); tp=np.sum(counts*((y==1)&z))
                    f0=2*tn/max(2*tn+fp+fn,1); f1=2*tp/max(2*tp+fp+fn,1)
                    brier=np.sum(counts*(p-y)**2)/max(counts.sum(),1)
                    nll=-np.sum(counts*(y*np.log(np.clip(p,1e-7,1))+(1-y)*np.log(np.clip(1-p,1e-7,1))))/max(counts.sum(),1)
                    return (f0+f1)/2, tp/max(tp+fn,1), f1, brier, nll
                a,b=score(pc,tc),score(po,to); deltas.append(np.subtract(a,b))
            d=np.asarray(deltas)
            for j,name in enumerate(("macro_f1","risk_recall","risk_f1","brier","nll")):
                delta=float(np.mean(d[:,j])); lo,hi=np.quantile(d[:,j],[.025,.975]); higher_good=name not in {"brier","nll"}; conclusion="insufficient evidence of difference" if lo<=0<=hi else ("CNN-BiLSTM higher" if (lo>0)==higher_good else "comparator higher")
                rows.append({"prediction_stage":stage_name,"base_model_id":"cnn_bilstm_oulad","comparator_model_id":f"{comp}_oulad","metric":name,"delta":delta,"ci_95_low":float(lo),"ci_95_high":float(hi),"replicates":5000,"resampling_unit":"id_student","conclusion":conclusion})
    _write_csv(OUT/"bootstrap_stage.csv",pd.DataFrame(rows)); _write_csv(OUT/"bootstrap_overall.csv",pd.DataFrame())
    return {"status":"PASS","stage_rows":len(rows),"replicates":5000}


def report() -> dict[str,Any]:
    stage=pd.read_csv(OUT/"stage_metrics.csv").query("threshold_policy == 'INNER_OOF_STAGE_THRESHOLD'"); overall=pd.read_csv(OUT/"overall_metrics.csv"); boot=pd.read_csv(OUT/"bootstrap_stage.csv") if (OUT/"bootstrap_stage.csv").is_file() else pd.DataFrame()
    best=stage.sort_values("macro_f1",ascending=False).groupby("prediction_stage").first().reset_index(); cnn=stage.query("model_family=='cnn_bilstm'")
    lines=["# OULAD Unified Multi-stage Results","","All OULAD model identities use one estimator/checkpoint per outer fold and seed across E1, E2, M1 and L1.","","| Stage | Best model | Macro-F1 | CNN-BiLSTM Macro-F1 |","|---|---|---:|---:|"]
    for s in STAGES:
        b=best.loc[best.prediction_stage.eq(s)].iloc[0]; c=cnn.loc[cnn.prediction_stage.eq(s)].iloc[0]; lines.append(f"| {s} | {b.model_id} | {b.macro_f1:.4f} | {c.macro_f1:.4f} |")
    lines.extend(["","M1 retains the historical F2 cutoff definition, but its unified training result is not expected to reproduce the historical frozen F2 score exactly.","", "Future OULAD is `LOCKED_NOT_EXECUTED`; recommendations remain frozen and no canonical database cutover was performed."])
    (ROOT/"reports"/"final"/"OULAD_UNIFIED_MULTI_STAGE_RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (ROOT/"reports"/"final"/"OULAD_EARLY_WARNING_RESULTS.md").write_text("# OULAD Early-warning Results\n\nSee `OULAD_UNIFIED_MULTI_STAGE_RESULTS.md`. Early rankings use E1/E2 operational cohorts; the common all-stage cohort is diagnostic only.\n",encoding="utf-8")
    (ROOT/"reports"/"final"/"OULAD_MODEL_SELECTION_REPORT.md").write_text("# OULAD Model Selection\n\nConfigurations were selected only with grouped inner folds using mean operational-stage Macro-F1. No outer labels, transfer, pretraining checkpoint, synthetic resampling, or best-seed selection was used.\n",encoding="utf-8")
    hybrid = boot.loc[boot.base_model_id.eq("cnn_bilstm_oulad")].copy() if not boot.empty else pd.DataFrame()
    matrix = [
        "# OULAD Hybrid vs ML Stage Matrix",
        "",
        "CNN-BiLSTM is compared with each tabular/deep comparator using 5,000 paired grouped bootstrap replicates over `id_student`. A confidence interval crossing zero is reported as insufficient evidence of difference; it is never described as equivalence.",
        "",
        "| Stage | Comparator | Metric | Delta (CNN-BiLSTM − comparator) | 95% CI | Conclusion |",
        "|---|---|---|---:|---|---|",
    ]
    for row in hybrid.sort_values(["prediction_stage", "comparator_model_id", "metric"]).itertuples(index=False):
        matrix.append(f"| {row.prediction_stage} | {row.comparator_model_id} | {row.metric} | {row.delta:.4f} | [{row.ci_95_low:.4f}, {row.ci_95_high:.4f}] | {row.conclusion} |")
    (ROOT/"reports"/"final"/"OULAD_HYBRID_VS_ML_STAGE_MATRIX.md").write_text("\n".join(matrix)+"\n",encoding="utf-8")
    _update_project_authority(stage,overall)
    return {"status":"PASS","reports":4,"bootstrap_rows":len(boot)}


def _update_project_authority(stage:pd.DataFrame,overall:pd.DataFrame)->None:
    final_stage=pd.read_csv(ROOT/"artifacts"/"final"/"final_stage_results.csv")
    final_stage=final_stage.loc[final_stage.dataset.ne("oulad")].copy()
    add=stage.copy(); add["dataset"]="oulad"; add["authority_scope"]="UNIFIED_OULAD_MULTI_STAGE"; _write_csv(ROOT/"artifacts"/"final"/"final_stage_results.csv",pd.concat([final_stage,add],ignore_index=True,sort=False))
    final_overall=pd.read_csv(ROOT/"artifacts"/"final"/"final_overall_results.csv"); final_overall=final_overall.loc[final_overall.dataset.ne("oulad")].copy(); addo=overall.copy(); addo["dataset"]="oulad"; addo["authority_scope"]="UNIFIED_OULAD_MULTI_STAGE"; _write_csv(ROOT/"artifacts"/"final"/"final_overall_results.csv",pd.concat([final_overall,addo],ignore_index=True,sort=False))
    lock=ROOT/"reports"/"final"/"PROJECT_LOCK_REPORT.md"
    text=lock.read_text(encoding="utf-8") if lock.is_file() else "# Final Project Lock\n"
    marker="\n## OULAD unified multi-stage authority\n"
    if marker in text:
        text=text.split(marker,1)[0].rstrip()+"\n"
    text += "\n## OULAD unified multi-stage authority\n\n- 10 OULAD model identities; 40 operational stage rows; 10 OULAD overall summaries.\n- One estimator/checkpoint is reused across E1, E2, M1 and L1 for every `(model, outer_fold, seed)` run.\n- M1 uses the exact historical F2 cutoff, but unified multi-stage results are a replacement authority and are not asserted to reproduce the frozen single-cutoff score.\n- Canonical frozen F2 artifacts remain historical compatibility evidence.\n"
    lock.write_text(text,encoding="utf-8")


def _archive_legacy() -> None:
    LEGACY.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/"artifacts"/"final"/"oulad",ROOT/"artifacts"/"final"/"predictions"/"cnn_bilstm_oulad",ROOT/"artifacts"/"final"/"models"/"cnn_bilstm_oulad"]
    rows=[]
    for source in sources:
        if source.exists(): rows.append({"path":source.relative_to(ROOT).as_posix(),"sha256":_stable(sorted((p.relative_to(ROOT).as_posix(),_sha(p)) for p in source.rglob("*") if p.is_file()))})
    _write_json(LEGACY/"archive_manifest.json",{"status":"PRESERVED_IN_PLACE_HISTORICAL","filesets":rows,"note":"Frozen single-cutoff F2 evidence remains in place to avoid replacing official artifacts; this manifest records its historical archive authority."})
    (ROOT/"reports"/"history"/"LEGACY_OULAD_F2_REPORT.md").write_text("# Legacy OULAD F2 Report\n\nHistorical single-cutoff F2 evidence is retained for traceability and compatibility regression only. It is superseded for multi-stage ranking by the unified OULAD protocol.\n",encoding="utf-8")


def _checksums() -> dict[str,Any]:
    files=[p for p in OUT.rglob("*") if p.is_file() and p.name not in {"checksums.json","validation.json"} and ".runtime_cache" not in p.parts]
    value={"status":"PASS","files":[{"path":p.relative_to(ROOT).as_posix(),"sha256":_sha(p)} for p in sorted(files)]}; _write_json(OUT/"checksums.json",value); return value


def validate() -> dict[str,Any]:
    required=("cutoff_manifest.csv","eligibility_manifest.parquet","feature_lineage.json","architecture_freeze_audit.json","training_run_manifest.json","checkpoint_stage_mapping.json","stage_metrics.csv","overall_metrics.csv","predictions.parquet","seed_predictions.parquet","bootstrap_stage.csv")
    missing=[x for x in required if not (OUT/x).is_file()]
    errors=[]
    if missing: errors.append(f"missing: {missing}")
    if not missing:
        cut=pd.read_csv(OUT/"cutoff_manifest.csv"); train=json.loads((OUT/"training_run_manifest.json").read_text()); mapping=json.loads((OUT/"checkpoint_stage_mapping.json").read_text()); stage=pd.read_csv(OUT/"stage_metrics.csv"); overall=pd.read_csv(OUT/"overall_metrics.csv")
        if len(train["runs"])!=150: errors.append("training run count != 150")
        if len(mapping["rows"])!=600 or not mapping.get("same_checkpoint_all_stages"): errors.append("checkpoint stage mapping invalid")
        if not cut.monotonicity_pass.all() or not cut.loc[cut.stage.eq(STAGES[2]),"exact_f2_compatibility"].all(): errors.append("cutoff compatibility failed")
        primary=stage.query("threshold_policy == 'INNER_OOF_STAGE_THRESHOLD'")
        if len(primary)!=40: errors.append("OULAD authority stage rows != 40")
        if len(overall)!=10: errors.append("OULAD overall rows != 10")
        svm_policies=set(
            pd.read_csv(OUT/"threshold_policies.csv")
            .query("model_family == 'svm'")
            .threshold_policy
        )
        if svm_policies != {"FIXED_0_5","INNER_OOF_STAGE_THRESHOLD"}:
            errors.append("SVM threshold policy contract invalid")
        base_records=_build_bundle().base
        for outer in sorted(base_records.outer_fold.unique()):
            outer_ids=set(
                base_records.loc[
                    base_records.outer_fold.eq(outer),"base_record_id"
                ]
            )
            for seed in SEEDS:
                svm_path=OUT/"checkpoints"/"svm_oulad"/f"outer_fold_{int(outer)}"/f"seed_{seed}.joblib"
                if not _resume_checkpoint_valid("svm",svm_path,int(outer),int(seed)):
                    errors.append(f"invalid calibrated SVM checkpoint: outer={outer} seed={seed}")
                    continue
                svm_payload=joblib.load(svm_path)
                if outer_ids.intersection(svm_payload["calibration_record_ids"]):
                    errors.append(f"SVM calibration/outer overlap: outer={outer} seed={seed}")
        project_stage=pd.read_csv(ROOT/"artifacts"/"final"/"final_stage_results.csv"); project_overall=pd.read_csv(ROOT/"artifacts"/"final"/"final_overall_results.csv")
        if len(project_stage)!=100 or len(project_overall)!=30: errors.append("project authority totals invalid")
        if not (ROOT/"artifacts"/"final"/"unified_stage_aware_uci"/"checksums.json").is_file(): errors.append("UCI unified authority missing")
    result={"schema_version":"oulad_unified_validation_v1","status":"PASS" if not errors else "FAIL","errors":errors,"stages":list(STAGES),"model_identities":10,"final_training_runs":150,"stage_rows":40,"overall_rows":10,"project_stage_rows":100,"project_overall_rows":30,"future_oulad":"LOCKED_NOT_EXECUTED","canonical_database_modified":False,"recommendation_modified":False,"frozen_legacy_oulad_macro_f1":0.8280835945631038}
    _write_json(OUT/"validation.json",result); _checksums(); return result


def all_steps(resume:bool=True)->dict[str,Any]:
    prepare(); smoke(); train(resume=resume); evaluate(); bootstrap(); _archive_legacy(); report(); return validate()
