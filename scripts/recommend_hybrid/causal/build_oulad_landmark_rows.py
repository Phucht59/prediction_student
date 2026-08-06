"""Build the authoritative OULAD landmark table for causal recommendation.

The script uses only pre-cutoff canonical views for confounders and frozen OOF
Hybrid embeddings. Post-cutoff raw events are used only to define observed
treatment behaviour. Learners who later withdraw remain in the trial cohort,
which avoids conditioning the sample on surviving to the next landmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.canonical_v3.oulad_data import build_canonical_bundle  # noqa: E402
from src.pipelines import oulad  # noqa: E402
from src.recommend_hybrid.causal.study_regularity import (  # noqa: E402
    study_regularity_score,
)
from src.recommend_hybrid.causal.protocol import LANDMARK_STAGES  # noqa: E402
from src.recommend_hybrid.contracts import Stage  # noqa: E402
from src.recommend_hybrid.final.actions import ACTION_ORDER  # noqa: E402
from src.recommend_hybrid.prediction_adapter import (  # noqa: E402
    ARCHITECTURE_HASH,
    HybridPredictionAdapter,
    file_sha256,
)

OUTPUT = ROOT / "artifacts/recommend_hybrid/causal/input/landmark_rows.parquet"
MANIFEST = ROOT / "artifacts/recommend_hybrid/causal/input/landmark_rows_manifest.json"
CHECKPOINT_MANIFEST = ROOT / "artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json"
RAW = ROOT / "data/raw"
STAGE_SOURCE = {
    "EARLY_20": "E1_EARLY_20PCT",
    "EARLY_35": "E2_EARLY_35PCT",
    "MIDDLE_50": "M1_MIDDLE_50PCT",
    "LATE_75": "L1_LATE_75PCT",
}
STAGE_ENUM = {
    "EARLY_20": Stage.EARLY_20,
    "EARLY_35": Stage.EARLY_35,
    "MIDDLE_50": Stage.MIDDLE_50,
    "LATE_75": Stage.LATE_75,
}
NEXT_FRACTION = {
    "EARLY_20": 0.35,
    "EARLY_35": 0.50,
    "MIDDLE_50": 0.75,
    "LATE_75": 1.00,
}
PROTOCOL_SPLIT_BY_OUTER_FOLD = {0: "test", 1: "validation", 2: "train"}
CONTENT_TYPES = {
    "oucontent",
    "resource",
    "page",
    "url",
    "glossary",
    "homepage",
    "subpage",
    "dataplus",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _restore_preprocessor(state: dict[str, Any]) -> Any:
    preprocessor = oulad._DeepPreprocessor()
    for key, value in state.items():
        setattr(preprocessor, key, value)
    return preprocessor


def _windows(bundle: Any) -> dict[str, pd.DataFrame]:
    windows: dict[str, pd.DataFrame] = {}
    for stage, source in STAGE_SOURCE.items():
        frame = bundle.stages[source].frame.copy()
        length = frame["module_presentation_length"].astype(int)
        cutoff = frame["cutoff_day"].astype(int)
        next_day = np.floor(length * NEXT_FRACTION[stage]).astype(int)
        if stage == "MIDDLE_50":
            next_day = np.minimum(next_day, length - 14)
        if stage == "LATE_75":
            next_day = length
        next_day = np.maximum(next_day, cutoff + 1)
        followup_days = next_day - cutoff
        baseline_start = np.maximum(0, cutoff - followup_days)
        window = pd.DataFrame(
            {
                "record_id": frame["base_record_id"].astype(str),
                "student_id": frame["id_student"].astype(str),
                "code_module": frame["code_module"].astype(str),
                "code_presentation": frame["code_presentation"].astype(str),
                "course_id": (
                    frame["code_module"].astype(str)
                    + "::"
                    + frame["code_presentation"].astype(str)
                ),
                "outer_fold": frame["outer_fold"].astype(int),
                "module_length": length,
                "baseline_start_day": baseline_start.astype(int),
                "cutoff_day": cutoff.astype(int),
                "followup_end_day": next_day.astype(int),
            }
        )
        if window["outer_fold"].map(PROTOCOL_SPLIT_BY_OUTER_FOLD).isna().any():
            raise RuntimeError(f"{stage}: unsupported frozen outer fold")
        window["stage"] = stage
        window["protocol_split"] = window["outer_fold"].map(
            PROTOCOL_SPLIT_BY_OUTER_FOLD
        )
        windows[stage] = window
    return windows


def _collect_weekly_activity(
    windows: dict[str, pd.DataFrame],
    *,
    chunksize: int,
) -> dict[str, pd.DataFrame]:
    site = pd.read_csv(
        RAW / "vle.csv",
        usecols=["code_module", "code_presentation", "id_site", "activity_type"],
    ).drop_duplicates()
    keys = ["code_module", "code_presentation", "id_student"]
    merge_windows = {
        stage: frame.assign(id_student=frame["student_id"].astype(int)).loc[
            :,
            [
                *keys,
                "record_id",
                "baseline_start_day",
                "cutoff_day",
                "followup_end_day",
            ],
        ]
        for stage, frame in windows.items()
    }
    daily_parts: dict[str, list[pd.DataFrame]] = {stage: [] for stage in windows}
    usecols = [
        "code_module",
        "code_presentation",
        "id_student",
        "id_site",
        "date",
        "sum_click",
    ]
    for chunk in pd.read_csv(RAW / "studentVle.csv", usecols=usecols, chunksize=chunksize):
        chunk = chunk.merge(
            site,
            on=["code_module", "code_presentation", "id_site"],
            how="left",
            validate="many_to_one",
        )
        for stage, window in merge_windows.items():
            selected = chunk.merge(window, on=keys, how="inner", validate="many_to_one")
            selected = selected.loc[
                (selected["date"] >= selected["baseline_start_day"])
                & (selected["date"] < selected["followup_end_day"])
            ].copy()
            if selected.empty:
                continue
            baseline = selected["date"] < selected["cutoff_day"]
            selected["period"] = np.where(baseline, "baseline", "followup")
            selected["period_start"] = np.where(
                baseline,
                selected["baseline_start_day"],
                selected["cutoff_day"],
            )
            selected["relative_week"] = (
                (selected["date"] - selected["period_start"]) // 7
            ).astype(int)
            selected["quiz_clicks"] = np.where(
                selected["activity_type"].eq("quiz"), selected["sum_click"], 0
            )
            selected["content_clicks"] = np.where(
                selected["activity_type"].isin(CONTENT_TYPES),
                selected["sum_click"],
                0,
            )
            daily = selected.groupby(
                ["record_id", "period", "relative_week", "date"],
                as_index=False,
                sort=False,
            ).agg(
                total_clicks=("sum_click", "sum"),
                quiz_clicks=("quiz_clicks", "sum"),
                content_clicks=("content_clicks", "sum"),
            )
            daily_parts[stage].append(daily)

    result: dict[str, pd.DataFrame] = {}
    for stage, parts in daily_parts.items():
        if not parts:
            result[stage] = pd.DataFrame(
                columns=[
                    "record_id",
                    "period",
                    "relative_week",
                    "total_clicks",
                    "quiz_clicks",
                    "content_clicks",
                    "active_days",
                ]
            )
            continue
        daily = pd.concat(parts, ignore_index=True)
        daily = daily.groupby(
            ["record_id", "period", "relative_week", "date"],
            as_index=False,
            sort=False,
        )[["total_clicks", "quiz_clicks", "content_clicks"]].sum()
        weekly = daily.groupby(
            ["record_id", "period", "relative_week"],
            as_index=False,
            sort=False,
        ).agg(
            total_clicks=("total_clicks", "sum"),
            quiz_clicks=("quiz_clicks", "sum"),
            content_clicks=("content_clicks", "sum"),
            active_days=("date", "nunique"),
        )
        result[stage] = weekly
    return result


def _weekly_vector(
    weekly: pd.DataFrame,
    record_id: str,
    period: str,
    week_count: int,
    column: str,
) -> np.ndarray:
    values = np.zeros(max(2, int(week_count)), dtype=np.float64)
    selected = weekly.loc[
        weekly["record_id"].eq(record_id) & weekly["period"].eq(period),
        ["relative_week", column],
    ]
    for row in selected.itertuples(index=False):
        index = int(row.relative_week)
        if 0 <= index < week_count:
            values[index] += float(getattr(row, column))
    return values


def _vle_measures(window: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in window.itertuples(index=False):
        baseline_days = int(item.cutoff_day - item.baseline_start_day)
        followup_days = int(item.followup_end_day - item.cutoff_day)
        baseline_weeks = max(2, int(np.ceil(baseline_days / 7.0)))
        followup_weeks = max(2, int(np.ceil(followup_days / 7.0)))
        payload: dict[str, Any] = {"record_id": item.record_id}
        for period, days, weeks in (
            ("baseline", baseline_days, baseline_weeks),
            ("followup", followup_days, followup_weeks),
        ):
            total = _weekly_vector(weekly, item.record_id, period, weeks, "total_clicks")
            active = _weekly_vector(weekly, item.record_id, period, weeks, "active_days")
            quiz = _weekly_vector(weekly, item.record_id, period, weeks, "quiz_clicks")
            content = _weekly_vector(
                weekly, item.record_id, period, weeks, "content_clicks"
            )
            payload[f"{period}__study_regularity_score"] = float(
                study_regularity_score(total[None, :])[0]
            )
            payload[f"{period}__vle_active_day_rate"] = float(
                np.clip(active.sum() / max(1, days), 0.0, 1.0)
            )
            payload[f"{period}__retrieval_practice_rate"] = float(
                np.mean(quiz[:weeks] > 0.0)
            )
            payload[f"{period}__content_review_coverage"] = float(
                np.mean(content[:weeks] > 0.0)
            )
        rows.append(payload)
    return pd.DataFrame(rows)


def _load_assessment_sources() -> tuple[dict[tuple[str, str], pd.DataFrame], dict[tuple[int, int], int]]:
    assessments = pd.read_csv(
        RAW / "assessments.csv",
        usecols=[
            "code_module",
            "code_presentation",
            "id_assessment",
            "assessment_type",
            "date",
        ],
    )
    assessments = assessments.loc[
        assessments["date"].notna() & ~assessments["assessment_type"].eq("Exam")
    ].copy()
    assessments["date"] = assessments["date"].astype(int)
    submissions = pd.read_csv(
        RAW / "studentAssessment.csv",
        usecols=["id_assessment", "id_student", "date_submitted"],
    )
    earliest = (
        submissions.groupby(["id_assessment", "id_student"], as_index=False)[
            "date_submitted"
        ]
        .min()
        .set_index(["id_assessment", "id_student"])["date_submitted"]
        .astype(int)
        .to_dict()
    )
    by_course = {
        (str(module), str(presentation)): group
        for (module, presentation), group in assessments.groupby(
            ["code_module", "code_presentation"], sort=False
        )
    }
    return by_course, earliest


def _assessment_measures(
    window: pd.DataFrame,
    by_course: dict[tuple[str, str], pd.DataFrame],
    earliest: dict[tuple[int, int], int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in window.itertuples(index=False):
        available = by_course.get((item.code_module, item.code_presentation))
        payload: dict[str, Any] = {"record_id": item.record_id}
        for period, start, end in (
            ("baseline", int(item.baseline_start_day), int(item.cutoff_day)),
            ("followup", int(item.cutoff_day), int(item.followup_end_day)),
        ):
            due = (
                available.loc[
                    (available["date"] >= start) & (available["date"] < end)
                ]
                if available is not None
                else pd.DataFrame()
            )
            due_count = int(len(due))
            complete = 0
            for assessment_id in due.get("id_assessment", pd.Series(dtype=int)):
                submitted = earliest.get((int(assessment_id), int(item.student_id)))
                if submitted is not None and int(submitted) < end:
                    complete += 1
            payload[f"{period}__assessment_completion_rate"] = (
                float(complete / due_count) if due_count else 0.0
            )
            payload[f"{period}__assessment_available"] = int(due_count > 0)
            payload[f"{period}__assessment_due_count"] = due_count
        rows.append(payload)
    return pd.DataFrame(rows)


def _behaviour_table(
    windows: dict[str, pd.DataFrame],
    weekly: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    by_course, earliest = _load_assessment_sources()
    result: dict[str, pd.DataFrame] = {}
    for stage, window in windows.items():
        vle = _vle_measures(window, weekly[stage])
        assessment = _assessment_measures(window, by_course, earliest)
        merged = window.merge(vle, on="record_id", validate="one_to_one").merge(
            assessment, on="record_id", validate="one_to_one"
        )
        measure_columns = [
            column
            for column in merged.columns
            if column.startswith("baseline__") or column.startswith("followup__")
        ]
        result[stage] = merged.loc[
            :,
            [
                "record_id",
                "module_length",
                "baseline_start_day",
                "cutoff_day",
                "followup_end_day",
                *measure_columns,
            ],
        ]
    return result


def _oof_embeddings(
    bundle: Any,
    stage: str,
    *,
    batch_size: int,
) -> pd.DataFrame:
    source = STAGE_SOURCE[stage]
    data = bundle.stages[source]
    row_parts: list[pd.DataFrame] = []
    for fold in sorted(data.frame["outer_fold"].astype(int).unique()):
        index = np.flatnonzero(data.frame["outer_fold"].to_numpy(dtype=int) == fold)
        if not len(index):
            continue
        adapter = HybridPredictionAdapter.from_manifest(
            ROOT,
            stage=STAGE_ENUM[stage],
            fold=int(fold),
        )
        checkpoint = ROOT / adapter.checkpoint_references[0].path
        if file_sha256(checkpoint) != adapter.checkpoint_references[0].sha256:
            raise RuntimeError("checkpoint changed after adapter validation")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        preprocessor = _restore_preprocessor(payload["preprocessor"])
        frame = data.frame.iloc[index].copy().reset_index(drop=True)
        aggregate, static = preprocessor.transform(frame, data.aggregate[index])
        student_embedding: list[np.ndarray] = []
        tabular_embedding: list[np.ndarray] = []
        risk_probability: list[np.ndarray] = []
        for start in range(0, len(index), batch_size):
            stop = min(start + batch_size, len(index))
            inputs = {
                "sequence": torch.from_numpy(data.sequence[index[start:stop]]),
                "lengths": torch.from_numpy(
                    data.lengths[index[start:stop]].astype(np.int64)
                ),
                "mask": torch.from_numpy(
                    data.mask[index[start:stop]].astype(np.float32)
                ),
                "aggregate": torch.from_numpy(aggregate[start:stop]),
                "static": torch.from_numpy(static[start:stop]),
            }
            output = adapter.predict(inputs)
            student_embedding.append(output.student_state_embedding.cpu().numpy())
            tabular_embedding.append(output.tabular_expert_embedding.cpu().numpy())
            risk_probability.append(output.probabilities[:, 1].cpu().numpy())
        student = np.concatenate(student_embedding, axis=0)
        tabular = np.concatenate(tabular_embedding, axis=0)
        risk = np.concatenate(risk_probability, axis=0)
        matrix = np.column_stack([student, tabular]).astype(np.float32)
        columns = [
            *[f"embedding__student_{column:03d}" for column in range(student.shape[1])],
            *[f"embedding__tabular_{column:03d}" for column in range(tabular.shape[1])],
        ]
        rows = pd.DataFrame(matrix, columns=columns)
        rows.insert(0, "checkpoint_outer_fold", int(fold))
        rows.insert(0, "prediction_risk_probability", risk.astype(np.float32))
        rows.insert(0, "record_id", frame["base_record_id"].astype(str).to_numpy())
        row_parts.append(rows)
    output = pd.concat(row_parts, ignore_index=True)
    if output["record_id"].duplicated().any() or len(output) != len(data.frame):
        raise RuntimeError(f"{stage}: OOF embedding coverage is incomplete")
    return output


def _base_features(
    bundle: Any,
    stage: str,
    behaviour: pd.DataFrame,
    embedding: pd.DataFrame,
) -> pd.DataFrame:
    source = STAGE_SOURCE[stage]
    frame = bundle.stages[source].frame.copy()
    keep = [
        "base_record_id",
        "id_student",
        "code_module",
        "code_presentation",
        "outer_fold",
        "target",
        "num_of_prev_attempts",
        "studied_credits",
        "registration_lead_time",
        "module_presentation_length",
        "progress_fraction",
        "observed_week_count",
        "weeks_remaining",
        "assessment_available_fraction",
    ]
    base = frame.loc[:, keep].rename(columns={"base_record_id": "record_id"})
    base["record_id"] = base["record_id"].astype(str)
    base = base.merge(behaviour, on="record_id", validate="one_to_one")
    base = base.merge(embedding, on="record_id", validate="one_to_one")
    if not np.array_equal(
        base["outer_fold"].to_numpy(dtype=int),
        base["checkpoint_outer_fold"].to_numpy(dtype=int),
    ):
        raise RuntimeError(f"{stage}: OOF checkpoint fold mismatch")
    base["protocol_split"] = base["outer_fold"].map(PROTOCOL_SPLIT_BY_OUTER_FOLD)
    base["student_id"] = base["id_student"].astype(str)
    base["course_id"] = (
        base["code_module"].astype(str)
        + "::"
        + base["code_presentation"].astype(str)
    )
    base["outcome_pass"] = 1 - base["target"].astype(int)
    base["prediction_target"] = base["target"].astype(int)
    # The canonical day windows are integer-quantized per presentation.  Store
    # the preregistered nominal fractions for protocol validation instead of
    # aggregating presentation-specific rounding across rows; the raw day
    # windows above remain the source of baseline/treatment measurements.
    cutoff_fraction = LANDMARK_STAGES[stage].cutoff_fraction
    base["baseline_progress"] = np.full(len(base), cutoff_fraction, dtype=np.float64)
    base["treatment_start_progress"] = np.full(
        len(base), cutoff_fraction + 1.0e-6, dtype=np.float64
    )
    base["treatment_end_progress"] = (
        base["followup_end_day"] / base["module_length"]
    ).astype(np.float32)
    return base


def _expand_actions(base: pd.DataFrame, stage: str) -> pd.DataFrame:
    measure = {
        "ASSESSMENT_COMPLETION": "assessment_completion_rate",
        "STUDY_REGULARITY": "study_regularity_score",
        "VLE_ENGAGEMENT": "vle_active_day_rate",
        "QUIZ_OR_RETRIEVAL_PRACTICE": "retrieval_practice_rate",
        "CONTENT_REVIEW": "content_review_coverage",
    }
    baseline_measure_columns = [
        "assessment_completion_rate",
        "study_regularity_score",
        "vle_active_day_rate",
        "retrieval_practice_rate",
        "content_review_coverage",
    ]
    embedding_columns = sorted(
        column for column in base.columns if column.startswith("embedding__")
    )
    feature_payload = {
        f"feature__{column.removeprefix('embedding__')}": base[column].to_numpy(
            dtype=np.float32
        )
        for column in embedding_columns
    }
    feature_payload.update(
        {
            "feature__risk_probability": base[
                "prediction_risk_probability"
            ].to_numpy(dtype=np.float32),
            "feature__previous_attempts": pd.to_numeric(
                base["num_of_prev_attempts"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=np.float32),
            "feature__studied_credits": pd.to_numeric(
                base["studied_credits"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=np.float32),
            "feature__registration_lead_time": pd.to_numeric(
                base["registration_lead_time"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=np.float32),
            "feature__course_progress": base["progress_fraction"].to_numpy(
                dtype=np.float32
            ),
            "feature__observed_week_count": base[
                "observed_week_count"
            ].to_numpy(dtype=np.float32),
            "feature__weeks_remaining": base["weeks_remaining"].to_numpy(
                dtype=np.float32
            ),
            "feature__assessment_available_fraction": base[
                "assessment_available_fraction"
            ].to_numpy(dtype=np.float32),
            "feature__baseline_assessment_available": base[
                "baseline__assessment_available"
            ].to_numpy(dtype=np.float32),
        }
    )
    for name in baseline_measure_columns:
        feature_payload[f"feature__baseline_{name}"] = base[
            f"baseline__{name}"
        ].to_numpy(dtype=np.float32)
    base = pd.concat([base, pd.DataFrame(feature_payload, index=base.index)], axis=1)

    rows: list[pd.DataFrame] = []
    for action_id in ACTION_ORDER:
        selected = (
            base.loc[base["followup__assessment_available"].eq(1)].copy()
            if action_id == "ASSESSMENT_COMPLETION"
            else base.copy()
        )
        if selected.empty:
            continue
        name = measure[action_id]
        selected["stage"] = stage
        selected["action_id"] = action_id
        selected["baseline_measure"] = selected[f"baseline__{name}"].astype(
            np.float32
        )
        selected["followup_measure"] = selected[f"followup__{name}"].astype(
            np.float32
        )
        rows.append(selected)
    return pd.concat(rows, ignore_index=True) if rows else base.iloc[0:0].copy()


def _checkpoint_hashes() -> dict[str, list[str]]:
    manifest = json.loads(CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))
    result: dict[str, list[str]] = {}
    for stage in STAGE_SOURCE:
        result[stage] = sorted(
            {
                str(row["sha256"])
                for row in manifest["checkpoints"]
                if stage in row["stages"]
            }
        )
        if not result[stage]:
            raise RuntimeError(f"checkpoint manifest does not cover {stage}")
    return result


def build(
    output_path: Path = OUTPUT,
    manifest_path: Path = MANIFEST,
    *,
    chunksize: int = 750_000,
    batch_size: int = 512,
    force_bundle: bool = False,
) -> pd.DataFrame:
    bundle = build_canonical_bundle(force=force_bundle)
    windows = _windows(bundle)
    weekly = _collect_weekly_activity(windows, chunksize=chunksize)
    behaviour = _behaviour_table(windows, weekly)
    stage_rows: list[pd.DataFrame] = []
    for stage in STAGE_SOURCE:
        embedding = _oof_embeddings(bundle, stage, batch_size=batch_size)
        base = _base_features(bundle, stage, behaviour[stage], embedding)
        stage_rows.append(_expand_actions(base, stage))
    output = pd.concat(stage_rows, ignore_index=True)
    if output.empty:
        raise RuntimeError("no landmark rows were generated")
    if output.duplicated(["record_id", "stage", "action_id"]).any():
        raise RuntimeError("duplicate record-stage-action rows were generated")
    if output.groupby("student_id")["protocol_split"].nunique().max() != 1:
        raise RuntimeError("student-level protocol split leakage detected")
    feature_columns = sorted(
        column for column in output.columns if column.startswith("feature__")
    )
    embedding_columns = sorted(
        column for column in output.columns if column.startswith("embedding__")
    )
    if any("followup" in column for column in feature_columns):
        raise RuntimeError("post-cutoff information entered the confounder feature set")
    if not np.isfinite(output.loc[:, feature_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("non-finite confounder features were generated")
    if not np.isfinite(output.loc[:, embedding_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("non-finite frozen embeddings were generated")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    payload = {
        "status": "COMPLETE",
        "output": str(output_path.relative_to(ROOT)),
        "sha256": _sha256(output_path),
        "row_count": int(len(output)),
        "record_count": int(output["record_id"].nunique()),
        "student_count": int(output["student_id"].nunique()),
        "stage_counts": {
            str(key): int(value)
            for key, value in output.groupby("stage").size().items()
        },
        "action_counts": {
            str(key): int(value)
            for key, value in output.groupby("action_id").size().items()
        },
        "feature_count": len(feature_columns),
        "embedding_count": len(embedding_columns),
        "architecture_hash": ARCHITECTURE_HASH,
        "checkpoint_hashes": _checkpoint_hashes(),
        "embedding_authority": "FROZEN_OOF_HYBRID_SEED_ENSEMBLE",
        "protocol_split_by_outer_fold": PROTOCOL_SPLIT_BY_OUTER_FOLD,
        "cluster_key": "student_id",
        "record_key": "record_id",
        "post_cutoff_use": "TREATMENT_DEFINITION_ONLY",
        "outcome_in_features": False,
        "withdrawals_conditioned_out": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--chunksize", type=int, default=750_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--force-bundle", action="store_true")
    args = parser.parse_args()
    frame = build(
        args.output,
        args.manifest,
        chunksize=args.chunksize,
        batch_size=args.batch_size,
        force_bundle=args.force_bundle,
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "rows": len(frame),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
