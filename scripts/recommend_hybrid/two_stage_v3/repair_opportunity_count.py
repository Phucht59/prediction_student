"""Repair the omitted ``opportunity_count`` column without changing labels.

The Hybrid-only silver builder already computed the exact action opportunity
count from the published course schedule before creating each candidate row,
but the value was accidentally omitted from the serialized row dictionary.
This migration recomputes that same cutoff-safe schedule quantity and appends
only the missing column.  It never reads V2.1 artifacts or future behaviour.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    ROOT
    / "artifacts/recommend_hybrid/hybrid_only_final/dataset/candidate_rows.parquet"
)
SCHEMA = ROOT / "artifacts/recommend_hybrid/hybrid_only_final/dataset/schema.json"
CHECKSUMS = (
    ROOT / "artifacts/recommend_hybrid/hybrid_only_final/dataset/CHECKSUMS.json"
)
AUDIT = (
    ROOT
    / "artifacts/recommend_hybrid/two_stage_v3/OPPORTUNITY_COUNT_REPAIR.json"
)
sys.path.insert(0, str(ROOT))

from src.pipelines.oulad import _build_bundle  # noqa: E402

STAGE_TRANSITIONS = {
    "EARLY_20": ("E1_EARLY_20PCT", "E2_EARLY_35PCT"),
    "EARLY_35": ("E2_EARLY_35PCT", "M1_MIDDLE_FROZEN"),
    "MIDDLE_50": ("M1_MIDDLE_FROZEN", "L1_LATE_75PCT"),
}
QUIZ_TYPES = {"quiz", "externalquiz", "questionnaire"}
CONTENT_TYPES = {
    "resource",
    "oucontent",
    "page",
    "subpage",
    "url",
    "folder",
    "glossary",
}
EXPECTED_ROWS = 82_847
EXPECTED_GROUPS = 29_043
EXPECTED_POSITIVE_GROUPS = 9_304


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _vle_opportunity(
    schedule: pd.DataFrame,
    start_day: float,
    stop_day: float,
    activity_types: set[str] | None = None,
) -> int:
    if schedule.empty:
        return 0
    start_week = start_day / 7.0
    stop_week = stop_day / 7.0
    week_from = schedule["week_from"].fillna(schedule["week_to"])
    week_to = schedule["week_to"].fillna(schedule["week_from"])
    selected = schedule[(week_to >= start_week) & (week_from <= stop_week)]
    if activity_types is not None:
        selected = selected[selected["activity_type"].isin(activity_types)]
    return int(len(selected))


def _action_opportunity(
    *,
    action_family: str,
    cutoff_day: float,
    target_day: float,
    assessment_dates: np.ndarray,
    vle_schedule: pd.DataFrame,
) -> int:
    if action_family == "ASSESSMENT_COMPLETION":
        if not len(assessment_dates):
            return 0
        return int(
            ((assessment_dates > cutoff_day) & (assessment_dates <= target_day)).sum()
        )
    if action_family == "STUDY_REGULARITY":
        return max(1, int(target_day - cutoff_day))
    if action_family == "VLE_ENGAGEMENT":
        return _vle_opportunity(vle_schedule, cutoff_day, target_day)
    if action_family == "QUIZ_OR_RETRIEVAL_PRACTICE":
        return _vle_opportunity(
            vle_schedule,
            cutoff_day,
            target_day,
            QUIZ_TYPES,
        )
    if action_family == "CONTENT_REVIEW":
        return _vle_opportunity(
            vle_schedule,
            cutoff_day,
            target_day,
            CONTENT_TYPES,
        )
    raise RuntimeError(f"unknown action family: {action_family}")


def _stage_lookup(bundle: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for stage_name, (source_stage, target_stage) in STAGE_TRANSITIONS.items():
        source = bundle.stages[source_stage].frame
        target = bundle.stages[target_stage].frame
        source_rows = {
            str(row.base_record_id): row
            for row in source.itertuples(index=False)
        }
        target_rows = {
            str(row.base_record_id): row
            for row in target.itertuples(index=False)
        }
        result[stage_name] = {
            "source": source_rows,
            "target": target_rows,
        }
    return result


def _validate_authority(frame: pd.DataFrame) -> None:
    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(
            f"candidate row authority changed: {len(frame)} != {EXPECTED_ROWS}"
        )
    if frame["group_id"].nunique() != EXPECTED_GROUPS:
        raise RuntimeError("ranking-group authority changed")
    positive = frame.groupby("group_id", sort=False)["silver_positive"].max()
    if int((positive > 0).sum()) != EXPECTED_POSITIVE_GROUPS:
        raise RuntimeError("positive-group label authority changed")
    if frame.duplicated(["group_id", "action_family"]).any():
        raise RuntimeError("duplicate action family inside a ranking group")


def _update_schema() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    runtime = list(schema.get("runtime_features", []))
    if "opportunity_count" not in runtime:
        insert_at = runtime.index("workload_minutes") if "workload_minutes" in runtime else len(runtime)
        runtime.insert(insert_at, "opportunity_count")
    schema["runtime_features"] = runtime
    schema["opportunity_count_authority"] = {
        "source": "published assessment and VLE schedules",
        "cutoff_safe": True,
        "future_behaviour_used": False,
        "repair": "serialization omission only",
    }
    _atomic_json(SCHEMA, schema)


def _update_checksums() -> None:
    directory = SOURCE.parent
    files = [item for item in directory.iterdir() if item.is_file() and item != CHECKSUMS]
    _atomic_json(
        CHECKSUMS,
        {
            str(item.relative_to(ROOT)).replace("\\", "/"): _sha256(item)
            for item in files
        },
    )


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    frame = pd.read_parquet(SOURCE)
    _validate_authority(frame)
    old_sha = _sha256(SOURCE)
    old_columns = list(frame.columns)
    old_snapshot = frame.copy(deep=True)

    if "opportunity_count" in frame.columns:
        values = pd.to_numeric(frame["opportunity_count"], errors="coerce")
        if values.isna().any() or (values <= 0).any():
            raise RuntimeError("existing opportunity_count is invalid")
        status = "ALREADY_REPAIRED"
    else:
        assessments = pd.read_csv(
            ROOT / "data/raw/assessments.csv",
            usecols=["code_module", "code_presentation", "date", "weight"],
        )
        assessments = assessments[assessments["weight"] > 0]
        assessment_map = {
            (str(module), str(presentation)): pd.to_numeric(
                group["date"], errors="coerce"
            ).dropna().to_numpy(dtype=np.float64)
            for (module, presentation), group in assessments.groupby(
                ["code_module", "code_presentation"], sort=False
            )
        }
        vle = pd.read_csv(
            ROOT / "data/raw/vle.csv",
            usecols=[
                "code_module",
                "code_presentation",
                "activity_type",
                "week_from",
                "week_to",
            ],
        )
        vle_map = {
            (str(module), str(presentation)): group.reset_index(drop=True)
            for (module, presentation), group in vle.groupby(
                ["code_module", "code_presentation"], sort=False
            )
        }
        bundle = _build_bundle()
        lookup = _stage_lookup(bundle)
        group_metadata = frame[
            [
                "group_id",
                "base_record_id",
                "stage",
                "course",
                "presentation",
            ]
        ].drop_duplicates("group_id")
        group_context: dict[str, tuple[float, float, np.ndarray, pd.DataFrame]] = {}
        for row in group_metadata.itertuples(index=False):
            stage_rows = lookup[str(row.stage)]
            source_row = stage_rows["source"].get(str(row.base_record_id))
            target_row = stage_rows["target"].get(str(row.base_record_id))
            if source_row is None or target_row is None:
                raise RuntimeError(f"group missing from bundle: {row.group_id}")
            key = (str(row.course), str(row.presentation))
            group_context[str(row.group_id)] = (
                float(source_row.cutoff_day),
                float(target_row.cutoff_day),
                assessment_map.get(key, np.empty(0, dtype=np.float64)),
                vle_map.get(key, pd.DataFrame()),
            )

        opportunities: list[int] = []
        for row in frame.itertuples(index=False):
            cutoff_day, target_day, assessment_dates, vle_schedule = group_context[
                str(row.group_id)
            ]
            opportunities.append(
                _action_opportunity(
                    action_family=str(row.action_family),
                    cutoff_day=cutoff_day,
                    target_day=target_day,
                    assessment_dates=assessment_dates,
                    vle_schedule=vle_schedule,
                )
            )
        frame["opportunity_count"] = np.asarray(opportunities, dtype=np.int32)
        if (frame["opportunity_count"] <= 0).any():
            bad = frame.loc[
                frame["opportunity_count"] <= 0,
                ["group_id", "action_family"],
            ].head(10)
            raise RuntimeError(
                "eligible candidate received zero opportunities: "
                + bad.to_dict(orient="records").__repr__()
            )
        pd.testing.assert_frame_equal(
            old_snapshot,
            frame[old_columns],
            check_exact=True,
            check_dtype=True,
        )
        _atomic_parquet(SOURCE, frame)
        _update_schema()
        _update_checksums()
        status = "REPAIRED"

    repaired = pd.read_parquet(SOURCE)
    _validate_authority(repaired)
    pd.testing.assert_frame_equal(
        old_snapshot,
        repaired[old_columns],
        check_exact=True,
        check_dtype=True,
    )
    values = pd.to_numeric(repaired["opportunity_count"], errors="raise")
    audit = {
        "schema_version": "two_stage_v3_opportunity_repair_v1",
        "status": status,
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "old_sha256": old_sha,
        "new_sha256": _sha256(SOURCE),
        "rows": int(len(repaired)),
        "groups": int(repaired["group_id"].nunique()),
        "positive_groups": int(
            (
                repaired.groupby("group_id", sort=False)["silver_positive"].max()
                > 0
            ).sum()
        ),
        "labels_changed": False,
        "existing_columns_changed": False,
        "v2_1_artifacts_used": False,
        "future_behaviour_used": False,
        "schedule_authority": "published assessment and VLE schedules",
        "minimum_opportunity_count": int(values.min()),
        "maximum_opportunity_count": int(values.max()),
        "mean_opportunity_count": float(values.mean()),
    }
    _atomic_json(AUDIT, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
