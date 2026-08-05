"""Prepare causal and imbalance archives from one audited landmark table."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.causal.protocol import STAGE_ORDER  # noqa: E402
from src.recommend_hybrid.causal.treatments import (  # noqa: E402
    fit_action_treatment_rule,
)
from src.recommend_hybrid.final.actions import ACTION_ORDER  # noqa: E402

DEFAULT_INPUT = ROOT / "artifacts/recommend_hybrid/causal/input/landmark_rows.parquet"
DEFAULT_OUTPUT = ROOT / "artifacts/recommend_hybrid/causal/input"
REQUIRED_COLUMNS = {
    "record_id",
    "student_id",
    "course_id",
    "stage",
    "action_id",
    "protocol_split",
    "outcome_pass",
    "prediction_target",
    "baseline_progress",
    "treatment_start_progress",
    "treatment_end_progress",
    "baseline_measure",
    "followup_measure",
}
ALLOWED_SPLITS = ("train", "validation", "test")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError("input must be Parquet or CSV")


def _feature_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = sorted(column for column in frame.columns if column.startswith(prefix))
    if not columns:
        raise ValueError(f"no columns found with prefix {prefix!r}")
    values = frame.loc[:, columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{prefix} columns contain missing or non-finite values")
    return columns


def _validate_frame(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise KeyError(f"landmark table is missing columns: {missing}")
    if frame.empty:
        raise ValueError("landmark table is empty")
    if frame.duplicated(["record_id", "stage", "action_id"]).any():
        raise ValueError("landmark table contains duplicate record-stage-action rows")
    split_count = frame.groupby("student_id", sort=False)["protocol_split"].nunique()
    if int(split_count.max()) != 1:
        raise ValueError("one student appears in multiple protocol splits")
    unknown_stage = sorted(set(frame["stage"].astype(str)).difference(STAGE_ORDER))
    unknown_action = sorted(set(frame["action_id"].astype(str)).difference(ACTION_ORDER))
    unknown_split = sorted(set(frame["protocol_split"].astype(str)).difference(ALLOWED_SPLITS))
    if unknown_stage or unknown_action or unknown_split:
        raise ValueError(
            "unsupported stage/action/split values: "
            f"{unknown_stage}, {unknown_action}, {unknown_split}"
        )
    for column in ("outcome_pass", "prediction_target"):
        values = frame[column].to_numpy()
        if not np.isin(values, [0, 1]).all():
            raise ValueError(f"{column} must be binary")
    for column in (
        "baseline_progress",
        "treatment_start_progress",
        "treatment_end_progress",
        "baseline_measure",
        "followup_measure",
    ):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must be finite")
    feature_columns = _feature_columns(frame, "feature__")
    embedding_columns = _feature_columns(frame, "embedding__")
    return feature_columns, embedding_columns


def _fit_treatments(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    retained: list[pd.DataFrame] = []
    registry: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        for action_id in ACTION_ORDER:
            selected = frame.loc[
                frame["stage"].eq(stage) & frame["action_id"].eq(action_id)
            ].copy()
            if selected.empty:
                registry.append(
                    {
                        "stage": stage,
                        "action_id": action_id,
                        "status": "TRIAL_DATA_NOT_AVAILABLE",
                    }
                )
                continue
            train = selected.loc[selected["protocol_split"].eq("train")]
            try:
                rule = fit_action_treatment_rule(
                    action_id=action_id,
                    baseline_measure=train["baseline_measure"].to_numpy(dtype=np.float64),
                    followup_measure=train["followup_measure"].to_numpy(dtype=np.float64),
                )
            except ValueError as exc:
                registry.append(
                    {
                        "stage": stage,
                        "action_id": action_id,
                        "status": "TREATMENT_NOT_MEASURABLE",
                        "reason": str(exc),
                    }
                )
                continue
            selected["treatment"] = rule.assign(
                selected["baseline_measure"].to_numpy(dtype=np.float64),
                selected["followup_measure"].to_numpy(dtype=np.float64),
            )
            retained.append(selected)
            registry.append(
                {
                    "stage": stage,
                    "action_id": action_id,
                    "status": "TREATMENT_RULE_FITTED",
                    "rule": rule.to_dict(),
                    "row_count": int(len(selected)),
                    "treated_count": int(selected["treatment"].sum()),
                    "control_count": int((1 - selected["treatment"]).sum()),
                }
            )
    output = pd.concat(retained, ignore_index=True) if retained else frame.iloc[0:0].copy()
    return output, registry


def run(input_path: Path, output_dir: Path) -> dict[str, object]:
    source = _read_table(input_path)
    feature_columns, embedding_columns = _validate_frame(source)
    trials, registry = _fit_treatments(source)
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_path = output_dir / "target_trials.npz"
    np.savez_compressed(
        trial_path,
        features=trials.loc[:, feature_columns].to_numpy(dtype=np.float32),
        treatment=trials["treatment"].to_numpy(dtype=np.int8),
        outcome=trials["outcome_pass"].to_numpy(dtype=np.int8),
        groups=trials["student_id"].astype(str).to_numpy(dtype=str),
        student_ids=trials["student_id"].astype(str).to_numpy(dtype=str),
        record_ids=trials["record_id"].astype(str).to_numpy(dtype=str),
        course_ids=trials["course_id"].astype(str).to_numpy(dtype=str),
        stages=trials["stage"].astype(str).to_numpy(dtype=str),
        action_ids=trials["action_id"].astype(str).to_numpy(dtype=str),
        baseline_progress=trials["baseline_progress"].to_numpy(dtype=np.float32),
        treatment_start_progress=trials["treatment_start_progress"].to_numpy(dtype=np.float32),
        treatment_end_progress=trials["treatment_end_progress"].to_numpy(dtype=np.float32),
        protocol_splits=trials["protocol_split"].astype(str).to_numpy(dtype=str),
        feature_names=np.asarray(feature_columns, dtype=str),
    )

    embedding_frame = source.drop_duplicates(["record_id", "stage"]).copy()
    split_payload: dict[str, np.ndarray] = {}
    for split in ALLOWED_SPLITS:
        selected = embedding_frame.loc[embedding_frame["protocol_split"].eq(split)]
        if selected.empty:
            raise ValueError(f"no embedding rows available for protocol split {split}")
        split_payload[f"{split}_embeddings"] = selected.loc[
            :, embedding_columns
        ].to_numpy(dtype=np.float32)
        split_payload[f"{split}_target"] = selected["prediction_target"].to_numpy(
            dtype=np.int8
        )
        split_payload[f"{split}_student_ids"] = selected["student_id"].astype(
            str
        ).to_numpy(dtype=str)
        split_payload[f"{split}_record_ids"] = selected["record_id"].astype(
            str
        ).to_numpy(dtype=str)
    embedding_path = output_dir / "frozen_embeddings.npz"
    np.savez_compressed(
        embedding_path,
        train_embeddings=split_payload["train_embeddings"],
        train_target=split_payload["train_target"],
        validation_embeddings=split_payload["validation_embeddings"],
        validation_target=split_payload["validation_target"],
        test_embeddings=split_payload["test_embeddings"],
        test_target=split_payload["test_target"],
        train_student_ids=split_payload["train_student_ids"],
        validation_student_ids=split_payload["validation_student_ids"],
        test_student_ids=split_payload["test_student_ids"],
        train_record_ids=split_payload["train_record_ids"],
        validation_record_ids=split_payload["validation_record_ids"],
        test_record_ids=split_payload["test_record_ids"],
        embedding_names=np.asarray(embedding_columns, dtype=str),
    )

    registry_path = output_dir / "treatment_registry.json"
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "COMPLETE",
        "source": str(input_path.relative_to(ROOT)),
        "source_row_count": int(len(source)),
        "source_record_count": int(source["record_id"].nunique()),
        "source_student_count": int(source["student_id"].nunique()),
        "trial_row_count": int(len(trials)),
        "feature_count": len(feature_columns),
        "embedding_count": len(embedding_columns),
        "treatment_registry": str(registry_path.relative_to(ROOT)),
        "target_trials": str(trial_path.relative_to(ROOT)),
        "frozen_embeddings": str(embedding_path.relative_to(ROOT)),
        "treatment_rules_fitted_on": "TRAIN_ONLY",
        "synthetic_sampling_applied": False,
        "cluster_key": "student_id",
        "record_key": "record_id",
    }
    (output_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(args.input, args.output_dir)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
