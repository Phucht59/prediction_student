"""Frozen Phase 8 Student State feature contract for EBM/baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..weak_supervision.matrix import FINAL_ACTIONS

ACTION_KEYS = {
    "A1": "assessment_recovery",
    "A2": "re_engagement",
    "A3": "study_planning",
    "A4": "progress_monitoring",
    "A5": "retrieval_practice",
}
ACTION_TO_KEY = {value: key for key, value in ACTION_KEYS.items()}
APPROVED_FEATURES = (
    "stage_code",
    "risk_probability",
    "inactive_streak",
    "active_days_ratio",
    "recent_activity",
    "activity_trend",
    "assessment_completion",
    "missing_assessments",
    "quiz_activity",
    "vle_available",
)
IDENTITY_COLUMNS = frozenset({
    "case_id", "student_id", "record_id", "enrollment_identity", "outer_fold",
    "module", "presentation", "dataset", "panel",
})
FORBIDDEN_FEATURES = frozenset({
    "case_id", "student_id", "record_id", "enrollment_identity", "outer_fold",
    "prediction_source_version", "source_feature_version", "prediction_seed_count",
    "risk_band", "risk_band_source", "sampling_risk_band",
    "confidence", "entropy", "silver_confidence", "silver_entropy", "hard_label",
    "aggregator_type", "feasibility_status", "final_result", "target", "score",
    "course_progress",
})
STAGE_CODES = {"20pct": 20.0, "35pct": 35.0, "50pct": 50.0, "75pct": 75.0}
METADATA_COLUMNS = (
    "expected_relevance", "silver_confidence", "silver_entropy", "silver_status",
    "aggregator_type", "label_model_version", "phase6_source_manifest_version",
)


def load_phase8_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def audit_course_progress(frame: pd.DataFrame) -> dict:
    if "course_progress" not in frame.columns or "stage" not in frame.columns:
        return {"status": "MISSING", "exclude": True, "reason": "FEATURE_EXCLUDED_REDUNDANT_STAGE"}
    pairs = frame[["stage", "course_progress"]].dropna().drop_duplicates()
    stage_unique = frame.groupby("stage")["course_progress"].nunique()
    deterministic = bool((stage_unique <= 1).all()) and set(frame["stage"].astype(str)).issubset(STAGE_CODES)
    mapping = {str(stage): float(group["course_progress"].iloc[0]) for stage, group in frame.groupby("stage") if len(group)}
    expected = {stage: code / 100.0 for stage, code in STAGE_CODES.items()}
    matches_stage = deterministic and all(abs(mapping.get(stage, -1) - expected[stage]) < 1e-9 for stage in mapping)
    return {
        "status": "FEATURE_EXCLUDED_REDUNDANT_STAGE" if matches_stage or deterministic else "KEEP",
        "exclude": bool(matches_stage or deterministic),
        "deterministic_from_stage": deterministic,
        "matches_stage_fraction": matches_stage,
        "n_unique_progress": int(frame["course_progress"].nunique()),
        "mapping": mapping,
        "reason": "course_progress equals stage/100 and is not an independent progress measure",
    }


def encode_state_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    output["stage_code"] = frame["stage"].astype(str).map(STAGE_CODES)
    if output["stage_code"].isna().any():
        raise ValueError("state contains a non-recommendation stage")
    for column in APPROVED_FEATURES:
        if column == "stage_code":
            continue
        if column not in frame.columns:
            raise ValueError(f"Student State missing approved feature: {column}")
        if column == "vle_available":
            output[column] = frame[column].astype(bool).astype(float)
        else:
            output[column] = pd.to_numeric(frame[column], errors="coerce")
    leaked = set(output.columns) & FORBIDDEN_FEATURES
    if leaked:
        raise ValueError(f"forbidden features reached the model matrix: {sorted(leaked)}")
    if list(output.columns) != list(APPROVED_FEATURES):
        raise ValueError("feature order is not the frozen contract")
    if output.isna().any().any():
        raise ValueError("approved features contain NaN")
    return output[list(APPROVED_FEATURES)]


def assert_no_leakage_features(columns) -> None:
    bad = set(columns) & FORBIDDEN_FEATURES
    if bad:
        raise ValueError(f"identity/label metadata in feature matrix: {sorted(bad)}")


def validate_phase7_authority(manifest_path: Path, silver_path: Path, phase6_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    silver = pd.read_parquet(silver_path, columns=["case_id", "action_id", "silver_status", "expected_relevance"])
    if manifest.get("version") != "recommendation.phase7_manifest.v1":
        raise ValueError("Phase 7 manifest version is not frozen")
    if set(manifest.get("actions", [])) != set(FINAL_ACTIONS):
        raise ValueError("Phase 7 manifest actions do not match the frozen taxonomy")
    if len(silver) != 2500:
        raise ValueError("silver_labels.parquet is not the frozen 2,500-row table")
    if silver.duplicated(["case_id", "action_id"]).any():
        raise ValueError("silver labels have duplicate case-action rows")
    phase6 = json.loads(phase6_path.read_text(encoding="utf-8"))
    if phase6.get("version") != manifest.get("phase6_source_manifest_version"):
        raise ValueError("Phase 7 lineage does not match the Phase 6 source manifest")
    return manifest
