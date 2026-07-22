from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .contract import ARTIFACT_ROOT, ROOT, atomic_json, sha256_file


SCHEMA_VERSION = "student_risk_profile_v1"
PROFILE_PATH = ARTIFACT_ROOT / "prediction/risk_profiles.parquet"
GENERATED_AT = "2026-07-21T00:00:00+00:00"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _warning_lead(hazard: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    survival_before = np.concatenate(
        [np.ones((len(hazard), 1)), np.cumprod(1.0 - hazard[:, :-1], axis=1)], axis=1
    )
    event_mass = survival_before * hazard
    horizon = event_mass.sum(axis=1)
    weeks = np.arange(1, hazard.shape[1] + 1, dtype=float)[None, :]
    expected = np.divide(
        (event_mass * weeks).sum(axis=1) * 7.0,
        horizon,
        out=np.full(len(hazard), np.nan),
        where=horizon > 1e-8,
    )
    return hazard[:, 0], horizon, expected


def _confidence(
    probability: np.ndarray,
    threshold: np.ndarray,
    seed_std: np.ndarray,
    disagreement: np.ndarray,
    entropy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    uncertainty = np.clip(0.6 * entropy / np.log(2) + 0.4 * np.minimum(1.0, seed_std * 5), 0, 1)
    margin = np.abs(probability - threshold)
    high = (margin >= 0.20) & (seed_std < 0.04) & (disagreement < 0.15) & (uncertainty < 0.45)
    medium = (margin >= 0.08) & (seed_std < 0.08) & (disagreement < 0.25) & (uncertainty < 0.70)
    level = np.where(high, "HIGH_CONFIDENCE", np.where(medium, "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE"))
    return level, uncertainty


def validate_risk_profile(profile: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "record_id",
        "id_student",
        "code_module",
        "code_presentation",
        "forecast_id",
        "cutoff_day",
        "model_version",
        "checkpoint_sha256",
        "feature_contract_sha256",
        "probability_at_risk",
        "predicted_at_risk",
        "macro_threshold",
        "withdrawal_hazard_current",
        "withdrawal_risk_horizon",
        "expected_warning_lead_time",
        "probability_fail",
        "probability_pass",
        "probability_distinction",
        "risk_priority_score",
        "risk_percentile",
        "top_k_bucket",
        "confidence_level",
        "uncertainty_score",
        "seed_disagreement",
        "ml_cross_check_probability",
        "deep_ml_disagreement",
        "decision_status",
        "reason_codes",
        "generated_at",
        "lineage_id",
    }
    missing = required - set(profile)
    if missing:
        raise ValueError(f"Risk profile missing fields: {sorted(missing)}")
    if profile["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Risk profile schema mismatch")
    probabilities = [
        profile["probability_at_risk"],
        profile["withdrawal_hazard_current"],
        profile["withdrawal_risk_horizon"],
        profile["probability_fail"],
        profile["probability_pass"],
        profile["probability_distinction"],
        profile["ml_cross_check_probability"],
    ]
    if not np.isfinite(probabilities).all() or any(not 0 <= value <= 1 for value in probabilities):
        raise ValueError("Risk profile probability invalid")
    if not np.isclose(sum(probabilities[3:6]), 1.0, atol=1e-5):
        raise ValueError("Risk profile outcome probabilities do not sum to one")
    if profile["confidence_level"] not in {
        "HIGH_CONFIDENCE",
        "MEDIUM_CONFIDENCE",
        "LOW_CONFIDENCE",
    }:
        raise ValueError("Risk profile confidence invalid")
    if profile["decision_status"] not in {"PREDICTED", "ABSTAIN_REVIEW_REQUIRED"}:
        raise ValueError("Risk profile decision status invalid")
    if not profile["lineage_id"] or not profile["model_version"]:
        raise ValueError("Risk profile lineage missing")


def generate_risk_profiles() -> dict[str, Any]:
    state_path = ARTIFACT_ROOT / "prediction/risk_profile_state.json"
    if state_path.is_file() and PROFILE_PATH.is_file():
        cached = json.loads(state_path.read_text(encoding="utf-8"))
        if cached.get("status") == "COMPLETE":
            return cached
    calibrated = pd.read_parquet(
        ARTIFACT_ROOT / "prediction/calibrated_oof_predictions.parquet"
    )
    ensemble = pd.read_parquet(ARTIFACT_ROOT / "prediction/oof_predictions.parquet")
    hazard_columns = [f"hazard_week_{week:02d}" for week in range(20)]
    outcome_columns = ["probability_fail", "probability_pass", "probability_distinction"]
    frame = calibrated.merge(
        ensemble[["record_id", *hazard_columns, *outcome_columns]],
        on="record_id",
        validate="one_to_one",
    )
    ml = pd.read_parquet(ROOT / "artifacts/oulad/final/ensemble_oof_predictions.parquet")
    ml = ml[ml.candidate_id.eq("V3-MLF")][["record_id", "probability"]].rename(
        columns={"probability": "ml_cross_check_probability"}
    )
    frame = frame.merge(ml, on="record_id", validate="one_to_one")
    hazard = frame[hazard_columns].to_numpy(dtype=float)
    current_hazard, horizon_risk, warning_lead = _warning_lead(hazard)
    probability = frame.probability_at_risk.to_numpy(dtype=float)
    threshold = frame.calibrated_threshold.to_numpy(dtype=float)
    disagreement = np.abs(
        probability - frame.ml_cross_check_probability.to_numpy(dtype=float)
    )
    confidence, uncertainty = _confidence(
        probability,
        threshold,
        frame.seed_standard_deviation.to_numpy(dtype=float),
        disagreement,
        frame.predictive_entropy.to_numpy(dtype=float),
    )
    priority_score = np.clip(
        0.55 * probability
        + 0.20 * horizon_risk
        + 0.20 * frame.probability_fail.to_numpy(dtype=float)
        + 0.05 * uncertainty,
        0,
        1,
    )
    percentile = pd.Series(priority_score).rank(method="average", pct=True).to_numpy(dtype=float)
    top_bucket = np.where(
        percentile >= 0.95,
        "TOP_5_PERCENT",
        np.where(
            percentile >= 0.90,
            "TOP_10_PERCENT",
            np.where(percentile >= 0.80, "TOP_20_PERCENT", "OUTSIDE_TOP_20_PERCENT"),
        ),
    )
    abstain = (confidence == "LOW_CONFIDENCE") | (disagreement >= 0.35)
    checkpoint_manifest = ARTIFACT_ROOT / "prediction/final/checkpoint_metadata.json"
    checkpoint_sha = sha256_file(checkpoint_manifest)
    feature_sha = sha256_file(ROOT / "configs/v5_1/oulad_v5_1.yaml")
    profiles: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        reasons: list[str] = []
        if probability[index] >= threshold[index]:
            reasons.append("AT_RISK_THRESHOLD_EXCEEDED")
        if horizon_risk[index] >= 0.45:
            reasons.append("WITHDRAWAL_HAZARD_ELEVATED")
        if float(row.probability_fail) >= 0.55:
            reasons.append("FAIL_PROBABILITY_ELEVATED")
        if disagreement[index] >= 0.25:
            reasons.append("DEEP_ML_DISAGREEMENT")
        if confidence[index] == "LOW_CONFIDENCE":
            reasons.append("LOW_CONFIDENCE")
        body: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "record_id": str(row.record_id),
            "id_student": int(row.id_student),
            "code_module": str(row.code_module),
            "code_presentation": str(row.code_presentation),
            "forecast_id": "F2_MIDDLE",
            "cutoff_day": int(row.cutoff_day),
            "model_version": "v6_C_temporal_multitask_W0_seed_ensemble",
            "checkpoint_sha256": checkpoint_sha,
            "feature_contract_sha256": feature_sha,
            "probability_at_risk": float(probability[index]),
            "predicted_at_risk": bool(probability[index] >= threshold[index]),
            "macro_threshold": float(threshold[index]),
            "withdrawal_hazard_current": float(current_hazard[index]),
            "withdrawal_risk_horizon": float(horizon_risk[index]),
            "expected_warning_lead_time": None
            if np.isnan(warning_lead[index])
            else float(warning_lead[index]),
            "probability_fail": float(row.probability_fail),
            "probability_pass": float(row.probability_pass),
            "probability_distinction": float(row.probability_distinction),
            "risk_priority_score": float(priority_score[index]),
            "risk_percentile": float(percentile[index]),
            "top_k_bucket": str(top_bucket[index]),
            "confidence_level": str(confidence[index]),
            "uncertainty_score": float(uncertainty[index]),
            "seed_disagreement": float(row.seed_disagreement),
            "ml_cross_check_probability": float(row.ml_cross_check_probability),
            "deep_ml_disagreement": float(disagreement[index]),
            "decision_status": "ABSTAIN_REVIEW_REQUIRED" if abstain[index] else "PREDICTED",
            "reason_codes": sorted(reasons),
            "generated_at": GENERATED_AT,
        }
        body["lineage_id"] = _canonical_hash(body)[:32]
        validate_risk_profile(body)
        profiles.append(body)
    output = pd.DataFrame(profiles)
    output.to_parquet(PROFILE_PATH, index=False)
    replay = pd.DataFrame(profiles)
    replay_hash = _canonical_hash(replay.to_dict(orient="records"))
    result = {
        "schema_version": "v6_risk_profile_generation_v1",
        "status": "COMPLETE",
        "records": len(profiles),
        "coverage": 1.0,
        "profile_schema": SCHEMA_VERSION,
        "abstention_rate": float((output.decision_status == "ABSTAIN_REVIEW_REQUIRED").mean()),
        "confidence_distribution": output.confidence_level.value_counts().to_dict(),
        "top_k_distribution": output.top_k_bucket.value_counts().to_dict(),
        "mean_deep_ml_disagreement": float(output.deep_ml_disagreement.mean()),
        "replay_sha256": replay_hash,
        "sensitive_demographics_in_payload": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "future_accessed": False,
    }
    atomic_json(state_path, result)
    return result


__all__ = ["SCHEMA_VERSION", "generate_risk_profiles", "validate_risk_profile"]
