from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .contract import ARTIFACT_ROOT, sha256_file
from .expert import export_expert_casebook, import_expert_scores
from .recommendation import generate_plan, load_plans, validate_plan
from .risk_profile import validate_risk_profile


def _profiles() -> pd.DataFrame:
    return pd.read_parquet(ARTIFACT_ROOT / "prediction/risk_profiles.parquet")


def predict_student_risk(record_id: str) -> dict[str, Any]:
    rows = _profiles()
    selected = rows[rows.record_id.eq(record_id)]
    if len(selected) != 1:
        raise KeyError(f"Unknown or duplicate record_id: {record_id}")
    profile = selected.iloc[0].to_dict()
    validate_risk_profile(profile)
    return profile


def batch_predict_risk(forecast_id: str) -> list[dict[str, Any]]:
    rows = _profiles()
    selected = rows[rows.forecast_id.eq(forecast_id)]
    return [predict_student_risk(record_id) for record_id in selected.record_id]


def generate_recommendation(risk_profile_id: str) -> dict[str, Any]:
    rows = _profiles()
    selected = rows[rows.lineage_id.eq(risk_profile_id)]
    if len(selected) != 1:
        raise KeyError(f"Unknown risk profile lineage: {risk_profile_id}")
    return generate_plan(selected.iloc[0].to_dict())


def regenerate_recommendation(record_id: str, reason: str) -> dict[str, Any]:
    if not reason.strip():
        raise ValueError("Recommendation regeneration requires a reason")
    old = [plan for plan in load_plans() if plan["record_id"] == record_id]
    profile = predict_student_risk(record_id)
    plan = generate_plan(profile)
    if old:
        plan["supersedes_plan_id"] = old[-1]["plan_id"]
        plan["plan_version"] = int(old[-1]["plan_version"]) + 1
        plan["revision_reason"] = reason
        plan["plan_id"] = __import__("hashlib").sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    validate_plan(plan)
    return plan


def export_expert_casebook_service() -> dict[str, Any]:
    return export_expert_casebook()


def import_expert_scores_service(file: Path) -> dict[str, Any]:
    return import_expert_scores(file)


def trace_recommendation(plan_id: str) -> dict[str, Any]:
    plans = [plan for plan in load_plans() if plan["plan_id"] == plan_id]
    if len(plans) != 1:
        raise KeyError(f"Unknown or duplicate plan_id: {plan_id}")
    plan = plans[0]
    profile = predict_student_risk(plan["record_id"])
    return {
        "plan": plan,
        "risk_profile": profile,
        "prediction_checkpoint_manifest": "artifacts/v6/prediction/final/checkpoint_metadata.json",
        "calibration": "artifacts/v6/prediction/calibration.json",
        "policy_version": plan["policy_version"],
        "engine_version": plan["recommendation_engine_version"],
    }


def validate_lineage(plan_id: str) -> dict[str, Any]:
    trace = trace_recommendation(plan_id)
    plan, profile = trace["plan"], trace["risk_profile"]
    checks = {
        "record_id": plan["record_id"] == profile["record_id"],
        "risk_profile_lineage": plan["risk_profile_lineage_id"] == profile["lineage_id"],
        "model_version": plan["prediction_model_version"] == profile["model_version"],
        "checkpoint_manifest": profile["checkpoint_sha256"]
        == sha256_file(ARTIFACT_ROOT / "prediction/final/checkpoint_metadata.json"),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def replay_prediction(run_id: str) -> dict[str, Any]:
    selected = json.loads(
        (ARTIFACT_ROOT / "prediction/selected_model.json").read_text(encoding="utf-8")
    )
    if run_id not in {selected["candidate"], "v6_final_prediction"}:
        raise KeyError(f"Unknown prediction run: {run_id}")
    metadata = json.loads(
        (ARTIFACT_ROOT / "prediction/final/checkpoint_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    checks = [
        sha256_file(Path(__file__).resolve().parents[3] / row["path"]) == row["sha256"]
        and row["replay_max_abs_difference"] <= 1e-7
        for row in metadata
    ]
    return {
        "status": "PASS" if all(checks) and len(checks) == 15 else "FAIL",
        "checkpoints": len(checks),
        "all_checksum_and_replay_checks_pass": all(checks),
    }


def replay_recommendation(plan_id: str) -> dict[str, Any]:
    trace = trace_recommendation(plan_id)
    replay = generate_plan(trace["risk_profile"])
    exact = replay == trace["plan"]
    return {"status": "PASS" if exact else "FAIL", "exact": exact, "replay": replay}


__all__ = [
    "batch_predict_risk",
    "export_expert_casebook_service",
    "generate_recommendation",
    "import_expert_scores_service",
    "predict_student_risk",
    "regenerate_recommendation",
    "replay_prediction",
    "replay_recommendation",
    "trace_recommendation",
    "validate_lineage",
]
