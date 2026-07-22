from __future__ import annotations

import json
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from .contract import ARTIFACT_ROOT, ROOT, atomic_json, protected_hash_status, sha256_file
from .recommendation import generate_plan, load_plans, validate_plan
from .risk_profile import validate_risk_profile


def validate_v6() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    reproduction = json.loads(
        (ARTIFACT_ROOT / "prediction/v5_1_reproduction.json").read_text(encoding="utf-8")
    )
    checks["v5_1_reproduction"] = reproduction["status"] == "PASS"
    checks["same_cohort"] = reproduction["metrics"]["records"] == 15378
    final = json.loads(
        (ARTIFACT_ROOT / "prediction/final/run_state.json").read_text(encoding="utf-8")
    )
    checks["final_complete"] = final["status"] == "COMPLETE"
    metadata = json.loads(
        (ARTIFACT_ROOT / "prediction/final/checkpoint_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    checks["seed_coverage"] = sorted({row["seed"] for row in metadata}) == [
        42,
        1201,
        2026,
        3407,
        7319,
    ]
    checks["fold_coverage"] = sorted({row["outer_fold"] for row in metadata}) == [0, 1, 2]
    checks["checkpoint_count"] = len(metadata) == 15
    checks["checkpoint_replay"] = all(
        row["replay_max_abs_difference"] <= 1e-7 for row in metadata
    )
    checks["checkpoint_checksums"] = all(
        sha256_file(ROOT / row["path"]) == row["sha256"] for row in metadata
    )
    checks["outer_test_not_used_for_selection"] = not final[
        "outer_test_used_for_selection"
    ]
    checks["future_locked"] = not final["future_accessed"]
    calibrated = pd.read_parquet(
        ARTIFACT_ROOT / "prediction/calibrated_oof_predictions.parquet"
    )
    probability = calibrated.probability_at_risk.to_numpy(dtype=float)
    checks["probability_validity"] = bool(
        np.isfinite(probability).all() and (probability >= 0).all() and (probability <= 1).all()
    )
    calibration = json.loads(
        (ARTIFACT_ROOT / "prediction/calibration.json").read_text(encoding="utf-8")
    )
    checks["calibration_train_only"] = not calibration["outer_test_used_to_fit"]
    checks["calibration_complete"] = calibration["status"] == "COMPLETE"
    domain = json.loads(
        (
            ARTIFACT_ROOT / "prediction/domain_generalization/run_state.json"
        ).read_text(encoding="utf-8")
    )
    checks["domain_generalization_complete"] = domain["status"] == "COMPLETE"
    profiles = pd.read_parquet(ARTIFACT_ROOT / "prediction/risk_profiles.parquet")
    profile_errors = 0
    for profile in profiles.to_dict(orient="records"):
        try:
            validate_risk_profile(profile)
        except (ValueError, TypeError):
            profile_errors += 1
    checks["risk_profile_schema"] = profile_errors == 0
    checks["risk_profile_coverage"] = len(profiles) == len(calibrated)
    checks["sensitive_fields_absent"] = not {
        "gender",
        "region",
        "disability",
        "age_band",
        "imd_band",
    }.intersection(profiles.columns)
    plans = load_plans()
    plan_errors = 0
    for plan in plans:
        try:
            validate_plan(plan)
        except (ValueError, TypeError):
            plan_errors += 1
    checks["recommendation_schema"] = plan_errors == 0
    checks["recommendation_coverage"] = len(plans) == len(profiles)
    checks["recommendation_duplicates"] = len({plan["plan_id"] for plan in plans}) == len(
        plans
    )
    profile_lookup = {
        profile["record_id"]: profile for profile in profiles.to_dict(orient="records")
    }
    checks["recommendation_lineage"] = all(
        plan["record_id"] in profile_lookup
        and plan["risk_profile_lineage_id"]
        == profile_lookup[plan["record_id"]]["lineage_id"]
        for plan in plans[:50]
    )
    checks["recommendation_replay"] = all(
        generate_plan(profile_lookup[plan["record_id"]]) == plan for plan in plans[:50]
    )
    technical = json.loads(
        (ARTIFACT_ROOT / "recommendation/technical_metrics.json").read_text(encoding="utf-8")
    )
    checks["recommendation_technical_pass"] = technical["status"] == "PASS"
    expert = json.loads(
        (ARTIFACT_ROOT / "recommendation/expert_evaluation/metrics.json").read_text(
            encoding="utf-8"
        )
    )
    checks["expert_status_valid"] = expert["status"] in {
        "PENDING_EXPERT_LABELS",
        "COMPLETE",
    }
    database = json.loads(
        (ARTIFACT_ROOT / "database/audit.json").read_text(encoding="utf-8")
    )
    checks["database_safe"] = database["status"] in {
        "PASS",
        "SKIP_NO_DISPOSABLE_DSN",
        "SKIP_DSN_NOT_PROVABLY_DISPOSABLE",
    } and not database["production_write"]
    protected = protected_hash_status()
    checks["protected_hashes"] = protected["pass"]
    core_pass = all(bool(value) for value in checks.values())
    git = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.splitlines()
    result = {
        "schema_version": "v6_validation_report_v1",
        "status": "PASS" if core_pass else "FAIL",
        "checks": checks,
        "check_count": len(checks),
        "passed": sum(bool(value) for value in checks.values()),
        "failed": [name for name, value in checks.items() if not value],
        "protected": protected,
        "expert_status": expert["status"],
        "database_status": database["status"],
        "future_oulad": "LOCKED_NOT_EXECUTED",
        "worktree_entries_at_validation": len(git),
    }
    atomic_json(ARTIFACT_ROOT / "validation_report.json", result)
    return result


__all__ = ["validate_v6"]
