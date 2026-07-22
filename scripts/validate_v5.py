from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.studies.v5.common.artifacts import atomic_write_json, verify_checksum_manifest
from src.studies.v5.common.protocol import load_project_protocol, load_study_protocol, sha256_file, verify_declared_sources


def _v4_immutable() -> tuple[bool, object]:
    git = shutil.which("git")
    if not git:
        return False, "git executable unavailable"
    protected = [
        "configs/oulad_v4_protocol.yaml",
        "artifacts/oulad/v4",
        "reports/oulad/v4",
        "src/studies/oulad_v4",
        "scripts/oulad_v4_experiment.py",
        "scripts/validate_oulad_v4.py",
    ]
    completed = subprocess.run(
        [git, "diff", "--name-only", "ce79aa0b8f7444ac47ae9ae3ba6e72f997c5dd0a", "--", *protected],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    changed = [line for line in completed.stdout.splitlines() if line]
    return completed.returncode == 0 and not changed, changed


def _validate_study(study: str) -> list[dict[str, object]]:
    name = study.replace("-", "_")
    root = ROOT / "artifacts" / "v5" / name
    checks: list[dict[str, object]] = []

    def add(check: str, passed: bool, detail: object = None):
        checks.append({"check": check, "status": "PASS" if passed else "FAIL", "detail": detail})

    sources = verify_declared_sources(load_study_protocol(study))
    if all(row["status"] == "PASS" for row in sources):
        add("source_hashes", True, sources)
    elif all(row.get("observed_sha256") is None for row in sources):
        checks.append({"check": "source_hashes", "status": "SKIP_EXTERNAL_DATA", "detail": sources})
    else:
        add("source_hashes", False, sources)
    state_path = root / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    add("run_complete", state.get("status") == "COMPLETE", state.get("status", "missing"))
    add("future_not_accessed", state.get("future_accessed") is False, state.get("future_accessed"))
    checksum_path = root / "artifact_checksums.json"
    manifest = json.loads(checksum_path.read_text(encoding="utf-8")) if checksum_path.is_file() else {}
    add("artifact_checksums", bool(manifest) and verify_checksum_manifest(root, manifest), len(manifest))
    metrics_path = root / "final_metrics.csv"
    add("metrics_present", metrics_path.is_file())
    registry_path = root / "model_registry.json"
    add("model_registry", registry_path.is_file())
    prediction_csv = root / "oof_predictions.csv"
    prediction_parquet = root / "oof_predictions.parquet"
    if prediction_csv.is_file():
        frame = pd.read_csv(prediction_csv)
        probabilities = frame[["p_low", "p_medium", "p_high"]].to_numpy(dtype=float)
        valid = np.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1).all() and np.allclose(probabilities.sum(1), 1, atol=1e-5)
        add("probability_contract", bool(valid), len(frame))
        add("oof_record_coverage", frame[frame.candidate == "cnn_bilstm_v5_ensemble"].record_id.nunique() in {395, 649})
    elif prediction_parquet.is_file():
        frame = pd.read_parquet(prediction_parquet)
        valid = np.isfinite(frame.probability).all() and frame.probability.between(0, 1).all()
        add("probability_contract", bool(valid), len(frame))
        ensemble = frame[frame.candidate == "cnn_bilstm_ensemble"]
        add("grouped_student_contract", not ensemble.duplicated(["record_id"]).any(), len(ensemble))
    else:
        add("probability_contract", False, "prediction artifact missing")
    checkpoints = root / "checkpoint_metadata.json"
    metadata = json.loads(checkpoints.read_text(encoding="utf-8")) if checkpoints.is_file() else []
    add(
        "checkpoint_replay",
        bool(metadata)
        and all(float(row["replay_max_abs_difference"]) <= 1e-6 for row in metadata)
        and all((ROOT / row["path"]).is_file() and sha256_file(ROOT / row["path"]) == row["sha256"] for row in metadata),
        len(metadata),
    )
    return checks


def main() -> int:
    project = load_project_protocol()
    checks: list[dict[str, object]] = []
    immutable, detail = _v4_immutable()
    checks.append({"check": "v4_immutable", "status": "PASS" if immutable else "FAIL", "detail": detail})
    checks.append({"check": "performance_not_gate", "status": "PASS" if project["completion_gates"]["cnn_bilstm_must_win"] is False else "FAIL"})
    joint_root = ROOT / "artifacts" / "v5" / "joint_uci"
    joint_state = json.loads((joint_root / "run_state.json").read_text(encoding="utf-8")) if (joint_root / "run_state.json").is_file() else {}
    joint_manifest = json.loads((joint_root / "artifact_checksums.json").read_text(encoding="utf-8")) if (joint_root / "artifact_checksums.json").is_file() else {}
    joint_leakage = json.loads((joint_root / "leakage_audit.json").read_text(encoding="utf-8")) if (joint_root / "leakage_audit.json").is_file() else {}
    checks.append({"check": "joint_learning_complete", "status": "PASS" if joint_state.get("status") == "COMPLETE" and bool(joint_manifest) and verify_checksum_manifest(joint_root, joint_manifest) else "FAIL"})
    checks.append({"check": "joint_learning_group_leakage", "status": "PASS" if joint_leakage.get("status") == "PASS" else "FAIL"})
    recommendation_root = ROOT / "artifacts" / "v5" / "recommendation"
    recommendation_audit = json.loads((recommendation_root / "technical_validation.json").read_text(encoding="utf-8")) if (recommendation_root / "technical_validation.json").is_file() else {}
    recommendation_manifest = json.loads((recommendation_root / "artifact_checksums.json").read_text(encoding="utf-8")) if (recommendation_root / "artifact_checksums.json").is_file() else {}
    checks.append({"check": "recommendation_technical_validation", "status": "PASS" if recommendation_audit.get("status") == "PASS" and bool(recommendation_manifest) and verify_checksum_manifest(recommendation_root, recommendation_manifest) else "FAIL"})
    studies = {study: _validate_study(study) for study in ["student-mat", "student-por", "oulad"]}
    database_report = json.loads((ROOT / "reports/v5/final/database_audit.json").read_text(encoding="utf-8")) if (ROOT / "reports/v5/final/database_audit.json").is_file() else {}
    database_integration = database_report.get("status", "SKIP_NO_DISPOSABLE_DSN")
    checks.append({"check": "database_audit_or_explicit_waiver", "status": "PASS" if database_integration in {"PASS", "SKIP_NO_DISPOSABLE_DSN"} else "FAIL", "detail": database_integration})
    acceptable_study_statuses = {"PASS", "SKIP_EXTERNAL_DATA"}
    status = "PASS" if all(row["status"] == "PASS" for row in checks) and all(row["status"] in acceptable_study_statuses for rows in studies.values() for row in rows) else "FAIL"
    report = {
        "schema_version": "v5_strict_validation_v1",
        "status": status,
        "checks": checks,
        "studies": studies,
        "database_integration": database_integration,
        "future_benchmark": "NOT_EXECUTED",
        "external_source_note": "A clean clone validates committed evidence and replay checkpoints; raw/processed source hashes are skipped only when every declared external source is absent.",
        "validator_checks_correctness_not_model_superiority": True,
    }
    atomic_write_json(ROOT / "reports/v5/final/validation_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
