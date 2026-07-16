"""Validate the frozen three-dataset thesis release without training models."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDY_A = ROOT / "artifacts" / "student_mat" / "final"
STUDY_B = ROOT / "artifacts" / "student_por" / "final"
STUDY_C = ROOT / "artifacts" / "oulad" / "final"
STUDY_C_REPORT = ROOT / "reports" / "oulad" / "final"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checksum_manifest(root: Path, manifest_path: Path) -> int:
    """Verify the three immutable checksum-manifest shapes used by the studies."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "entries" in manifest:
        rows = manifest["entries"]
    elif "files" in manifest:
        rows = manifest["files"]
    else:
        rows = [{"path": path, "sha256": digest} for path, digest in manifest.items()]
    for row in rows:
        path = root / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"Checksum validation failed: {path}")
    return len(rows)


def csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["candidate_id"]: row for row in csv.DictReader(handle)}


def require_close(actual: str, expected: float, label: str) -> None:
    if abs(float(actual) - expected) > 1e-12:
        raise RuntimeError(f"Frozen metric mismatch for {label}: {actual} != {expected}")


def validate_metrics() -> dict[str, int]:
    with (STUDY_A / "final_metrics.csv").open(newline="", encoding="utf-8") as handle:
        study_a = {row["model"]: row for row in csv.DictReader(handle)}
    require_close(study_a["R0"]["macro_f1"], 0.8988360425446519, "student-mat R0 Macro-F1")
    require_close(study_a["M1"]["macro_f1"], 0.8999548661053872, "student-mat Random Forest Macro-F1")
    require_close(study_a["N0"]["macro_f1"], 0.8503646133406478, "student-mat CNN-BiLSTM Macro-F1")

    study_b = csv_by_id(STUDY_B / "metrics_summary.csv")
    require_close(study_b["B-RF0"]["macro_f1"], 0.8698054388209258, "student-por Random Forest Macro-F1")
    require_close(study_b["B-H1"]["macro_f1"], 0.8469612236583831, "student-por CNN-BiLSTM Macro-F1")

    study_c = csv_by_id(STUDY_C / "ensemble_metrics.csv")
    require_close(study_c["V3-A0F-ENS"]["macro_f1"], 0.8286716600730241, "OULAD MLP Macro-F1")
    require_close(study_c["V3-D0-ENS"]["macro_f1"], 0.8311261008483025, "OULAD CNN-BiLSTM ensemble Macro-F1")
    return {"study_a_models": len(study_a), "study_b_models": len(study_b), "study_c_models": len(study_c)}


def validate_documents_and_source_scope() -> None:
    for name in ("README.md", "PROJECT.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for required in ("student-mat", "student-por", "OULAD"):
            if required not in text:
                raise RuntimeError(f"{name} does not describe {required}")
    obsolete = [
        "src/strategy_b_phase_ab.py",
        "src/strategy_b_phase_c.py",
        "src/strategy_b_phase_e_prediction.py",
        "scripts/run_strategy_b_phase_ab.py",
        "scripts/run_strategy_b_phase_c.py",
        "scripts/run_strategy_b_phase_d_recommendation.py",
        "scripts/run_strategy_b_phase_e_prediction.py",
        "scripts/run_final_repository_closure.py",
        "scripts/run_extension_end_to_end.py",
        "scripts/build_extension_evidence.py",
        "scripts/finalize_extension_execution.py",
        "scripts/run_study_b_seed_stability.py",
        "scripts/run_study_c_seed_stability.py",
        "MODEL_IMPROVEMENT_PLAN_V3.md",
        "SCIENTIFIC_PROTOCOL_V2.md",
    ]
    present = [path for path in obsolete if (ROOT / path).exists()]
    if present:
        raise RuntimeError(f"Obsolete final-source entrypoints remain: {present}")
    if not (ROOT / "src" / "models" / "student_grade.py").is_file():
        raise RuntimeError("The active UCI neural model module is missing")


def main() -> None:
    checksums = {
        "student_mat_closure_files": verify_checksum_manifest(STUDY_A, STUDY_A / "artifact_checksums.json"),
        "student_por_files": verify_checksum_manifest(STUDY_B, STUDY_B / "artifact_checksums.json"),
        "oulad_closure_files": verify_checksum_manifest(STUDY_C, STUDY_C / "artifact_checksums.json"),
    }
    metrics = validate_metrics()
    validate_documents_and_source_scope()
    command = [
        sys.executable,
            str(ROOT / "scripts" / "validate_oulad_final.py"),
        "--artifact-root",
        str(STUDY_C),
        "--report-root",
        str(STUDY_C_REPORT),
        "--check-only",
    ]
    result = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"OULAD strict validation failed:\n{result.stdout}\n{result.stderr}")
    payload = {
        "status": "PASS",
        "scope": ["student-mat", "student-por", "OULAD"],
        "training_executed": False,
        "checksums": checksums,
        "metrics": metrics,
        "oulad_strict_validation": "PASS",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
