from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_B = {
    "resolved_config.yaml", "source_manifest.json", "fold_manifest.csv",
    "model_registry.json", "search_trials.csv", "selected_configs.csv",
    "selected_configs.json", "oof_predictions.csv", "metrics_summary.csv",
    "class_metrics.csv", "paired_deltas.csv", "transfer_predictions.csv",
    "transfer_metrics.csv", "overlap_audit.json", "runtime.csv",
    "seed_stability.csv", "seed_disagreement.csv", "artifact_checksums.json",
    "validation_report.json", "README.md",
}

REQUIRED_C = {
    "resolved_config.yaml", "source_manifest.json", "cohort_flow.csv",
    "cohort_by_forecast.csv", "class_distribution.csv", "feature_contract.json",
    "split_manifest.csv", "future_test_manifest.csv", "model_registry.json",
    "search_trials.csv", "selected_configs.csv", "selected_configs.json",
    "oof_predictions.parquet", "future_predictions.parquet",
    "metrics_by_model_forecast.csv", "class_metrics_by_model_forecast.csv",
    "paired_deltas.csv", "seed_stability.csv", "seed_disagreement.csv",
    "module_metrics.csv", "presentation_metrics.csv", "learning_curves.csv",
    "runtime_resources.csv", "parameter_counts.csv", "leakage_audit.json",
    "checkpoint_validation.json", "artifact_checksums.json",
    "validation_report.json", "README.md",
}

REQUIRED_FIGURES = {
    "target_distribution_by_forecast", "cohort_flow",
    "model_macro_f1_by_forecast", "at_risk_recall_by_forecast",
    "pr_auc_by_forecast", "deep_vs_ml_delta", "confusion_matrix_flagship",
    "learning_curves_flagship", "module_stability",
    "future_presentation_comparison",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_manifest_valid(root: Path) -> bool:
    manifest = json.loads((root / "artifact_checksums.json").read_text(encoding="utf-8"))
    return all(
        (root / entry["path"]).is_file()
        and (root / entry["path"]).stat().st_size == entry["bytes"]
        and sha256(root / entry["path"]) == entry["sha256"]
        for entry in manifest["entries"]
    )


def compact_mirror_valid(artifact: Path, report: Path) -> bool:
    compact = [path for path in artifact.rglob("*") if path.is_file() and path.suffix not in {".pt", ".pkl", ".parquet"}]
    return all((report / path.relative_to(artifact)).is_file() and sha256(path) == sha256(report / path.relative_to(artifact)) for path in compact)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-b-run", required=True)
    parser.add_argument("--study-c-run", required=True)
    parser.add_argument("--execution-run", required=True)
    args = parser.parse_args()
    b = ROOT / "artifacts" / "study_b_student_por" / args.study_b_run
    c = ROOT / "artifacts" / "study_c_oulad" / args.study_c_run
    rb = ROOT / "reports" / "study_b_student_por" / args.study_b_run
    rc = ROOT / "reports" / "study_c_oulad" / args.study_c_run
    execution = ROOT / "reports" / "extension_execution" / args.execution_run

    b_files = {path.name for path in b.iterdir() if path.is_file()}
    c_files = {path.name for path in c.iterdir() if path.is_file()}
    leakage = json.loads((c / "leakage_audit.json").read_text(encoding="utf-8"))
    split = pd.read_csv(c / "split_manifest.csv")
    pending_path = c / "job_ledger_pending.csv"
    try:
        pending_count = len(pd.read_csv(pending_path))
    except pd.errors.EmptyDataError:
        pending_count = 0
    test_report = json.loads((execution / "test_report.json").read_text(encoding="utf-8"))
    changed = subprocess.check_output(["git", "diff", "--name-only", "main...HEAD"], cwd=ROOT, text=True).splitlines()
    protected_prefixes = (
        "artifacts/strategy_b_", "reports/strategy_b_",
        "artifacts/final_repository_closure/", "reports/final_repository_closure/",
        "artifacts/strategy_b_phase_d_recommendation/", "reports/strategy_b_phase_d_recommendation/",
    )
    protected_changes = [path for path in changed if path.startswith(protected_prefixes)]
    figure_files = {path.stem for path in (c / "figures").glob("*.png")}
    checks = {
        "study_b_required_artifacts": REQUIRED_B <= b_files,
        "study_c_required_artifacts": REQUIRED_C <= c_files,
        "study_c_required_figures": REQUIRED_FIGURES <= figure_files,
        "study_c_learning_curve_directory": (c / "learning_curves").is_dir() and any((c / "learning_curves").glob("*.csv")),
        "study_c_confusion_directory": (c / "confusion_matrices").is_dir() and any((c / "confusion_matrices").glob("*.json")),
        "study_b_validation_report_present": (b / "validation_report.json").is_file(),
        "study_c_validation_report_present": (c / "validation_report.json").is_file(),
        "no_pending_jobs": pending_count == 0,
        "zero_global_student_overlap": leakage.get("future_student_overlap") == 0,
        "no_legacy_observed_access": leakage.get("legacy_observed_accessed") is False,
        "no_target_feature": leakage.get("forbidden_feature_scan") is True,
        "no_withdrawal_feature": leakage.get("forbidden_feature_scan") is True,
        "study_b_checksums": checksum_manifest_valid(b),
        "study_c_checksums": checksum_manifest_valid(c),
        "study_b_report_mirror": compact_mirror_valid(b, rb),
        "study_c_report_mirror": compact_mirror_valid(c, rc),
        "full_test_suite": test_report.get("return_code") == 0 and test_report.get("failed") == 0,
        "study_a_protected_evidence_unchanged": not protected_changes,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "protected_changes": protected_changes,
        "study_b_run": args.study_b_run,
        "study_c_run": args.study_c_run,
        "test_counts": {key: test_report.get(key) for key in ["collected", "passed", "skipped", "failed"]},
    }
    destination = execution / "validation_report.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (b / "validation_report.json").write_text(json.dumps({"status": result["status"], "scope": "Study B", "parent_validation": str(destination.relative_to(ROOT)), "checks": {key: value for key, value in checks.items() if key.startswith("study_b") or key in {"no_legacy_observed_access", "full_test_suite", "study_a_protected_evidence_unchanged"}}}, indent=2) + "\n", encoding="utf-8")
    (c / "validation_report.json").write_text(json.dumps({"status": result["status"], "scope": "Study C", "parent_validation": str(destination.relative_to(ROOT)), "checks": {key: value for key, value in checks.items() if key.startswith("study_c") or key in {"no_pending_jobs", "zero_global_student_overlap", "no_legacy_observed_access", "no_target_feature", "no_withdrawal_feature", "full_test_suite", "study_a_protected_evidence_unchanged"}}}, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(b / "validation_report.json", rb / "validation_report.json")
    shutil.copy2(c / "validation_report.json", rc / "validation_report.json")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
