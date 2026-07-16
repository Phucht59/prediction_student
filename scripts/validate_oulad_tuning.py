from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.evidence_paths import resolve_evidence_path
REQUIRED = {
    "resolved_protocol.yaml", "source_provenance.json", "v1_comparators.json", "candidate_registry.json",
    "outer_fold_manifest.csv", "inner_fold_manifest.csv", "optuna_trials.csv", "selected_configs.json",
    "adaptive_decision_log.jsonl", "oof_predictions.parquet", "metrics_summary.csv", "metrics_by_seed.csv",
    "class_metrics.csv", "module_metrics.csv", "paired_deltas.csv", "parameter_counts.csv", "runtime_resources.csv",
    "learning_curves.csv", "grouped_bootstrap.csv", "checkpoint_validation.json", "probability_validation.json", "gate_assessment.json",
    "future_policy_audit.json", "test_report.json", "validation_report.json", "README.md", "artifact_checksums.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="oulad-deep-v2-f2-20260716-v1")
    args = parser.parse_args()
    root = ROOT / "artifacts" / "oulad" / "tuning"
    failures: list[str] = []
    missing = sorted(REQUIRED - {path.name for path in root.iterdir() if path.is_file()}) if root.exists() else sorted(REQUIRED)
    if missing:
        failures.append(f"missing_artifacts={missing}")
    if not failures:
        protocol = json.loads((root / "resolved_protocol.yaml").read_text(encoding="utf-8"))
        v1 = protocol["v1_immutable"]
        if sha256(resolve_evidence_path(ROOT, v1["artifact_path"]) / "metrics_by_model_forecast.csv") != v1["metrics_sha256"]:
            failures.append("v1_metrics_mutated")
        if sha256(resolve_evidence_path(ROOT, v1["artifact_path"]) / "oof_predictions.parquet") != v1["oof_predictions_sha256"]:
            failures.append("v1_oof_mutated")
        outer = pd.read_csv(root / "outer_fold_manifest.csv")
        for fold in range(3):
            train = outer.loc[outer.outer_fold != fold, "id_student"]
            validation = outer.loc[outer.outer_fold == fold, "id_student"]
            if set(train) & set(validation): failures.append(f"outer_student_overlap_{fold}")
        inner = pd.read_csv(root / "inner_fold_manifest.csv")
        for keys, frame in inner.groupby(["outer_fold", "inner_fold"]):
            train = frame.loc[frame.role == "inner_train", "id_student"]
            validation = frame.loc[frame.role == "inner_validation", "id_student"]
            if set(train) & set(validation): failures.append(f"inner_student_overlap_{keys}")
        predictions = pd.read_parquet(root / "oof_predictions.parquet")
        trained = predictions.loc[predictions.candidate_id.isin(["V2-H2T", "V2-A0", "V2-T0", "V2-H3C"])]
        if len(trained) != 4 * 3 * len(outer): failures.append("trained_oof_row_count")
        if not np.isfinite(trained.probability).all() or not trained.probability.between(0, 1).all(): failures.append("probability_contract")
        reported = pd.read_csv(root / "metrics_by_seed.csv").set_index(["candidate_id", "seed"])
        for keys, frame in predictions.groupby(["candidate_id", "seed"]):
            value = f1_score(frame.target_at_risk, frame.predicted_label, average="macro", zero_division=0)
            if abs(value - reported.loc[keys, "macro_f1"]) > 1e-12: failures.append(f"metric_mismatch_{keys}")
        checkpoint = json.loads((root / "checkpoint_validation.json").read_text(encoding="utf-8"))
        probability = json.loads((root / "probability_validation.json").read_text(encoding="utf-8"))
        future = json.loads((root / "future_policy_audit.json").read_text(encoding="utf-8"))
        tests = json.loads((root / "test_report.json").read_text(encoding="utf-8"))
        if checkpoint["status"] != "PASS": failures.append("checkpoint_validation")
        if probability["status"] != "PASS": failures.append("probability_validation")
        if future["future_benchmark_accessed_during_selection"]: failures.append("future_access")
        if tests["status"] != "PASS" or tests["failed"] != 0: failures.append("test_suite")
        checksums = json.loads((root / "artifact_checksums.json").read_text(encoding="utf-8"))
        for relative, expected in checksums.items():
            if not (root / relative).exists() or sha256(root / relative) != expected: failures.append(f"checksum_{relative}")
    payload = {"status": "PASS" if not failures else "FAIL", "failures": failures, "run_id": args.run_id}
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
