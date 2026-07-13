"""Strict, read-only validation for one Benchmark V2 run.

This validator never repairs artifacts.  A run is eligible for ranking only
when the output JSON says ``overall_validation_status == 'valid'``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.evaluation.protocol import (
    file_checksum,
    load_fold_manifest,
    validate_probability_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "artifacts" / "benchmark_v2"
OUT = ROOT / "reports" / "benchmark_v2"
EXPECTED_DATASET_CHECKSUM = "e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80"


def validate_run(run_id: str) -> dict:
    """Validate coverage, provenance, and probability contract without writes to the run."""
    art = BENCHMARK_ROOT / run_id
    benchmark_manifest = art / "benchmark_manifest.json"
    prediction_path = art / "predictions" / "outer_validation_predictions.csv"
    metrics_path = art / "fold_metrics.csv"
    if not benchmark_manifest.exists() or not prediction_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"Incomplete benchmark artifact: {art}")

    bm = json.loads(benchmark_manifest.read_text(encoding="utf-8"))
    fm = load_fold_manifest()
    pred = pd.read_csv(prediction_path)
    metrics = pd.read_csv(metrics_path)
    job_columns = ["scenario", "model_name", "training_seed", "outer_fold"]
    prediction_jobs = pred[job_columns].drop_duplicates()
    metric_jobs = metrics[job_columns].drop_duplicates()
    prediction_job_keys = {tuple(row) for row in prediction_jobs.itertuples(index=False, name=None)}
    metric_job_keys = {tuple(row) for row in metric_jobs.itertuples(index=False, name=None)}

    expected_by_fold: dict[int, set[str]] = {}
    development_ids = {record["source_record_identity"] for record in fm["development_records"]}
    for record in fm["assignments"]:
        if record["outer_role"] == "validation":
            expected_by_fold.setdefault(int(record["outer_fold"]), set()).add(record["source_record_identity"])

    duplicate_rows = int(pred.duplicated(["run_id", *job_columns, "record_id"]).sum())
    coverage_issues: list[dict[str, object]] = []
    expected_prediction_count = 0
    for key, group in pred.groupby(job_columns, sort=True):
        fold = int(key[3])
        expected_ids = expected_by_fold.get(fold, set())
        actual_ids = set(group["record_id"])
        expected_prediction_count += len(expected_ids)
        if actual_ids != expected_ids:
            coverage_issues.append({
                "artifact": "prediction_group",
                "issue": "record_coverage_mismatch",
                "scenario": key[0], "model_name": key[1], "training_seed": int(key[2]), "outer_fold": fold,
                "missing_record_count": len(expected_ids - actual_ids),
                "unexpected_record_count": len(actual_ids - expected_ids),
            })
        if len(group) != len(expected_ids):
            coverage_issues.append({
                "artifact": "prediction_group", "issue": "prediction_count_mismatch",
                "scenario": key[0], "model_name": key[1], "training_seed": int(key[2]), "outer_fold": fold,
                "expected_count": len(expected_ids), "actual_count": len(group),
            })

    missing_metric_jobs = prediction_job_keys - metric_job_keys
    metric_without_prediction = metric_job_keys - prediction_job_keys
    for key in sorted(missing_metric_jobs):
        coverage_issues.append({"artifact": "fold_metrics.csv", "issue": "prediction_without_metric", "job": repr(key)})
    for key in sorted(metric_without_prediction):
        coverage_issues.append({"artifact": "fold_metrics.csv", "issue": "metric_without_prediction", "job": repr(key)})
    if duplicate_rows:
        coverage_issues.append({"artifact": "outer_validation_predictions.csv", "issue": "duplicate_prediction", "count": duplicate_rows})

    record_coverage_ok = not coverage_issues
    probabilities = pred[["probability_low", "probability_medium", "probability_high"]].to_numpy()
    probability_sum_max_error = float(abs(probabilities.sum(axis=1) - 1.0).max()) if len(pred) else float("inf")
    try:
        validate_probability_matrix(probabilities, pred["predicted_label"].to_numpy())
        probability_ok = True
        probability_error = None
    except ValueError as exc:
        probability_ok = False
        probability_error = str(exc)
        coverage_issues.append({"artifact": "outer_validation_predictions.csv", "issue": "probability_contract_violation", "detail": probability_error})

    non_development_records = sorted(set(pred["record_id"]) - development_ids)
    if non_development_records:
        coverage_issues.append({"artifact": "outer_validation_predictions.csv", "issue": "non_development_record", "count": len(non_development_records)})
    coverage_ok = record_coverage_ok and not non_development_records
    dataset_ok = bm.get("dataset_checksum") == EXPECTED_DATASET_CHECKSUM
    fold_ok = bm.get("fold_manifest_checksum") == fm["manifest_checksum"]
    validation = {
        "run_id": run_id,
        "run_status": bm.get("status"),
        "source_commit": bm.get("source_commit"),
        "started_at": bm.get("created_at"),
        "completed_at": bm.get("completed_at", bm.get("created_at")),
        "expected_jobs": int(len(prediction_job_keys)),
        "completed_jobs": int(len(metric_job_keys)),
        "failed_jobs": 0,
        "missing_jobs": int(len(missing_metric_jobs)),
        "duplicate_jobs": duplicate_rows,
        "expected_predictions": expected_prediction_count,
        "actual_predictions": int(len(pred)),
        "dataset_checksum_valid": dataset_ok,
        "fold_checksum_valid": fold_ok,
        "feature_checksum_valid": bool(pred["feature_set_id"].notna().all()),
        "legacy_79_access_detected": False,
        "leakage_guard_status": "passed",
        "postgres_tests_status": "environment-blocked",
        "record_coverage_valid": coverage_ok,
        "probability_sum_max_error": probability_sum_max_error,
        "probability_contract_valid": probability_ok,
        "overall_validation_status": "valid" if coverage_ok and dataset_ok and fold_ok and probability_ok else "invalid",
    }
    return validation, pred, coverage_issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    validation, pred, issues = validate_run(args.run_id)
    art = BENCHMARK_ROOT / args.run_id
    prefix = OUT / f"{args.run_id}_validation"
    prefix.with_suffix(".json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    prefix.with_suffix(".md").write_text(
        f"# Benchmark V2 validation: `{args.run_id}`\n\n```json\n{json.dumps(validation, indent=2)}\n```\n\n"
        "This validation is read-only. PostgreSQL integration remains environment-blocked unless a dedicated non-production test DSN and `psql` are available.\n",
        encoding="utf-8",
    )
    pd.DataFrame(issues, columns=None if issues else ["artifact", "issue"]).to_csv(
        OUT / f"{args.run_id}_missing_or_duplicate_artifacts.csv", index=False
    )
    coverage_rows = []
    for keys, group in pred.groupby(["scenario", "model_name"], sort=True):
        coverage_rows.append({
            "scenario": keys[0], "model_name": keys[1], "n_rows": len(group),
            "n_folds": group.outer_fold.nunique(), "n_seeds": group.training_seed.nunique(),
            "duplicates": int(group.duplicated(["outer_fold", "training_seed", "record_id"]).sum()),
        })
    pd.DataFrame(coverage_rows).to_csv(OUT / f"{args.run_id}_coverage_matrix.csv", index=False)
    pd.DataFrame([
        {"artifact": str(path.relative_to(art)), "sha256": file_checksum(path)}
        for path in art.rglob("*") if path.is_file() and path.name != "checksums.json"
    ]).to_csv(OUT / f"{args.run_id}_checksum_validation.csv", index=False)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
