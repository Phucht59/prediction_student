"""Assemble and verify a final, database-backed evidence bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluation import _connect
from src.evidence_metrics import bootstrap_confidence_intervals, classification_metrics, reliability_rows
from src.recommendation import structural_validity_metrics

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--database-run-id", required=True)
    p.add_argument("--selection-dir", required=True)
    p.add_argument(
        "--reference-evidence-dir",
        default=str(ROOT / "artifacts" / "final" / "final-a2945d79-9845-4979-b148-159f4853eca3"),
        help="Existing text evidence used for baseline/ablation tables; predictions are always read from PostgreSQL.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_id = args.database_run_id
    selection = Path(args.selection_dir)
    reference = Path(args.reference_evidence_dir)
    out = ROOT / "artifacts" / "final" / f"final-{run_id}"
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"Final evidence directory exists: {out}")
    out.mkdir(parents=True, exist_ok=True)

    connection = _connect()
    try:
        cursor = connection.cursor()
        cursor.execute("""
            select sr.source_row_number, p.true_label, p.predicted_label, p.confidence, p.probability
            from ml_predictions p
            join source_records sr on sr.record_id=p.record_id
            where p.run_id=%s and p.split_name='test'
            order by sr.source_row_number
        """, (run_id,))
        db_prediction_rows = cursor.fetchall()
    finally:
        connection.close()
    predictions = pd.DataFrame([
        {
            "__source_row_number": int(row[0]),
            "True_Label": int(row[1]),
            "Pred_Label": int(row[2]),
            "Confidence": float(row[3]),
            "Prob_Class_0": float(row[4]["Low"]),
            "Prob_Class_1": float(row[4]["Medium"]),
            "Prob_Class_2": float(row[4]["High"]),
        }
        for row in db_prediction_rows
    ])
    y_true = predictions["True_Label"].to_numpy(dtype=int)
    y_pred = predictions["Pred_Label"].to_numpy(dtype=int)
    probabilities = predictions[["Prob_Class_0", "Prob_Class_1", "Prob_Class_2"]].to_numpy(dtype=float)
    metrics = classification_metrics(y_true, y_pred, probabilities)
    ci = bootstrap_confidence_intervals(y_true, y_pred, n_resamples=2000, seed=42)
    if len(predictions) != 79 or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("Final prediction artifact has invalid row count or probabilities.")

    for source, target in [
        (selection / "selected_config.json", "selected_config.json"),
        (selection / "selection_manifest.json", "model_selection_manifest.json"),
        (selection / "protocol_manifest.json", "protocol_manifest.json"),
        (selection / "outer_oof_predictions.csv", "oof_predictions.csv"),
        (selection / "nested_optuna_trials.csv", "optuna_trials.csv"),
        (selection / "nested_model_selection_summary.json", "optuna_best_trials.json"),
    ]:
        shutil.copy2(source, out / target)
    predictions.to_csv(out / "locked_test_predictions.csv", index=False)

    for name in ["baseline_results.csv", "deep_ablation_results.csv", "ablation_results.csv", "fairness_slices.csv"]:
        shutil.copy2(reference / name, out / name)
    shutil.copy2(reference / "baseline_results.csv", out / "scenario_results.csv")
    shutil.copy2(reference / "deep_ablation_results.csv", out / "imbalance_ablation_results.csv")

    cm = pd.DataFrame(metrics["confusion_matrix"], index=["true_0", "true_1", "true_2"], columns=["pred_0", "pred_1", "pred_2"])
    cm.to_csv(out / "confusion_matrix.csv")
    dump(out / "classification_report.json", metrics)
    dump(out / "ordinal_metrics.json", {k: metrics[k] for k in ["quadratic_weighted_kappa", "ordinal_mae", "one_step_errors", "two_step_errors"]})
    dump(out / "bootstrap_confidence_intervals.json", ci)
    dump(out / "calibration_metrics.json", {"method": "none_frozen_before_locked_test", "brier": metrics["multiclass_brier_score"], "ece": metrics["ece"]})
    pd.DataFrame(reliability_rows(y_true, probabilities)).to_csv(out / "reliability_curve_data.csv", index=False)

    pr = []
    for label in range(3):
        prec, rec, th = precision_recall_curve((y_true == label).astype(int), probabilities[:, label])
        pr.extend({"class_label": label, "precision": float(p), "recall": float(r), "threshold": None if i == len(th) else float(th[i])} for i, (p, r) in enumerate(zip(prec, rec)))
    pd.DataFrame(pr).to_csv(out / "pr_curve_data.csv", index=False)

    c = _connect()
    try:
        cur = c.cursor()
        cur.execute("select run_id, dataset_version_id, status, train_config, target_definition, artifact_uri from ml_experiment_runs where run_id=%s", (run_id,))
        row = cur.fetchone()
        if row is None or row[2] != "completed":
            raise RuntimeError("Database run is missing or not completed.")
        cur.execute("select split_name, count(*) from ml_run_record_splits where run_id=%s group by split_name order by split_name", (run_id,))
        splits = dict(cur.fetchall())
        cur.execute("select count(*), count(distinct record_id) from ml_predictions where run_id=%s and split_name='test'", (run_id,))
        pred_counts = cur.fetchone()
        cur.execute("select count(*) from ml_recommendations r join ml_predictions p on p.prediction_id=r.prediction_id where p.run_id=%s", (run_id,))
        rec_count = cur.fetchone()[0]
        cur.execute("""
            select sr.source_row_number, p.predicted_label, p.confidence, r.learning_path
            from ml_recommendations r
            join ml_predictions p on p.prediction_id=r.prediction_id
            join source_records sr on sr.record_id=p.record_id
            where p.run_id=%s order by sr.source_row_number
        """, (run_id,))
        recommendation_rows = cur.fetchall()
        cur.execute("select metric_name, metric_value from ml_run_metrics where run_id=%s and split_name='test' order by metric_name", (run_id,))
        db_metrics = dict(cur.fetchall())
    finally:
        c.close()
    if splits.get("test") != len(predictions) or pred_counts != (len(predictions), len(predictions)) or rec_count != len(predictions):
        raise RuntimeError("Database split/prediction/recommendation counts do not match final CSV.")
    db_manifest = {"database_run_id": run_id, "dataset_version_id": row[1], "status": row[2], "split_counts": splits, "test_prediction_counts": pred_counts, "recommendation_count": rec_count, "database_metrics": db_metrics, "integrity": "passed"}
    dump(out / "database_run_manifest.json", db_manifest)
    recommendation_payloads = [dict(row[3]) for row in recommendation_rows]
    recommendation_metrics = structural_validity_metrics(recommendation_payloads)
    recommendation_metrics.update({"source_model": "cnn_bilstm_classifier_single_seed_42", "expert_evaluation": "not_collected", "causal_effectiveness_claimed": False})
    dump(out / "recommendation_evaluation.json", recommendation_metrics)
    pd.DataFrame([
        {"source_row_number": int(row[0]), "predicted_label": int(row[1]), "confidence": float(row[2]), "recommendation": json.dumps(row[3], ensure_ascii=False), "expert_rating": None}
        for row in recommendation_rows[:12]
    ]).to_csv(out / "recommendation_expert_review_cases.csv", index=False)

    split = json.loads((selection / "protocol_manifest.json").read_text(encoding="utf-8"))["split_hashes"]
    selection_protocol = json.loads((selection / "protocol_manifest.json").read_text(encoding="utf-8"))
    dataset_manifest = {"dataset": "student-mat", "dataset_checksum": selection_protocol["dataset_checksum"], "dataset_version_id": row[1], "row_count": 395, "data_source": "postgresql"}
    dump(out / "dataset_manifest.json", dataset_manifest)
    dump(out / "split_hashes.json", split)
    dump(out / "split_manifest.json", {"protocol": "stratified_80_20_locked_test", "seed": 42, **split})
    dump(out / "environment.json", {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__})
    outer = json.loads((selection / "nested_model_selection_summary.json").read_text(encoding="utf-8"))["outer_summary"]
    pd.DataFrame([outer]).to_csv(out / "outer_fold_metrics.csv", index=False)
    dump(out / "outer_fold_metrics.json", outer)

    dump(out / "model_checksums.json", {"checkpoints": {}, "parameter_count": json.loads((selection / "selected_config.json").read_text())["model_parameter_count"]})
    manifest = {"final_run_id": run_id, "selection_run_id": json.loads((selection / "selected_config.json").read_text())["selection_run_id"], "timestamp": datetime.now(timezone.utc).isoformat(), "dataset_checksum": dataset_manifest["dataset_checksum"], "dataset_version_id": row[1], "selected_config_checksum": sha(out / "selected_config.json"), "split_hashes": split, "database_run_id": run_id, "metrics_summary": metrics, "database_integrity": "passed"}
    dump(out / "run_manifest.json", manifest)
    (out / "README.md").write_text(f"# Final DB-first evidence run\n\n- Final run: `{run_id}` (completed)\n- Selection: nested-full-20260710, 5 outer × 3 inner folds, 30 trials, fixed seed 42.\n- Locked-test Macro-F1: {metrics['macro_f1']:.4f}.\n- CNN–BiLSTM is not claimed to beat the G2 baseline; see `baseline_results.csv`.\n- No expert review score is fabricated.\n", encoding="utf-8")
    checksum_exclusions = {"artifact_checksums.json", "run_manifest.json"}
    checksums = {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name not in checksum_exclusions}
    dump(out / "artifact_checksums.json", {"algorithm": "sha256", "excluded": sorted(checksum_exclusions), "files": checksums})
    (ROOT / "artifacts" / "final" / "LATEST_RUN.txt").write_text(out.name + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "metrics": metrics, "database": db_manifest}, default=str))

if __name__ == "__main__":
    main()
