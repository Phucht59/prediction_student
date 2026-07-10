"""Fail-fast verification for a final evidence bundle and its DB run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.evaluation import _connect
from src.evidence_metrics import classification_metrics


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    latest = Path(__file__).resolve().parents[1] / "artifacts" / "final" / "LATEST_RUN.txt"
    default_dir = None
    if latest.exists():
        default_dir = latest.parent / latest.read_text(encoding="utf-8").strip()
    parser.add_argument("--evidence-dir", type=Path, default=default_dir)
    parser.add_argument("--skip-db", action="store_true", help="Verify portable artifacts without a live PostgreSQL connection.")
    args = parser.parse_args()
    if args.evidence_dir is None:
        raise SystemExit("No evidence directory supplied and LATEST_RUN.txt is missing.")
    directory = Path(args.evidence_dir)
    if (directory / "SMOKE_RUN.md").exists():
        raise SystemExit("Refusing a smoke evidence directory.")
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    checksums = json.loads((directory / "artifact_checksums.json").read_text(encoding="utf-8"))
    for name, expected in checksums["files"].items():
        path = directory / name
        if not path.exists() or sha(path) != expected:
            raise SystemExit(f"Checksum mismatch: {name}")
    if sha(directory / "selected_config.json") != manifest["selected_config_checksum"]:
        raise SystemExit("Selected config checksum mismatch.")
    dataset_manifest = json.loads((directory / "dataset_manifest.json").read_text(encoding="utf-8"))
    if dataset_manifest.get("data_source") != "postgresql":
        raise SystemExit("Final evidence is not marked as PostgreSQL-backed.")
    with (directory / "locked_test_predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    source_rows = [int(row["__source_row_number"]) for row in rows]
    y_true = np.asarray([int(row["True_Label"]) for row in rows], dtype=int)
    y_pred = np.asarray([int(row["Pred_Label"]) for row in rows], dtype=int)
    probs = np.asarray([[float(row[f"Prob_Class_{label}"]) for label in range(3)] for row in rows], dtype=float)
    if len(rows) != 79 or len(set(source_rows)) != 79 or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise SystemExit("Invalid final prediction rows or probabilities.")
    metrics = classification_metrics(y_true, y_pred, probs)
    if abs(metrics["macro_f1"] - manifest["metrics_summary"]["macro_f1"]) > 1e-12:
        raise SystemExit("Metric recomputation mismatch.")
    if args.skip_db:
        print(json.dumps({"status": "passed", "run_id": manifest["final_run_id"], "macro_f1": metrics["macro_f1"], "database": "skipped"}))
        return
    c = _connect()
    try:
        cur = c.cursor()
        run_id = manifest["database_run_id"]
        cur.execute("select status from ml_experiment_runs where run_id=%s", (run_id,))
        row = cur.fetchone()
        if not row or row[0] != "completed":
            raise SystemExit("Database run is not completed.")
        cur.execute("select count(*), count(distinct record_id) from ml_predictions where run_id=%s and split_name='test'", (run_id,))
        if tuple(cur.fetchone()) != (79, 79):
            raise SystemExit("Database prediction count mismatch.")
        cur.execute("select count(*) from ml_recommendations r join ml_predictions p on p.prediction_id=r.prediction_id where p.run_id=%s", (run_id,))
        if int(cur.fetchone()[0]) != 79:
            raise SystemExit("Database recommendation count mismatch.")
    finally:
        c.close()
    print(json.dumps({"status": "passed", "run_id": manifest["final_run_id"], "macro_f1": metrics["macro_f1"]}))


if __name__ == "__main__":
    main()
