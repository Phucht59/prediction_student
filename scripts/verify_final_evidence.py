"""Fail-fast verification for a final evidence bundle and its DB run."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    parser.add_argument("--evidence-dir", required=True)
    args = parser.parse_args()
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
    frame = pd.read_csv(directory / "locked_test_predictions.csv")
    probs = frame[["Prob_Class_0", "Prob_Class_1", "Prob_Class_2"]].to_numpy(float)
    if len(frame) != 79 or frame["__source_row_number"].nunique() != 79 or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise SystemExit("Invalid final prediction rows or probabilities.")
    metrics = classification_metrics(frame["True_Label"], frame["Pred_Label"], probs)
    if abs(metrics["macro_f1"] - manifest["metrics_summary"]["macro_f1"]) > 1e-12:
        raise SystemExit("Metric recomputation mismatch.")
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
    finally:
        c.close()
    print(json.dumps({"status": "passed", "run_id": manifest["final_run_id"], "macro_f1": metrics["macro_f1"]}))


if __name__ == "__main__":
    main()
