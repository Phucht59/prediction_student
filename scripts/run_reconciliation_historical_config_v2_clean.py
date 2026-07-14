"""Diagnostic only: historical per-outer config on V2-clean training control.

It consumes the immutable historical artifact and the V2 shared manifest.  It
does not tune, touch legacy-79, or alter any benchmark artifact.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import ROOT_DIR
from src.data_pipeline import SOURCE_ROW_NUMBER_COLUMN, process_target_and_stratify
from src.evaluation.protocol import (
    assert_no_legacy_records,
    load_fold_manifest,
    outer_folds_from_manifest,
    source_record_identity,
    validate_probability_matrix,
)
from src.model_selection import fit_fold_predict_proba
from src.postgres_data_source import load_dataset_version_from_postgres


HISTORICAL_ROOT = ROOT_DIR / "artifacts" / "model_selection" / "nested-full-20260710"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="reconciliation-historical-config-v2-clean-seed42-20260714")
    args = parser.parse_args()
    root = ROOT_DIR / "artifacts" / "reconciliation" / args.run_id
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    raw, dataset_meta = load_dataset_version_from_postgres("student-mat", 1)
    frame = process_target_and_stratify(raw.copy(), "G3", "student", "3class").drop(columns=["_strat_target"])
    manifest = load_fold_manifest()
    allowed_rows = {int(row["source_record_identity"].rsplit(":", 1)[1]) for row in manifest["development_records"]}
    frame = frame[frame[SOURCE_ROW_NUMBER_COLUMN].astype(int).isin(allowed_rows)].sort_values(SOURCE_ROW_NUMBER_COLUMN).reset_index(drop=True)
    assert_no_legacy_records([source_record_identity(1, row) for row in frame[SOURCE_ROW_NUMBER_COLUMN]])
    folds = outer_folds_from_manifest(frame, manifest)
    historical = json.loads((HISTORICAL_ROOT / "nested_model_selection_summary.json").read_text(encoding="utf-8"))
    params_by_fold = {int(item["outer_fold"]): item["inner_best"]["best_params"] for item in historical["outer_results"]}
    spec = type("Spec", (), {"target_col": "G3", "kind": "student"})()
    rows: list[dict] = []
    fold_rows: list[dict] = []
    for fold_index, (train_index, validation_index) in enumerate(folds):
        result = fit_fold_predict_proba(
            train_fold=frame.iloc[train_index].copy(),
            validation_fold=frame.iloc[validation_index].copy(),
            spec=spec,
            params=params_by_fold[fold_index],
            seed=42,
            fold_index=fold_index,
        )
        validate_probability_matrix(result.probabilities, result.predictions)
        score = float(f1_score(result.true_labels, result.predictions, average="macro", zero_division=0))
        fold_rows.append({"outer_fold": fold_index, "macro_f1": score, "config": params_by_fold[fold_index]})
        for local_index, (_, record) in enumerate(frame.iloc[validation_index].iterrows()):
            rows.append({
                "run_id": args.run_id,
                "record_id": source_record_identity(1, record[SOURCE_ROW_NUMBER_COLUMN]),
                "source_row_number": int(record[SOURCE_ROW_NUMBER_COLUMN]),
                "outer_fold": fold_index,
                "training_seed": 42,
                "true_label": int(result.true_labels[local_index]),
                "predicted_label": int(result.predictions[local_index]),
                "probability_low": float(result.probabilities[local_index, 0]),
                "probability_medium": float(result.probabilities[local_index, 1]),
                "probability_high": float(result.probabilities[local_index, 2]),
            })
    predictions = pd.DataFrame(rows)
    predictions.to_csv(root / "predictions.csv", index=False)
    pd.DataFrame(fold_rows).drop(columns=["config"]).to_csv(root / "fold_metrics.csv", index=False)
    summary = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Historical per-outer configuration under V2 clean refit protocol; diagnostic only.",
        "dataset_checksum": dataset_meta["content_hash"],
        "fold_manifest_checksum": manifest["manifest_checksum"],
        "historical_config_source": str(HISTORICAL_ROOT / "nested_model_selection_summary.json"),
        "training_protocol": "V2 internal epoch selection then full outer-train fixed-epoch refit",
        "seed": 42,
        "fold_macro_f1": [row["macro_f1"] for row in fold_rows],
        "mean_macro_f1": float(np.mean([row["macro_f1"] for row in fold_rows])),
        "std_macro_f1": float(np.std([row["macro_f1"] for row in fold_rows], ddof=1)),
        "n_predictions": int(len(predictions)),
    }
    (root / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(root)


if __name__ == "__main__":
    main()
