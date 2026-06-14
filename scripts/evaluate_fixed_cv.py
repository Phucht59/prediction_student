"""Evaluate locked hyperparameters with stratified five-fold cross-validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_pipeline import load_or_create_splits
from src.config import DATASETS, METRICS_DIR, MODELS_DIR
from src.train_pipeline import objective


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_name in ("student-mat", "student-por", "xapi"):
        train_pool, _ = load_or_create_splits(dataset_name, "3class")
        params_path = MODELS_DIR / f"{dataset_name}_3class_best_params.json"
        params = json.loads(params_path.read_text(encoding="utf-8"))
        value = objective(
            optuna.trial.FixedTrial(params, number=0),
            train_pool,
            DATASETS[dataset_name],
            "3class",
            cv_folds=5,
        )
        result = {
            "dataset": dataset_name,
            "protocol": "Stratified 5-fold CV on train pool with locked hyperparameters",
            "f1_macro_mean": value,
            "hyperparameters_source": params_path.name,
        }
        out_path = METRICS_DIR / f"{dataset_name}_3class_fixed_cv.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{dataset_name}: {value:.6f}")


if __name__ == "__main__":
    main()
