from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.paper_replication.pipeline import (
    DATASETS,
    PROJECT_ROOT,
    ensure_dirs,
    prepare_dataset,
    train_deep,
)


RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
TRIALS_PATH = RESULTS_DIR / "paper_replication_optuna_trials.csv"
BEST_PARAMS_PATH = RESULTS_DIR / "paper_replication_optuna_best_params.json"


def run_optuna(
    dataset: str,
    *,
    trials: int,
    epochs: int,
    seed: int = 42,
) -> dict:
    try:
        import optuna
    except Exception as exc:
        raise RuntimeError("Optuna is required for run_paper_optuna.py") from exc

    ensure_dirs()
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset}. Expected one of {sorted(DATASETS)}")
    prepare_dataset(dataset, seed=seed)
    trial_rows = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "conv_channels": trial.suggest_categorical("conv_channels", [16, 32, 64]),
            "bilstm_hidden": trial.suggest_categorical("bilstm_hidden", [16, 32, 64]),
            "bilstm_layers": trial.suggest_int("bilstm_layers", 1, 2),
            "dense_hidden": trial.suggest_categorical("dense_hidden", [64, 128, 256]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "patience": trial.suggest_categorical("patience", [3, 5, 10, 15]),
        }
        rows, info = train_deep(
            dataset,
            seed=seed,
            epochs=epochs,
            run_label=f"optuna_trial_{trial.number}",
            save_outputs=False,
            **params,
        )
        test_row = next(row for row in rows if row["split"] == "test")
        trial_rows.append(
            {
                "dataset": dataset,
                "trial": trial.number,
                "objective_val_macro_f1": float(info["best_score"]),
                "test_accuracy": test_row["accuracy"],
                "test_f1_macro": test_row["f1_macro"],
                "best_epoch": info["best_epoch"],
                "params_json": json.dumps(params, ensure_ascii=True, sort_keys=True),
            }
        )
        return float(info["best_score"])

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials)
    pd.DataFrame(trial_rows).to_csv(TRIALS_PATH, index=False)
    payload = {
        "dataset": dataset,
        "trials": trials,
        "epochs_per_trial": epochs,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "note": "Full Optuna sweep intentionally not run unless requested by user.",
    }
    BEST_PARAMS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna smoke/full search for paper-style CNN-BiLSTM.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_optuna(args.dataset, trials=args.trials, epochs=args.epochs, seed=args.seed)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

