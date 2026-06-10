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


def dataset_best_params_path(dataset: str) -> Path:
    return RESULTS_DIR / f"paper_replication_{dataset}_optuna_best_params.json"


def load_all_best_params() -> dict:
    if not BEST_PARAMS_PATH.exists():
        return {"datasets": {}}
    payload = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    if "datasets" in payload:
        return payload
    dataset = payload.get("dataset")
    if dataset:
        return {"datasets": {dataset: payload}}
    return {"datasets": {}}


def write_best_payload(dataset: str, payload: dict) -> None:
    dataset_best_params_path(dataset).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    aggregate = load_all_best_params()
    aggregate.setdefault("datasets", {})[dataset] = payload
    BEST_PARAMS_PATH.write_text(json.dumps(aggregate, indent=2, ensure_ascii=True), encoding="utf-8")


def append_trial_rows(rows: list[dict]) -> None:
    if not rows:
        return
    new_data = pd.DataFrame(rows)
    if TRIALS_PATH.exists():
        existing = pd.read_csv(TRIALS_PATH)
        data = pd.concat([existing, new_data], ignore_index=True)
    else:
        data = new_data
    data.to_csv(TRIALS_PATH, index=False)


def load_best_params(dataset: str) -> dict:
    dataset_path = dataset_best_params_path(dataset)
    if dataset_path.exists():
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    else:
        payload = load_all_best_params().get("datasets", {}).get(dataset)
    if not payload:
        raise FileNotFoundError(f"Missing Optuna best params for {dataset}. Run Optuna first.")
    return dict(payload["best_params"])


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
    append_trial_rows(trial_rows)
    payload = {
        "dataset": dataset,
        "trials": trials,
        "epochs_per_trial": epochs,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "objective": "val_macro_f1",
        "seed": seed,
        "note": "Optuna sweep completed from the local project run.",
    }
    write_best_payload(dataset, payload)
    return payload


def train_best_from_optuna(dataset: str, *, epochs: int, seed: int = 42, run_label: str = "optuna_best_full"):
    prepare_dataset(dataset, seed=seed)
    params = load_best_params(dataset)
    patience = int(params.pop("patience", 15))
    return train_deep(
        dataset,
        seed=seed,
        epochs=epochs,
        patience=patience,
        run_label=run_label,
        save_outputs=True,
        **params,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna smoke/full search for paper-style CNN-BiLSTM.")
    parser.add_argument("--dataset", choices=["all", *sorted(DATASETS)], required=True)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-best", action="store_true", help="Train and save CNN-BiLSTM using Optuna best params.")
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=None,
        help="Epochs for final Optuna-best training. Defaults to --epochs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    results = {}
    for dataset in datasets:
        payload = run_optuna(dataset, trials=args.trials, epochs=args.epochs, seed=args.seed)
        results[dataset] = payload
        if args.train_best:
            train_epochs = int(args.train_epochs or args.epochs)
            rows, info = train_best_from_optuna(
                dataset,
                epochs=train_epochs,
                seed=args.seed,
                run_label=f"optuna_best_full_{args.trials}_trials",
            )
            test_row = next(row for row in rows if row["split"] == "test")
            results[dataset]["final_train"] = {
                "epochs": train_epochs,
                "best_epoch": int(info["best_epoch"]),
                "best_val_f1_macro": float(info["best_score"]),
                "test_accuracy": float(test_row["accuracy"]),
                "test_f1_macro": float(test_row["f1_macro"]),
            }
    print(json.dumps({"datasets": results}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
