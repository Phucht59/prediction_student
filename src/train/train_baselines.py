from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    classification_metrics,
    regression_metrics,
    save_confusion_matrix_plot,
)
from src.models.baselines import (  # noqa: E402
    get_classification_baselines,
    get_regression_baselines,
    optional_model_skip_reasons,
)


DATASETS = ("student-mat", "student-por", "student-combined")
SCENARIOS = ("mid", "late")
TASKS = ("classification", "regression")
REQUIRED_FILES = (
    "X_train.npy",
    "X_val.npy",
    "X_test.npy",
    "y_train_class.npy",
    "y_val_class.npy",
    "y_test_class.npy",
    "y_train_reg.npy",
    "y_val_reg.npy",
    "y_test_reg.npy",
    "metadata.json",
    "feature_names.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train traditional baseline models.")
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--task", choices=["all", *TASKS], default="all")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def selected(selection: str, values: tuple[str, ...]) -> list[str]:
    return list(values) if selection == "all" else [selection]


def load_processed_split(dataset: str, scenario: str) -> dict:
    split_dir = PROJECT_ROOT / "data" / "processed" / dataset / scenario
    if not split_dir.exists():
        raise FileNotFoundError(f"Processed directory does not exist: {split_dir}")

    missing = [filename for filename in REQUIRED_FILES if not (split_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed files in {split_dir}: {missing}")

    data = {
        "split_dir": split_dir,
        "X_train": np.load(split_dir / "X_train.npy", allow_pickle=False),
        "X_val": np.load(split_dir / "X_val.npy", allow_pickle=False),
        "X_test": np.load(split_dir / "X_test.npy", allow_pickle=False),
        "y_train_class": np.load(split_dir / "y_train_class.npy", allow_pickle=False),
        "y_val_class": np.load(split_dir / "y_val_class.npy", allow_pickle=False),
        "y_test_class": np.load(split_dir / "y_test_class.npy", allow_pickle=False),
        "y_train_reg": np.load(split_dir / "y_train_reg.npy", allow_pickle=False),
        "y_val_reg": np.load(split_dir / "y_val_reg.npy", allow_pickle=False),
        "y_test_reg": np.load(split_dir / "y_test_reg.npy", allow_pickle=False),
        "metadata": json.loads((split_dir / "metadata.json").read_text(encoding="utf-8")),
        "feature_names": json.loads((split_dir / "feature_names.json").read_text(encoding="utf-8")),
    }
    validate_processed_data(dataset, scenario, data)
    return data


def validate_processed_data(dataset: str, scenario: str, data: dict) -> None:
    for split in ("train", "val", "test"):
        X = data[f"X_{split}"]
        y_class = data[f"y_{split}_class"]
        y_reg = data[f"y_{split}_reg"]
        if X.ndim != 2:
            raise ValueError(f"{dataset}/{scenario}/{split}: X must be 2D, got shape {X.shape}")
        if X.shape[0] != y_class.shape[0] or X.shape[0] != y_reg.shape[0]:
            raise ValueError(
                f"{dataset}/{scenario}/{split}: shape mismatch "
                f"X={X.shape}, y_class={y_class.shape}, y_reg={y_reg.shape}"
            )
        if np.isnan(X).any():
            raise ValueError(f"{dataset}/{scenario}/{split}: X contains NaN values")

    leakage = data["metadata"].get("leakage_checks", {})
    if leakage.get("passed") is not True:
        raise ValueError(f"{dataset}/{scenario}: leakage check did not pass: {leakage}")


def save_predictions(output_path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(output_path, index=False)


def upsert_results(path: Path, rows: list[dict], key_columns: list[str]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if "scenario" in new_df.columns:
        new_df = new_df[new_df["scenario"].ne("early")].copy()

    if path.exists():
        existing = pd.read_csv(path)
        if "scenario" in existing.columns:
            existing = existing[existing["scenario"].ne("early")].copy()
        if not existing.empty:
            new_keys = set(map(tuple, new_df[key_columns].astype(str).to_numpy()))
            keep_mask = [
                tuple(row) not in new_keys
                for row in existing[key_columns].astype(str).to_numpy()
            ]
            existing = existing.loc[keep_mask]
        output = pd.concat([existing, new_df], ignore_index=True)
    else:
        output = new_df

    output.to_csv(path, index=False)


def train_classification_models(
    dataset: str,
    scenario: str,
    data: dict,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    skips: list[dict] = []
    models = get_classification_baselines(seed=seed)
    X_train = data["X_train"]
    split_data = {
        "train": (data["X_train"], data["y_train_class"]),
        "val": (data["X_val"], data["y_val_class"]),
        "test": (data["X_test"], data["y_test_class"]),
    }

    for model_name, model in models.items():
        print(f"Training classification: {dataset}/{scenario}/{model_name}")
        try:
            model.fit(X_train, data["y_train_class"])
            model_dir = (
                PROJECT_ROOT
                / "models"
                / "saved"
                / "baselines"
                / dataset
                / scenario
                / "classification"
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_dir / f"{model_name}.joblib")

            split_metrics: dict[str, dict] = {}
            for split, (X, y_true) in split_data.items():
                y_pred = model.predict(X)
                metrics = classification_metrics(y_true, y_pred)
                split_metrics[split] = metrics
                row = {
                    "dataset": dataset,
                    "scenario": scenario,
                    "task": "classification",
                    "model_name": model_name,
                    "split": split,
                    "n_rows": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                    **metrics,
                    "seed": seed,
                }
                rows.append(row)

                prediction_path = (
                    PROJECT_ROOT
                    / "reports"
                    / "results"
                    / "baseline_predictions"
                    / dataset
                    / scenario
                    / "classification"
                    / f"{model_name}_{split}_predictions.csv"
                )
                save_predictions(prediction_path, y_true, y_pred)

                if split == "test":
                    figure_path = (
                        PROJECT_ROOT
                        / "reports"
                        / "figures"
                        / "baselines"
                        / dataset
                        / scenario
                        / f"{model_name}_test_confusion_matrix.png"
                    )
                    save_confusion_matrix_plot(
                        y_true,
                        y_pred,
                        figure_path,
                        f"{dataset} {scenario} {model_name} test confusion matrix",
                    )

            val = split_metrics["val"]
            test = split_metrics["test"]
            print(
                "  val f1_macro={:.4f}, test f1_macro={:.4f}, test accuracy={:.4f}".format(
                    val["f1_macro"],
                    test["f1_macro"],
                    test["accuracy"],
                )
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"  SKIPPED {model_name}: {reason}")
            skips.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "task": "classification",
                    "model_name": model_name,
                    "reason": reason,
                    "traceback": traceback.format_exc(),
                    "seed": seed,
                }
            )

    return rows, skips


def train_regression_models(
    dataset: str,
    scenario: str,
    data: dict,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    skips: list[dict] = []
    models = get_regression_baselines(seed=seed)
    X_train = data["X_train"]
    split_data = {
        "train": (data["X_train"], data["y_train_reg"]),
        "val": (data["X_val"], data["y_val_reg"]),
        "test": (data["X_test"], data["y_test_reg"]),
    }

    for model_name, model in models.items():
        print(f"Training regression: {dataset}/{scenario}/{model_name}")
        try:
            model.fit(X_train, data["y_train_reg"])
            model_dir = (
                PROJECT_ROOT
                / "models"
                / "saved"
                / "baselines"
                / dataset
                / scenario
                / "regression"
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_dir / f"{model_name}.joblib")

            split_metrics: dict[str, dict] = {}
            for split, (X, y_true) in split_data.items():
                y_pred = model.predict(X)
                metrics = regression_metrics(y_true, y_pred)
                split_metrics[split] = metrics
                rows.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "task": "regression",
                        "model_name": model_name,
                        "split": split,
                        "n_rows": int(X.shape[0]),
                        "n_features": int(X.shape[1]),
                        **metrics,
                        "seed": seed,
                    }
                )

                prediction_path = (
                    PROJECT_ROOT
                    / "reports"
                    / "results"
                    / "baseline_predictions"
                    / dataset
                    / scenario
                    / "regression"
                    / f"{model_name}_{split}_predictions.csv"
                )
                save_predictions(prediction_path, y_true, y_pred)

            val = split_metrics["val"]
            test = split_metrics["test"]
            print(
                "  val rmse={:.4f}, test rmse={:.4f}, test mae={:.4f}".format(
                    val["rmse"],
                    test["rmse"],
                    test["mae"],
                )
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"  SKIPPED {model_name}: {reason}")
            skips.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "task": "regression",
                    "model_name": model_name,
                    "reason": reason,
                    "traceback": traceback.format_exc(),
                    "seed": seed,
                }
            )

    return rows, skips


def add_optional_skip_notes(
    skip_rows: list[dict],
    dataset_names: list[str],
    scenarios: list[str],
    tasks: list[str],
    seed: int,
) -> None:
    optional_skips = optional_model_skip_reasons()
    for optional_skip in optional_skips:
        if optional_skip["task"] not in tasks:
            continue
        for dataset in dataset_names:
            for scenario in scenarios:
                skip_rows.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "task": optional_skip["task"],
                        "model_name": optional_skip["model_name"],
                        "reason": optional_skip["reason"],
                        "traceback": "",
                        "seed": seed,
                    }
                )


def write_training_report(skip_rows: list[dict]) -> None:
    report_path = PROJECT_ROOT / "reports" / "tables" / "baseline_training_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["Baseline Training Report", f"Project root: {PROJECT_ROOT}", ""]
    if skip_rows:
        lines.append("Skipped models:")
        for row in skip_rows:
            lines.append(
                "- {dataset}/{scenario}/{task}/{model_name}: {reason}".format(**row)
            )
    else:
        lines.append("Skipped models: none")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_names = selected(args.dataset, DATASETS)
    scenarios = selected(args.scenario, SCENARIOS)
    tasks = selected(args.task, TASKS)

    classification_rows: list[dict] = []
    regression_rows: list[dict] = []
    skip_rows: list[dict] = []
    add_optional_skip_notes(skip_rows, dataset_names, scenarios, tasks, args.seed)

    for dataset in dataset_names:
        for scenario in scenarios:
            print(f"\nLoading processed data: dataset={dataset}, scenario={scenario}")
            data = load_processed_split(dataset, scenario)
            print(
                "Processed data OK: train={}, val={}, test={}, n_features={}".format(
                    data["X_train"].shape[0],
                    data["X_val"].shape[0],
                    data["X_test"].shape[0],
                    data["X_train"].shape[1],
                )
            )

            if "classification" in tasks:
                rows, skips = train_classification_models(dataset, scenario, data, args.seed)
                classification_rows.extend(rows)
                skip_rows.extend(skips)
            if "regression" in tasks:
                rows, skips = train_regression_models(dataset, scenario, data, args.seed)
                regression_rows.extend(rows)
                skip_rows.extend(skips)

    upsert_results(
        PROJECT_ROOT / "reports" / "results" / "baseline_classification_results.csv",
        classification_rows,
        ["dataset", "scenario", "task", "model_name", "split", "seed"],
    )
    upsert_results(
        PROJECT_ROOT / "reports" / "results" / "baseline_regression_results.csv",
        regression_rows,
        ["dataset", "scenario", "task", "model_name", "split", "seed"],
    )
    upsert_results(
        PROJECT_ROOT / "reports" / "results" / "baseline_skipped_models.csv",
        skip_rows,
        ["dataset", "scenario", "task", "model_name", "seed"],
    )
    write_training_report(skip_rows)

    print("\nBaseline training completed.")
    if skip_rows:
        print(f"Skipped model records: {len(skip_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
