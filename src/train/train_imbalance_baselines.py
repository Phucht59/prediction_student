from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import classification_metrics, save_confusion_matrix_plot  # noqa: E402
from src.features.imbalance import (  # noqa: E402
    clone_with_class_weight,
    class_distribution,
    resample_train_data,
    supports_class_weight,
)
from src.models.baselines import get_classification_baselines  # noqa: E402


DATASETS = ("student-mat", "student-por", "student-combined")
SCENARIOS = ("mid", "late")
MODEL_NAMES = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "svm_rbf",
)
IMBALANCE_STRATEGIES = (
    "none",
    "random_over",
    "smote",
    "borderline_smote",
    "adasyn",
    "class_weight_balanced",
)
REQUIRED_FILES = (
    "X_train.npy",
    "X_val.npy",
    "X_test.npy",
    "y_train_class.npy",
    "y_val_class.npy",
    "y_test_class.npy",
    "metadata.json",
    "feature_names.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train imbalance handling classification baselines.")
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--strategy", choices=["all", *IMBALANCE_STRATEGIES], default="all")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def selected(selection: str, values: tuple[str, ...]) -> list[str]:
    return list(values) if selection == "all" else [selection]


def load_processed_data(dataset: str, scenario: str) -> dict:
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
        "y_train": np.load(split_dir / "y_train_class.npy", allow_pickle=False),
        "y_val": np.load(split_dir / "y_val_class.npy", allow_pickle=False),
        "y_test": np.load(split_dir / "y_test_class.npy", allow_pickle=False),
        "metadata": json.loads((split_dir / "metadata.json").read_text(encoding="utf-8")),
        "feature_names": json.loads((split_dir / "feature_names.json").read_text(encoding="utf-8")),
    }
    validate_processed_data(dataset, scenario, data)
    return data


def validate_processed_data(dataset: str, scenario: str, data: dict) -> None:
    for split in ("train", "val", "test"):
        X = data[f"X_{split}"]
        y = data[f"y_{split}"]
        if X.ndim != 2:
            raise ValueError(f"{dataset}/{scenario}/{split}: X must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"{dataset}/{scenario}/{split}: X/y row mismatch {X.shape} vs {y.shape}")
        if np.isnan(X).any():
            raise ValueError(f"{dataset}/{scenario}/{split}: X contains NaN values")
        labels = set(np.unique(y).tolist())
        if not labels.issubset({0, 1, 2}):
            raise ValueError(f"{dataset}/{scenario}/{split}: y has invalid class labels {labels}")

    leakage = data["metadata"].get("leakage_checks", {})
    if leakage.get("passed") is not True:
        raise ValueError(f"{dataset}/{scenario}: leakage check did not pass: {leakage}")


def model_registry(seed: int) -> dict:
    all_models = get_classification_baselines(seed=seed)
    return {model_name: all_models[model_name] for model_name in MODEL_NAMES}


def prepare_training_data(strategy: str, data: dict, seed: int):
    X_train = data["X_train"]
    y_train = data["y_train"]
    before_distribution = class_distribution(y_train)

    if strategy == "class_weight_balanced":
        return X_train, y_train, before_distribution, before_distribution

    return resample_train_data(X_train, y_train, strategy=strategy, seed=seed)


def build_model_for_strategy(base_model, strategy: str):
    if strategy == "class_weight_balanced":
        return clone_with_class_weight(base_model)
    return clone(base_model)


def save_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(path, index=False)


def upsert_csv(path: Path, rows: list[dict], key_columns: list[str]) -> None:
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


def append_skip(
    skipped_rows: list[dict],
    dataset: str,
    scenario: str,
    model_name: str,
    strategy: str,
    reason: str,
    trace: str = "",
) -> None:
    skipped_rows.append(
        {
            "dataset": dataset,
            "scenario": scenario,
            "model_name": model_name,
            "imbalance_strategy": strategy,
            "reason": reason,
            "traceback": trace,
        }
    )


def train_one_configuration(
    *,
    dataset: str,
    scenario: str,
    model_name: str,
    strategy: str,
    model,
    data: dict,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    before_distribution: dict[int, int],
    after_distribution: dict[int, int],
    seed: int,
) -> list[dict]:
    print(f"Training imbalance: {dataset}/{scenario}/{strategy}/{model_name}")
    model.fit(X_fit, y_fit)

    model_dir = (
        PROJECT_ROOT
        / "models"
        / "saved"
        / "imbalance_baselines"
        / dataset
        / scenario
        / strategy
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / f"{model_name}.joblib")

    split_inputs = {
        "train": (data["X_train"], data["y_train"]),
        "val": (data["X_val"], data["y_val"]),
        "test": (data["X_test"], data["y_test"]),
    }
    rows: list[dict] = []
    split_metrics: dict[str, dict] = {}

    for split, (X, y_true) in split_inputs.items():
        y_pred = model.predict(X)
        metrics = classification_metrics(y_true, y_pred)
        split_metrics[split] = metrics
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "task": "classification",
                "model_name": model_name,
                "imbalance_strategy": strategy,
                "split": split,
                "n_rows_original_train": int(data["X_train"].shape[0]),
                "n_rows_after_resampling": int(X_fit.shape[0]),
                "n_features": int(data["X_train"].shape[1]),
                "class_distribution_train_before": json.dumps(before_distribution, sort_keys=True),
                "class_distribution_train_after": json.dumps(after_distribution, sort_keys=True),
                **metrics,
                "seed": seed,
            }
        )

        prediction_path = (
            PROJECT_ROOT
            / "reports"
            / "results"
            / "imbalance_predictions"
            / dataset
            / scenario
            / strategy
            / f"{model_name}_{split}_predictions.csv"
        )
        save_predictions(prediction_path, y_true, y_pred)

        if split == "test":
            figure_path = (
                PROJECT_ROOT
                / "reports"
                / "figures"
                / "imbalance_baselines"
                / dataset
                / scenario
                / strategy
                / f"{model_name}_test_confusion_matrix.png"
            )
            save_confusion_matrix_plot(
                y_true,
                y_pred,
                figure_path,
                f"{dataset} {scenario} {strategy} {model_name} test confusion matrix",
            )

    val = split_metrics["val"]
    test = split_metrics["test"]
    print(
        "  val f1_macro={:.4f}, val recall_weak={:.4f}, "
        "test f1_macro={:.4f}, test recall_weak={:.4f}".format(
            val["f1_macro"],
            val["recall_weak"],
            test["f1_macro"],
            test["recall_weak"],
        )
    )
    return rows


def process_dataset_scenario(
    dataset: str,
    scenario: str,
    strategies: list[str],
    seed: int,
    result_rows: list[dict],
    skipped_rows: list[dict],
) -> None:
    print(f"\nLoading processed data: dataset={dataset}, scenario={scenario}")
    data = load_processed_data(dataset, scenario)
    print(
        "Processed data OK: train={}, val={}, test={}, n_features={}, train_distribution={}".format(
            data["X_train"].shape[0],
            data["X_val"].shape[0],
            data["X_test"].shape[0],
            data["X_train"].shape[1],
            class_distribution(data["y_train"]),
        )
    )

    models = model_registry(seed)
    for strategy in strategies:
        try:
            X_fit, y_fit, before_distribution, after_distribution = prepare_training_data(
                strategy,
                data,
                seed,
            )
            print(
                f"Strategy={strategy}: before={before_distribution}, "
                f"after={after_distribution}"
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"  SKIPPED strategy {strategy} for all models: {reason}")
            for model_name in MODEL_NAMES:
                append_skip(
                    skipped_rows,
                    dataset,
                    scenario,
                    model_name,
                    strategy,
                    reason,
                    traceback.format_exc(),
                )
            continue

        for model_name, base_model in models.items():
            if strategy == "class_weight_balanced" and not supports_class_weight(base_model):
                reason = f"Model {type(base_model).__name__} does not support class_weight."
                print(f"  SKIPPED {strategy}/{model_name}: {reason}")
                append_skip(skipped_rows, dataset, scenario, model_name, strategy, reason)
                continue

            try:
                model = build_model_for_strategy(base_model, strategy)
                rows = train_one_configuration(
                    dataset=dataset,
                    scenario=scenario,
                    model_name=model_name,
                    strategy=strategy,
                    model=model,
                    data=data,
                    X_fit=X_fit,
                    y_fit=y_fit,
                    before_distribution=before_distribution,
                    after_distribution=after_distribution,
                    seed=seed,
                )
                result_rows.extend(rows)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                print(f"  SKIPPED {strategy}/{model_name}: {reason}")
                append_skip(
                    skipped_rows,
                    dataset,
                    scenario,
                    model_name,
                    strategy,
                    reason,
                    traceback.format_exc(),
                )


def write_training_report(skipped_rows: list[dict]) -> None:
    report_path = PROJECT_ROOT / "reports" / "tables" / "imbalance_training_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Imbalance Training Report", f"Project root: {PROJECT_ROOT}", ""]
    if skipped_rows:
        lines.append("Skipped records:")
        for row in skipped_rows:
            lines.append(
                "- {dataset}/{scenario}/{imbalance_strategy}/{model_name}: {reason}".format(
                    **row
                )
            )
    else:
        lines.append("Skipped records: none")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    datasets = selected(args.dataset, DATASETS)
    scenarios = selected(args.scenario, SCENARIOS)
    strategies = selected(args.strategy, IMBALANCE_STRATEGIES)

    result_rows: list[dict] = []
    skipped_rows: list[dict] = []

    for dataset in datasets:
        for scenario in scenarios:
            process_dataset_scenario(
                dataset,
                scenario,
                strategies,
                args.seed,
                result_rows,
                skipped_rows,
            )

    upsert_csv(
        PROJECT_ROOT / "reports" / "results" / "imbalance_classification_results.csv",
        result_rows,
        ["dataset", "scenario", "model_name", "imbalance_strategy", "split", "seed"],
    )
    upsert_csv(
        PROJECT_ROOT / "reports" / "results" / "imbalance_skipped_records.csv",
        skipped_rows,
        ["dataset", "scenario", "model_name", "imbalance_strategy"],
    )
    write_training_report(skipped_rows)

    print("\nImbalance classification training completed.")
    print(f"Result rows written/updated: {len(result_rows)}")
    print(f"Skipped records: {len(skipped_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
