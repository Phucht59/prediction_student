from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _feature_refers_to_column(feature_name: str, column: str) -> bool:
    suffix = feature_name.split("__", 1)[-1]
    return suffix == column or suffix.endswith(f"__{column}") or suffix.endswith(f"_{column}")


def _contains_column(feature_names: list[str], column: str) -> bool:
    return any(_feature_refers_to_column(name, column) for name in feature_names)


def load_processed_classification_data(dataset_name: str, scenario: str, project_root) -> dict:
    root = Path(project_root)
    split_dir = root / "data" / "processed" / dataset_name / scenario
    required_files = (
        "X_train.npy",
        "X_val.npy",
        "X_test.npy",
        "y_train_class.npy",
        "y_val_class.npy",
        "y_test_class.npy",
        "train_raw.csv",
        "val_raw.csv",
        "test_raw.csv",
        "feature_names.json",
        "metadata.json",
    )
    missing = [filename for filename in required_files if not (split_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed deep-learning files in {split_dir}: {missing}")

    data = {
        "split_dir": split_dir,
        "X_train": np.load(split_dir / "X_train.npy", allow_pickle=False),
        "X_val": np.load(split_dir / "X_val.npy", allow_pickle=False),
        "X_test": np.load(split_dir / "X_test.npy", allow_pickle=False),
        "y_train": np.load(split_dir / "y_train_class.npy", allow_pickle=False),
        "y_val": np.load(split_dir / "y_val_class.npy", allow_pickle=False),
        "y_test": np.load(split_dir / "y_test_class.npy", allow_pickle=False),
        "train_raw": pd.read_csv(split_dir / "train_raw.csv"),
        "val_raw": pd.read_csv(split_dir / "val_raw.csv"),
        "test_raw": pd.read_csv(split_dir / "test_raw.csv"),
        "feature_names": json.loads((split_dir / "feature_names.json").read_text(encoding="utf-8")),
        "metadata": json.loads((split_dir / "metadata.json").read_text(encoding="utf-8")),
    }
    validate_processed_classification_data(dataset_name, scenario, data)
    return data


def validate_processed_classification_data(dataset_name: str, scenario: str, data: dict) -> None:
    for split in ("train", "val", "test"):
        X = data[f"X_{split}"]
        y = data[f"y_{split}"]
        if X.ndim != 2:
            raise ValueError(f"{dataset_name}/{scenario}/{split}: X must be 2D, got {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"{dataset_name}/{scenario}/{split}: X/y row mismatch.")
        if np.isnan(X).any():
            raise ValueError(f"{dataset_name}/{scenario}/{split}: X contains NaN.")
        labels = set(np.unique(y).tolist())
        if not labels.issubset({0, 1, 2}):
            raise ValueError(f"{dataset_name}/{scenario}/{split}: y has invalid labels {labels}.")

    leakage = data["metadata"].get("leakage_checks", {})
    if leakage.get("passed") is not True:
        raise ValueError(f"{dataset_name}/{scenario}: metadata leakage check failed: {leakage}")

    feature_names = data["feature_names"]
    if _contains_column(feature_names, "G3"):
        raise ValueError(f"{dataset_name}/{scenario}: feature_names contain G3.")
    if scenario == "mid" and _contains_column(feature_names, "G2"):
        raise ValueError(f"{dataset_name}/{scenario}: mid feature_names contain G2.")


def find_grade_feature_indices(feature_names: list[str]) -> list[int]:
    indices: list[int] = []
    for index, feature_name in enumerate(feature_names):
        suffix = feature_name.split("__", 1)[-1]
        if suffix in {"G1", "G2"} or feature_name in {"G1", "G2"}:
            indices.append(index)
        elif feature_name.endswith("__G1") or feature_name.endswith("__G2"):
            indices.append(index)
    return indices


def remove_grade_features_from_X(X, feature_names: list[str]):
    removed_indices = find_grade_feature_indices(feature_names)
    keep_indices = [index for index in range(len(feature_names)) if index not in removed_indices]
    X_static = X[:, keep_indices]
    static_feature_names = [feature_names[index] for index in keep_indices]
    return X_static, static_feature_names, removed_indices


def build_grade_sequence_from_raw(raw_df: pd.DataFrame, scenario: str):
    if scenario == "mid":
        required_columns = ["G1"]
    elif scenario == "late":
        required_columns = ["G1", "G2"]
    else:
        raise ValueError(f"Invalid scenario '{scenario}'.")

    missing = [column for column in required_columns if column not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing grade columns for {scenario}: {missing}")
    if "G3" in required_columns:
        raise ValueError("G3 must not be used in grade sequence.")

    grades = raw_df[required_columns].astype("float32").to_numpy() / 20.0
    if np.isnan(grades).any():
        raise ValueError(f"Grade sequence for {scenario} contains NaN values.")
    return grades[:, :, None].astype("float32")
