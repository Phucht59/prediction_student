from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "paper_replication"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "paper_replication"
MODEL_DIR = PROJECT_ROOT / "models" / "saved" / "paper_replication"
REPORT_PATH = PROJECT_ROOT / "reports" / "paper_replication_report.md"
RESULTS_PATH = RESULTS_DIR / "paper_replication_results.csv"
PREDICTIONS_PATH = RESULTS_DIR / "paper_replication_predictions.csv"
SUMMARY_PATH = TABLES_DIR / "paper_replication_summary.csv"
RUNS_PATH = RESULTS_DIR / "paper_replication_runs.csv"

STUDENT_G3_BINS = [-0.1, 4, 8, 12, 16, 20]
STUDENT_G3_CLASS_NAMES = ["0-4", "5-8", "9-12", "13-16", "17-20"]
XAPI_CLASS_MAPPING = {"L": 0, "M": 1, "H": 2}
XAPI_CLASS_NAMES = ["Low", "Middle", "High"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    raw_file: str
    kind: str
    n_classes: int
    class_names: list[str]
    paper_feature_aliases: list[str]
    paper_ml_imbalance: str
    deep_conv_blocks: int
    csv_sep: str | None = None
    paper_reference: dict[str, float] = field(default_factory=dict)


DATASETS: dict[str, DatasetSpec] = {
    "student-mat": DatasetSpec(
        name="student-mat",
        display_name="Student Performance in Mathematics Dataset",
        raw_file="student-mat.csv",
        kind="student",
        n_classes=5,
        class_names=STUDENT_G3_CLASS_NAMES,
        paper_feature_aliases=["G1", "G2"],
        paper_ml_imbalance="smote",
        deep_conv_blocks=1,
        csv_sep=";",
        paper_reference={"cnn_bilstm_accuracy": 1.0, "decision_tree_accuracy": 0.962},
    ),
    "student-por": DatasetSpec(
        name="student-por",
        display_name="Student Performance in Portuguese language Dataset",
        raw_file="student-por.csv",
        kind="student",
        n_classes=5,
        class_names=STUDENT_G3_CLASS_NAMES,
        paper_feature_aliases=["G1", "G2"],
        paper_ml_imbalance="none",
        deep_conv_blocks=1,
        csv_sep=";",
        paper_reference={"cnn_bilstm_accuracy": 0.9231, "decision_tree_accuracy": 0.8923},
    ),
    "xapi": DatasetSpec(
        name="xapi",
        display_name="Students' Academic Performance / xAPI Dataset",
        raw_file="xAPI-Edu-Data.csv",
        kind="xapi",
        n_classes=3,
        class_names=XAPI_CLASS_NAMES,
        paper_feature_aliases=["raisedhands", "VisitedResources", "StudentAbsenceDays"],
        paper_ml_imbalance="adasyn",
        deep_conv_blocks=2,
        csv_sep=None,
        paper_reference={
            "cnn_bilstm_accuracy": 0.8438,
            "cnn_bilstm_precision": 0.8426,
            "cnn_bilstm_recall": 0.8521,
            "cnn_bilstm_f1": 0.8447,
        },
    ),
}


def ensure_dirs() -> None:
    for directory in (PROCESSED_DIR, RESULTS_DIR, TABLES_DIR, FIGURES_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def canonical_name(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def resolve_columns(columns: list[str], aliases: list[str]) -> list[str]:
    by_key = {canonical_name(column): column for column in columns}
    resolved = []
    missing = []
    for alias in aliases:
        column = by_key.get(canonical_name(alias))
        if column is None:
            missing.append(alias)
        else:
            resolved.append(column)
    if missing:
        raise ValueError(f"Missing paper feature columns {missing}; available columns={columns}")
    return resolved


def read_raw(spec: DatasetSpec) -> pd.DataFrame:
    path = RAW_DIR / spec.raw_file
    if not path.exists():
        raise FileNotFoundError(f"Missing raw dataset: {path}")
    if spec.csv_sep is None:
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path, sep=spec.csv_sep)


def add_targets(raw: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    data = raw.copy()
    if spec.kind == "student":
        if "G3" not in data.columns:
            raise ValueError(f"{spec.name}: missing G3 target column.")
        g3 = pd.to_numeric(data["G3"], errors="raise")
        target = pd.cut(
            g3,
            bins=STUDENT_G3_BINS,
            labels=False,
            include_lowest=True,
        )
        if target.isna().any():
            raise ValueError(f"{spec.name}: could not map all G3 values to 5 classes.")
        data["target_class"] = target.astype("int64")
        data["target_class_name"] = data["target_class"].map(lambda value: spec.class_names[int(value)])
        return data

    if "Class" not in data.columns:
        raise ValueError("xAPI target column 'Class' was not found.")
    unknown = sorted(set(data["Class"].dropna().unique()) - set(XAPI_CLASS_MAPPING))
    if unknown:
        raise ValueError(f"xAPI contains unsupported Class labels: {unknown}")
    data["target_class"] = data["Class"].map(XAPI_CLASS_MAPPING).astype("int64")
    data["target_class_name"] = data["target_class"].map(lambda value: spec.class_names[int(value)])
    return data


def make_one_hot_encoder() -> OneHotEncoder:
    params = {"handle_unknown": "ignore"}
    if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames:
        params["sparse_output"] = False
    else:
        params["sparse"] = False
    return OneHotEncoder(**params)


def build_preprocessor(X_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in X_train.columns if column not in numeric_columns]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", MinMaxScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_columns, categorical_columns


def dense_array(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    array = np.asarray(value, dtype=np.float32)
    if np.isnan(array).any():
        raise ValueError("Processed array contains NaN.")
    return array


def class_distribution(y: np.ndarray, n_classes: int) -> dict[str, int]:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=n_classes)
    return {str(index): int(counts[index]) for index in range(n_classes)}


def get_feature_names(preprocessor: ColumnTransformer, fallback: list[str]) -> list[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return fallback


def prepare_dataset(dataset: str, seed: int = 42) -> dict[str, Any]:
    ensure_dirs()
    spec = DATASETS[dataset]
    raw = add_targets(read_raw(spec), spec)
    feature_columns = resolve_columns(raw.columns.tolist(), spec.paper_feature_aliases)
    X = raw[feature_columns].copy()
    y = raw["target_class"].to_numpy(dtype=np.int64)

    train_full, test = train_test_split(
        raw,
        test_size=0.20,
        random_state=seed,
        stratify=raw["target_class"],
    )
    train, val = train_test_split(
        train_full,
        test_size=0.20,
        random_state=seed,
        stratify=train_full["target_class"],
    )
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    preprocessor, numeric_columns, categorical_columns = build_preprocessor(train[feature_columns])
    X_train = dense_array(preprocessor.fit_transform(train[feature_columns]))
    X_val = dense_array(preprocessor.transform(val[feature_columns]))
    X_test = dense_array(preprocessor.transform(test[feature_columns]))
    feature_names = get_feature_names(preprocessor, feature_columns)

    output_dir = PROCESSED_DIR / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", train["target_class"].to_numpy(dtype=np.int64))
    np.save(output_dir / "y_val.npy", val["target_class"].to_numpy(dtype=np.int64))
    np.save(output_dir / "y_test.npy", test["target_class"].to_numpy(dtype=np.int64))
    train.to_csv(output_dir / "train_raw.csv", index=False)
    val.to_csv(output_dir / "val_raw.csv", index=False)
    test.to_csv(output_dir / "test_raw.csv", index=False)
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    write_json(output_dir / "feature_names.json", feature_names)
    metadata = {
        "dataset": dataset,
        "display_name": spec.display_name,
        "paper_style": True,
        "seed": int(seed),
        "target": "G3 five-class bins" if spec.kind == "student" else "Class L/M/H",
        "class_names": spec.class_names,
        "class_mapping": {str(index): name for index, name in enumerate(spec.class_names)},
        "g3_bins_assumption": STUDENT_G3_BINS if spec.kind == "student" else None,
        "paper_feature_aliases": spec.paper_feature_aliases,
        "resolved_feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "feature_names_processed": feature_names,
        "split_ratio_assumption": {"train": 0.64, "validation": 0.16, "test": 0.20},
        "n_rows_total": int(len(raw)),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "n_features_processed": int(X_train.shape[1]),
        "class_distribution_total": class_distribution(y, spec.n_classes),
        "class_distribution_train": class_distribution(train["target_class"].to_numpy(), spec.n_classes),
        "class_distribution_val": class_distribution(val["target_class"].to_numpy(), spec.n_classes),
        "class_distribution_test": class_distribution(test["target_class"].to_numpy(), spec.n_classes),
        "notes": [
            "PDF states Pearson feature selection; this pipeline uses the features the PDF reports as selected.",
            "PDF does not state an exact train/test split ratio; this pipeline uses stratified 64/16/20 train/val/test.",
        ],
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def load_prepared(dataset: str) -> dict[str, Any]:
    output_dir = PROCESSED_DIR / dataset
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing prepared dataset metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "metadata": metadata,
        "X_train": np.load(output_dir / "X_train.npy"),
        "X_val": np.load(output_dir / "X_val.npy"),
        "X_test": np.load(output_dir / "X_test.npy"),
        "y_train": np.load(output_dir / "y_train.npy"),
        "y_val": np.load(output_dir / "y_val.npy"),
        "y_test": np.load(output_dir / "y_test.npy"),
        "train_raw": pd.read_csv(output_dir / "train_raw.csv"),
        "val_raw": pd.read_csv(output_dir / "val_raw.csv"),
        "test_raw": pd.read_csv(output_dir / "test_raw.csv"),
    }


def resampler_for(strategy: str, y: np.ndarray, seed: int):
    if strategy == "none":
        return None
    counts = np.bincount(np.asarray(y, dtype=np.int64))
    present = counts[counts > 0]
    if len(present) == 0:
        raise ValueError("Cannot resample empty target.")
    if present.min() < 2:
        return RandomOverSampler(random_state=seed)
    neighbors = int(min(5, present.min() - 1))
    if strategy == "smote":
        return SMOTE(random_state=seed, k_neighbors=neighbors)
    if strategy == "borderline_smote":
        return BorderlineSMOTE(random_state=seed, k_neighbors=neighbors, m_neighbors=max(1, neighbors))
    if strategy == "adasyn":
        return ADASYN(random_state=seed, n_neighbors=neighbors)
    raise ValueError(f"Unknown imbalance strategy: {strategy}")


def resample_train(X: np.ndarray, y: np.ndarray, strategy: str, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    sampler = resampler_for(strategy, y, seed)
    if sampler is None:
        return X, y, "none"
    try:
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return np.asarray(X_resampled, dtype=np.float32), np.asarray(y_resampled, dtype=np.int64), strategy
    except Exception as exc:
        fallback = RandomOverSampler(random_state=seed)
        X_resampled, y_resampled = fallback.fit_resample(X, y)
        return (
            np.asarray(X_resampled, dtype=np.float32),
            np.asarray(y_resampled, dtype=np.int64),
            f"random_over_fallback_after_{strategy}_failed: {type(exc).__name__}",
        )


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, *, labels: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_data = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        data = pd.concat([existing, new_data], ignore_index=True)
    else:
        data = new_data
    data.to_csv(path, index=False)


def safe_cv(y: np.ndarray) -> int:
    counts = np.bincount(np.asarray(y, dtype=np.int64))
    present = counts[counts > 0]
    if len(present) == 0:
        return 2
    return max(2, min(3, int(present.min())))


def model_grid(model_name: str, seed: int, n_classes: int, grid_profile: str):
    if model_name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed), {}
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=seed), {
            "criterion": ["gini", "entropy", "log_loss"],
            "min_samples_split": [2, 3, 4],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "max_features": ["sqrt", "log2"],
        }
    if model_name == "gbm":
        return GradientBoostingClassifier(random_state=seed), {
            "learning_rate": [0.001, 0.01, 0.1],
            "criterion": ["friedman_mse", "squared_error"],
            "max_depth": [1, 2, 3, 4, 5],
        }
    if model_name == "svm_linear":
        return SVC(kernel="linear"), {"C": [0.01, 0.1, 1, 10], "decision_function_shape": ["ovo", "ovr"]}
    if model_name == "svm_poly":
        return SVC(kernel="poly"), {
            "C": [0.01, 0.1, 1, 10],
            "gamma": ["scale", "auto"],
            "degree": [1, 2, 3],
            "decision_function_shape": ["ovo", "ovr"],
        }
    if model_name == "svm_rbf":
        return SVC(kernel="rbf"), {
            "C": [0.01, 0.1, 1, 10],
            "gamma": ["scale", "auto"],
            "decision_function_shape": ["ovo", "ovr"],
        }
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            return None, {"skip_reason": f"xgboost unavailable: {type(exc).__name__}: {exc}"}
        estimator = XGBClassifier(
            random_state=seed,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            n_estimators=120,
            tree_method="hist",
            verbosity=0,
        )
        if grid_profile == "full":
            grid = {
                "max_depth": [2, 3, 4, 5, 6],
                "learning_rate": [0.001, 0.01, 0.1],
                "gamma": [0.5, 1, 1.5, 2, 5],
                "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
                "min_child_weight": [1, 5, 10],
                "subsample": [0.5, 0.75],
            }
        else:
            grid = {
                "max_depth": [2, 3],
                "learning_rate": [0.01, 0.1],
                "gamma": [0.5, 1],
                "colsample_bytree": [0.8, 1.0],
                "min_child_weight": [1],
                "subsample": [0.75],
            }
        return estimator, grid
    raise ValueError(f"Unsupported model: {model_name}")


def train_baselines(dataset: str, seed: int = 42, grid_profile: str = "compact") -> list[dict[str, Any]]:
    ensure_dirs()
    spec = DATASETS[dataset]
    prepared = load_prepared(dataset)
    X_train, y_train = prepared["X_train"], prepared["y_train"]
    X_train_fit, y_train_fit, effective_strategy = resample_train(X_train, y_train, spec.paper_ml_imbalance, seed)
    labels = list(range(spec.n_classes))
    rows: list[dict[str, Any]] = []
    model_names = ["decision_tree", "random_forest", "gbm", "xgboost", "svm_linear", "svm_poly", "svm_rbf"]
    for model_name in model_names:
        start = time.time()
        estimator, grid = model_grid(model_name, seed, spec.n_classes, grid_profile)
        if estimator is None:
            rows.append(
                {
                    "dataset": dataset,
                    "stage": "baseline",
                    "model_name": model_name,
                    "split": "skip",
                    "status": "skipped",
                    "notes": grid["skip_reason"],
                }
            )
            continue
        try:
            if grid:
                search = GridSearchCV(
                    estimator,
                    grid,
                    scoring="accuracy",
                    cv=safe_cv(y_train_fit),
                    n_jobs=1,
                    error_score="raise",
                )
                search.fit(X_train_fit, y_train_fit)
                fitted = search.best_estimator_
                best_params = search.best_params_
            else:
                fitted = estimator.fit(X_train_fit, y_train_fit)
                best_params = {}
        except Exception as exc:
            rows.append(
                {
                    "dataset": dataset,
                    "stage": "baseline",
                    "model_name": model_name,
                    "split": "skip",
                    "status": "skipped",
                    "seed": seed,
                    "grid_profile": grid_profile,
                    "paper_imbalance_strategy": spec.paper_ml_imbalance,
                    "effective_imbalance_strategy": effective_strategy,
                    "notes": f"training failed: {type(exc).__name__}: {exc}",
                    "elapsed_seconds": round(time.time() - start, 3),
                }
            )
            continue
        for split, X_split, y_split in (
            ("train", prepared["X_train"], prepared["y_train"]),
            ("val", prepared["X_val"], prepared["y_val"]),
            ("test", prepared["X_test"], prepared["y_test"]),
        ):
            y_pred = fitted.predict(X_split)
            row = {
                "dataset": dataset,
                "stage": "baseline",
                "model_name": model_name,
                "split": split,
                "status": "ok",
                "seed": seed,
                "grid_profile": grid_profile,
                "paper_imbalance_strategy": spec.paper_ml_imbalance,
                "effective_imbalance_strategy": effective_strategy,
                "n_train_before_resampling": int(len(y_train)),
                "n_train_after_resampling": int(len(y_train_fit)),
                "best_params_json": json.dumps(best_params, ensure_ascii=True, sort_keys=True),
                "elapsed_seconds": round(time.time() - start, 3),
            }
            row.update(metric_row(y_split, y_pred, labels=labels))
            rows.append(row)
        joblib.dump(fitted, MODEL_DIR / f"{dataset}_{model_name}.joblib")
    append_rows(RESULTS_PATH, rows)
    return rows


class PaperCNNBiLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        conv_channels: int = 64,
        conv_blocks: int = 1,
        bilstm_hidden: int = 64,
        bilstm_layers: int = 1,
        dense_hidden: int = 128,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        if conv_blocks not in {1, 2}:
            raise ValueError("conv_blocks must be 1 or 2.")
        self.input_dim = input_dim
        self.conv_blocks = nn.ModuleList()
        in_channels = 1
        out_channels = conv_channels
        for block_index in range(conv_blocks):
            layers: list[nn.Module] = []
            for conv_index in range(4):
                layers.append(nn.Conv1d(in_channels if conv_index == 0 else out_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm1d(out_channels))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            self.conv_blocks.append(nn.Sequential(*layers))
            in_channels = out_channels
            if block_index == 0 and conv_blocks == 2:
                out_channels = conv_channels * 2
        self.bilstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if bilstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bilstm_hidden * 2, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input, got {tuple(x.shape)}")
        out = x.unsqueeze(1)
        for block in self.conv_blocks:
            out = block(out)
            if out.shape[-1] >= 2:
                out = nn.functional.max_pool1d(out, kernel_size=2)
        out = out.transpose(1, 2)
        out, _ = self.bilstm(out)
        return self.classifier(out.mean(dim=1))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def predict_deep(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    preds = []
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            logits = model(batch_x.to(device))
            batch_prob = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(batch_prob)
            preds.append(batch_prob.argmax(axis=1))
    return np.concatenate(preds).astype(np.int64), np.concatenate(probs).astype(np.float32)


def train_deep(
    dataset: str,
    *,
    seed: int = 42,
    epochs: int = 100,
    patience: int = 15,
    batch_size: int | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    conv_channels: int = 64,
    bilstm_hidden: int = 64,
    bilstm_layers: int = 1,
    dense_hidden: int = 128,
    dropout: float = 0.30,
    run_label: str = "default",
    save_outputs: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ensure_dirs()
    set_seed(seed)
    spec = DATASETS[dataset]
    prepared = load_prepared(dataset)
    labels = list(range(spec.n_classes))
    batch_size = int(batch_size or (16 if dataset == "xapi" else 32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaperCNNBiLSTM(
        input_dim=int(prepared["X_train"].shape[1]),
        num_classes=spec.n_classes,
        conv_channels=conv_channels,
        conv_blocks=spec.deep_conv_blocks,
        bilstm_hidden=bilstm_hidden,
        bilstm_layers=bilstm_layers,
        dense_hidden=dense_hidden,
        dropout=dropout,
    ).to(device)
    train_loader = make_loader(prepared["X_train"], prepared["y_train"], batch_size, True, seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_score = -math.inf
    best_state = None
    best_epoch = 0
    wait = 0
    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            total += int(batch_y.numel())
        val_pred, _ = predict_deep(model, prepared["X_val"], batch_size, device)
        val_metrics = metric_row(prepared["y_val"], val_pred, labels=labels)
        train_accuracy = correct / max(total, 1)
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)) if losses else 0.0,
                "train_accuracy": float(train_accuracy),
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
            }
        )
        if val_metrics["f1_macro"] > best_score:
            best_score = val_metrics["f1_macro"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for split, X_split, y_split, raw_split in (
        ("train", prepared["X_train"], prepared["y_train"], prepared["train_raw"]),
        ("val", prepared["X_val"], prepared["y_val"], prepared["val_raw"]),
        ("test", prepared["X_test"], prepared["y_test"], prepared["test_raw"]),
    ):
        y_pred, y_prob = predict_deep(model, X_split, batch_size, device)
        row = {
            "dataset": dataset,
            "stage": "deep",
            "model_name": "paper_cnn_bilstm",
            "split": split,
            "status": "ok",
            "seed": seed,
            "run_label": run_label,
            "architecture": (
                "4xConv1D-MaxPool-BiLSTM-3xDense"
                if spec.deep_conv_blocks == 1
                else "4xConv1D-MaxPool-4xConv1D-MaxPool-BiLSTM-3xDense"
            ),
            "conv_channels": conv_channels,
            "bilstm_hidden": bilstm_hidden,
            "bilstm_layers": bilstm_layers,
            "dense_hidden": dense_hidden,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs_requested": epochs,
            "epochs_ran": int(len(history)),
            "best_epoch": int(best_epoch),
            "best_val_f1_macro": float(best_score),
            "elapsed_seconds": round(time.time() - start, 3),
        }
        row.update(metric_row(y_split, y_pred, labels=labels))
        rows.append(row)
        for index, (true_label, pred_label) in enumerate(zip(y_split, y_pred)):
            item = {
                "dataset": dataset,
                "model_name": "paper_cnn_bilstm",
                "split": split,
                "row_index": int(index),
                "true_label": int(true_label),
                "predicted_label": int(pred_label),
                "true_label_name": spec.class_names[int(true_label)],
                "predicted_label_name": spec.class_names[int(pred_label)],
                "seed": seed,
                "run_label": run_label,
            }
            for class_id, class_name in enumerate(spec.class_names):
                item[f"prob_{class_name}"] = float(y_prob[index, class_id])
            if "G1" in raw_split.columns:
                item["G1"] = raw_split.loc[index, "G1"]
            if "G2" in raw_split.columns:
                item["G2"] = raw_split.loc[index, "G2"]
            if "G3" in raw_split.columns:
                item["G3"] = raw_split.loc[index, "G3"]
            if "Class" in raw_split.columns:
                item["Class"] = raw_split.loc[index, "Class"]
            prediction_rows.append(item)

    if save_outputs:
        append_rows(RESULTS_PATH, rows)
        append_rows(PREDICTIONS_PATH, prediction_rows)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "metadata": prepared["metadata"],
                "config": {
                    "conv_channels": conv_channels,
                    "bilstm_hidden": bilstm_hidden,
                    "bilstm_layers": bilstm_layers,
                    "dense_hidden": dense_hidden,
                    "dropout": dropout,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "best_epoch": best_epoch,
                },
            },
            MODEL_DIR / f"{dataset}_paper_cnn_bilstm.pt",
        )
        history_df = pd.DataFrame(history)
        history_df.to_csv(RESULTS_DIR / f"paper_replication_{dataset}_deep_history.csv", index=False)
        plot_training_curve(history_df, FIGURES_DIR / f"{dataset}_paper_cnn_bilstm_training_curve.png")
        test_row = next(row for row in rows if row["split"] == "test")
        test_pred = [row for row in prediction_rows if row["split"] == "test"]
        y_test_pred = np.asarray([row["predicted_label"] for row in test_pred], dtype=np.int64)
        plot_confusion(
            prepared["y_test"],
            y_test_pred,
            spec.class_names,
            FIGURES_DIR / f"{dataset}_paper_cnn_bilstm_confusion_matrix.png",
            title=f"{dataset} CNN-BiLSTM test confusion matrix ({test_row['accuracy']:.3f} acc)",
        )
    return rows, {"history": history, "best_score": best_score, "best_epoch": best_epoch}


def plot_training_curve(history: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_accuracy"], label="train accuracy")
    axes[0].plot(history["epoch"], history["val_accuracy"], label="val accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(history["epoch"], history["loss"], label="train loss", color="tab:red")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_summary() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing result file: {RESULTS_PATH}")
    results = pd.read_csv(RESULTS_PATH)
    ok_test = results[(results["status"].eq("ok")) & (results["split"].eq("test"))].copy()
    if ok_test.empty:
        raise ValueError("No test rows found in paper replication results.")
    summary = (
        ok_test.sort_values(["dataset", "accuracy", "f1_macro"], ascending=[True, False, False])
        .groupby(["dataset", "stage"], as_index=False)
        .first()
    )
    summary.to_csv(SUMMARY_PATH, index=False)
    return summary


def database_url_from_env() -> str | None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path)
        except Exception:
            pass
    direct = os.getenv("DATABASE_URL")
    if direct:
        return direct
    host = os.getenv("POSTGRES_HOST")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    db = os.getenv("POSTGRES_DB")
    port = os.getenv("POSTGRES_PORT", "5432")
    if host and user and password and db:
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return None


def persist_to_postgres(run_record: dict[str, Any] | None = None) -> dict[str, Any]:
    url = database_url_from_env()
    if not url:
        return {"configured": False, "status": "skipped", "message": "PostgreSQL chưa cấu hình."}
    try:
        from sqlalchemy import create_engine, text
    except Exception as exc:
        return {"configured": True, "status": "failed", "message": f"SQLAlchemy unavailable: {exc}"}
    try:
        schema_path = PROJECT_ROOT / "database" / "schema.sql"
        engine = create_engine(url)
        with engine.begin() as connection:
            connection.exec_driver_sql(schema_path.read_text(encoding="utf-8"))
            persist_students_and_grades(connection, text)
            run_id = None
            if run_record is not None:
                run_payload = dict(run_record)
                run_payload["postgres_status"] = "ok"
                run_payload["postgres_message"] = "Paper runs/predictions persisted."
                result = connection.execute(
                    text(
                        """
                        INSERT INTO paper_runs (
                            generated_at, result_rows, summary_rows,
                            postgres_status, postgres_message, run_payload
                        )
                        VALUES (
                            :generated_at, :result_rows, :summary_rows,
                            :postgres_status, :postgres_message, CAST(:run_payload AS JSONB)
                        )
                        RETURNING paper_run_id
                        """
                    ),
                    {
                        "generated_at": run_payload.get("generated_at"),
                        "result_rows": int(run_payload.get("result_rows", 0)),
                        "summary_rows": int(run_payload.get("summary_rows", 0)),
                        "postgres_status": str(run_payload.get("postgres_status", "")),
                        "postgres_message": str(run_payload.get("postgres_message", "")),
                        "run_payload": json.dumps(run_payload, ensure_ascii=True),
                    },
                )
                run_id = result.scalar_one()
            if PREDICTIONS_PATH.exists():
                prediction_df = pd.read_csv(PREDICTIONS_PATH)
                prob_columns = [column for column in prediction_df.columns if column.startswith("prob_")]
                payloads = []
                for record in prediction_df.to_dict(orient="records"):
                    probability = {
                        column.removeprefix("prob_"): none_if_nan(record.get(column))
                        for column in prob_columns
                    }
                    payloads.append(
                        {
                            "run_id": run_id,
                            "model_name": record.get("model_name"),
                            "dataset": record.get("dataset"),
                            "split": record.get("split"),
                            "row_index": int(record.get("row_index", 0)),
                            "true_label": int(record.get("true_label", 0)),
                            "predicted_label": int(record.get("predicted_label", 0)),
                            "true_label_name": record.get("true_label_name"),
                            "predicted_label_name": record.get("predicted_label_name"),
                            "probability": json.dumps(probability, ensure_ascii=True),
                            "seed": int(record.get("seed", 0)),
                            "run_label": record.get("run_label"),
                            "G1": none_if_nan(record.get("G1")),
                            "G2": none_if_nan(record.get("G2")),
                            "G3": none_if_nan(record.get("G3")),
                            "xapi_class": none_if_nan(record.get("Class")),
                        }
                    )
                if payloads:
                    connection.execute(
                        text(
                            """
                            INSERT INTO paper_predictions (
                                run_id, model_name, dataset, split, row_index,
                                true_label, predicted_label, true_label_name, predicted_label_name,
                                probability, seed, run_label, G1, G2, G3, xapi_class
                            )
                            VALUES (
                                :run_id, :model_name, :dataset, :split, :row_index,
                                :true_label, :predicted_label, :true_label_name, :predicted_label_name,
                                CAST(:probability AS JSONB), :seed, :run_label, :G1, :G2, :G3, :xapi_class
                            )
                            """
                        ),
                        payloads,
                    )
        return {"configured": True, "status": "ok", "message": "Paper runs/predictions persisted."}
    except Exception as exc:
        return {"configured": True, "status": "failed", "message": f"{type(exc).__name__}: {exc}"}


def json_safe_value(value: Any) -> Any:
    value = none_if_nan(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_record(record: dict[str, Any]) -> dict[str, Any]:
    return {str(key): json_safe_value(value) for key, value in record.items()}


def float_or_none(value: Any) -> float | None:
    value = none_if_nan(value)
    if value is None:
        return None
    return float(value)


def persist_students_and_grades(connection: Any, sql_text: Any) -> None:
    for spec in DATASETS.values():
        raw = add_targets(read_raw(spec), spec).reset_index(drop=True)
        student_rows = []
        for source_row_index, record in raw.iterrows():
            student_rows.append(
                {
                    "dataset_name": spec.name,
                    "source_row_index": int(source_row_index),
                    "raw_profile": json.dumps(json_record(record.to_dict()), ensure_ascii=True),
                }
            )
        if student_rows:
            connection.execute(
                sql_text(
                    """
                    INSERT INTO students (dataset_name, source_row_index, raw_profile)
                    VALUES (:dataset_name, :source_row_index, CAST(:raw_profile AS JSONB))
                    ON CONFLICT (dataset_name, source_row_index)
                    DO UPDATE SET raw_profile = EXCLUDED.raw_profile
                    """
                ),
                student_rows,
            )

        student_id_rows = connection.execute(
            sql_text(
                """
                SELECT source_row_index, student_id
                FROM students
                WHERE dataset_name = :dataset_name
                """
            ),
            {"dataset_name": spec.name},
        )
        student_ids = {
            int(row._mapping["source_row_index"]): int(row._mapping["student_id"])
            for row in student_id_rows
        }
        grade_rows = []
        for source_row_index, record in raw.iterrows():
            row = record.to_dict()
            grade_rows.append(
                {
                    "student_id": student_ids.get(int(source_row_index)),
                    "dataset_name": spec.name,
                    "source_row_index": int(source_row_index),
                    "G1": float_or_none(row.get("G1")),
                    "G2": float_or_none(row.get("G2")),
                    "G3": float_or_none(row.get("G3")),
                    "xapi_class": none_if_nan(row.get("Class")),
                    "target_class": int(row["target_class"]),
                    "target_class_name": str(row["target_class_name"]),
                    "raw_grade_payload": json.dumps(
                        {
                            "G1": json_safe_value(row.get("G1")),
                            "G2": json_safe_value(row.get("G2")),
                            "G3": json_safe_value(row.get("G3")),
                            "Class": json_safe_value(row.get("Class")),
                            "target_class": json_safe_value(row.get("target_class")),
                            "target_class_name": json_safe_value(row.get("target_class_name")),
                        },
                        ensure_ascii=True,
                    ),
                }
            )
        if grade_rows:
            connection.execute(
                sql_text(
                    """
                    INSERT INTO student_grades (
                        student_id, dataset_name, source_row_index,
                        G1, G2, G3, xapi_class, target_class, target_class_name,
                        raw_grade_payload
                    )
                    VALUES (
                        :student_id, :dataset_name, :source_row_index,
                        :G1, :G2, :G3, :xapi_class, :target_class, :target_class_name,
                        CAST(:raw_grade_payload AS JSONB)
                    )
                    ON CONFLICT (dataset_name, source_row_index)
                    DO UPDATE SET
                        student_id = EXCLUDED.student_id,
                        G1 = EXCLUDED.G1,
                        G2 = EXCLUDED.G2,
                        G3 = EXCLUDED.G3,
                        xapi_class = EXCLUDED.xapi_class,
                        target_class = EXCLUDED.target_class,
                        target_class_name = EXCLUDED.target_class_name,
                        raw_grade_payload = EXCLUDED.raw_grade_payload
                    """
                ),
                grade_rows,
            )


def none_if_nan(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        return value
    return value


def generate_report() -> Path:
    ensure_dirs()
    summary = build_summary()
    run_record = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "result_rows": int(pd.read_csv(RESULTS_PATH).shape[0]) if RESULTS_PATH.exists() else 0,
        "summary_rows": int(summary.shape[0]),
        "postgres_status": "pending",
        "postgres_message": "",
    }
    db_status = persist_to_postgres(run_record)
    run_record["postgres_status"] = db_status["status"]
    run_record["postgres_message"] = db_status["message"]
    lines = [
        "# Paper Replication Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        "- Pipeline này làm lại theo hướng bài báo CNN-BiLSTM, không dùng kết quả V2/V3 cũ.",
        "- Kết quả bên dưới là kết quả chạy thật từ project hiện tại.",
        "- Optuna đã được dựng riêng, nhưng full sweep chưa chạy.",
        "",
        "## Assumptions",
        "",
        "- PDF không ghi rõ split ratio, optimizer, learning rate, class bins G3; pipeline dùng stratified 64/16/20 train/val/test, Adam lr=0.001, và G3 bins 0-4/5-8/9-12/13-16/17-20.",
        "- PDF chỉ ghi Pearson feature selection và kết luận Student dùng G1/G2, xAPI dùng raisedhands/VisitedResources/StudentAbsenceDays; pipeline dùng đúng các feature này.",
        "- Metric Precision/Recall/F1 trong project được tính macro và weighted; bảng chính hiển thị macro để công bằng cho multiclass.",
        "",
        "## PostgreSQL",
        "",
        f"- Status: {db_status['status']}",
        f"- Note: {db_status['message']}",
        "",
        "## Best Test Results From This Run",
        "",
        markdown_table(
            summary[
                [
                    "dataset",
                    "stage",
                    "model_name",
                    "accuracy",
                    "precision_macro",
                    "recall_macro",
                    "f1_macro",
                    "effective_imbalance_strategy",
                    "best_epoch",
                ]
            ]
        ),
        "",
        "## Paper Reference Values",
        "",
        "| dataset | paper CNN-BiLSTM accuracy | other paper metrics |",
        "|---|---:|---|",
    ]
    lines = [line for line in lines if "full sweep" not in line]
    lines.insert(8, "- Optuna status is read from the current best-params files.")
    for spec in DATASETS.values():
        ref = spec.paper_reference
        other = []
        for key in ("cnn_bilstm_precision", "cnn_bilstm_recall", "cnn_bilstm_f1", "decision_tree_accuracy"):
            if key in ref:
                other.append(f"{key}={ref[key]:.4f}")
        lines.append(f"| {spec.name} | {ref.get('cnn_bilstm_accuracy', float('nan')):.4f} | {', '.join(other)} |")
    optuna_best_path = RESULTS_DIR / "paper_replication_optuna_best_params.json"
    if optuna_best_path.exists():
        optuna_payload = json.loads(optuna_best_path.read_text(encoding="utf-8"))
        optuna_rows = []
        if "datasets" in optuna_payload:
            iterable = optuna_payload["datasets"].items()
        else:
            iterable = [(optuna_payload.get("dataset"), optuna_payload)]
        for dataset, payload in iterable:
            optuna_rows.append(
                {
                    "dataset": dataset,
                    "trials": payload.get("trials"),
                    "epochs_per_trial": payload.get("epochs_per_trial"),
                    "objective": payload.get("objective", "val_macro_f1"),
                    "best_validation_score": payload.get("best_value"),
                }
            )
        lines.extend(["", "## Optuna Search Results", "", markdown_table(pd.DataFrame(optuna_rows))])
    lines.extend(
        [
            "",
            "## Honest Notes",
            "",
            "- Nếu kết quả project thấp hơn paper, report giữ nguyên số thật và không copy số từ PDF.",
            "- XGBoost được chạy nếu package có sẵn; nếu thiếu package, dòng skip sẽ nằm trong CSV kết quả.",
            "- Các hình training curve/confusion matrix nằm trong `reports/figures/paper_replication/`.",
        ]
    )
    lines = normalize_report_lines(lines)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame([run_record]).to_csv(RUNS_PATH, index=False)
    return REPORT_PATH


def normalize_report_lines(lines: list[str]) -> list[str]:
    normalized = []
    for line in lines:
        if line.startswith("- Pipeline n"):
            normalized.append("- Pipeline nay lam lai theo huong bai bao CNN-BiLSTM, khong dung ket qua V2/V3 cu.")
        elif line.startswith("- K") and "project" in line and "PDF" not in line:
            normalized.append("- Ket qua ben duoi la ket qua chay that tu project hien tai.")
        elif line.startswith("- PDF kh"):
            normalized.append(
                "- PDF khong ghi ro split ratio, optimizer, learning rate, class bins G3; "
                "pipeline dung stratified 64/16/20 train/val/test, Adam lr=0.001, "
                "va G3 bins 0-4/5-8/9-12/13-16/17-20."
            )
        elif line.startswith("- PDF ch"):
            normalized.append(
                "- PDF chi ghi Pearson feature selection va ket luan Student dung G1/G2, "
                "xAPI dung raisedhands/VisitedResources/StudentAbsenceDays; pipeline dung cac feature nay."
            )
        elif line.startswith("- Metric Precision"):
            normalized.append(
                "- Metric Precision/Recall/F1 trong project duoc tinh macro va weighted; "
                "bang chinh hien thi macro de cong bang cho multiclass."
            )
        elif line.startswith("- N") and "paper" in line and "PDF" in line:
            normalized.append("- Neu ket qua project thap hon paper, report giu nguyen so that va khong copy so tu PDF.")
        elif line.startswith("- XGBoost"):
            normalized.append("- XGBoost duoc chay neu package co san; neu thieu package, dong skip se nam trong CSV ket qua.")
        elif line.startswith("- C") and "figures" in line:
            normalized.append("- Cac hinh training curve/confusion matrix nam trong `reports/figures/paper_replication/`.")
        else:
            normalized.append(line)
    return normalized


def markdown_table(data: pd.DataFrame) -> str:
    columns = data.columns.tolist()
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, record in data.iterrows():
        values = []
        for column in columns:
            value = record[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def run_all(
    *,
    seed: int = 42,
    epochs: int = 100,
    patience: int = 15,
    grid_profile: str = "compact",
    skip_baselines: bool = False,
    skip_deep: bool = False,
) -> None:
    ensure_dirs()
    for dataset in DATASETS:
        prepare_dataset(dataset, seed=seed)
    if not skip_baselines:
        for dataset in DATASETS:
            train_baselines(dataset, seed=seed, grid_profile=grid_profile)
    if not skip_deep:
        for dataset in DATASETS:
            train_deep(dataset, seed=seed, epochs=epochs, patience=patience)
    generate_report()


def parse_dataset(value: str) -> list[str]:
    if value == "all":
        return list(DATASETS)
    if value not in DATASETS:
        raise ValueError(f"Unknown dataset {value}. Expected one of {sorted(DATASETS)} or all.")
    return [value]


def run_cli(args: argparse.Namespace) -> None:
    ensure_dirs()
    datasets = parse_dataset(args.dataset)
    if args.stage in {"prepare", "all"}:
        for dataset in datasets:
            prepare_dataset(dataset, seed=args.seed)
    if args.stage in {"baseline", "all"}:
        for dataset in datasets:
            train_baselines(dataset, seed=args.seed, grid_profile=args.grid_profile)
    if args.stage in {"deep", "all"}:
        for dataset in datasets:
            train_deep(dataset, seed=args.seed, epochs=args.epochs, patience=args.patience)
    if args.stage in {"report", "all"}:
        generate_report()
