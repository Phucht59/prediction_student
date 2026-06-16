from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, r2_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from src.config import DATASETS, DEFAULT_SEED, RAW_DIR, REPORTS_DIR
from src.data_pipeline import (
    DataPreprocessor,
    FeatureSelector,
    apply_feature_engineering,
    create_and_save_locked_test,
    load_splits,
)
from src.utils import setup_logger

logger = setup_logger("technical_experiments")

STUDENT_DATASETS = ("student-mat", "student-por")
SCENARIOS = {
    "early": {"drop": ["G1", "G2"], "sequence": []},
    "midterm": {"drop": ["G2"], "sequence": ["G1"]},
    "late": {"drop": [], "sequence": ["G1", "G2"]},
}
IMBALANCE_STRATEGIES = (
    "none",
    "class_weight",
    "smotenc",
    "random_oversampling",
    "focal_loss",
    "smotenc_focal_loss",
)
POINT_CENTERS_3CLASS = np.array([4.5, 11.5, 17.0], dtype=float)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = DEFAULT_SEED
    cv_folds: int = 5
    target_mode: str = "3class"
    use_feature_selection: bool = True
    smote_ratio: float = 1.0
    resampling_k_neighbors: int = 5


def ensure_technical_report_dirs() -> dict[str, Path]:
    dirs = {
        "scenarios": REPORTS_DIR / "scenarios",
        "baselines": REPORTS_DIR / "baselines",
        "imbalance": REPORTS_DIR / "imbalance",
        "ablation": REPORTS_DIR / "ablation",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_config(path: Path, config: ExperimentConfig, extra: dict | None = None) -> None:
    payload = asdict(config)
    if extra:
        payload.update(extra)
    save_json(path, payload)


def load_or_create_student_splits(dataset_name: str, target_mode: str = "3class") -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset_name not in STUDENT_DATASETS:
        raise ValueError("Technical Student experiments intentionally exclude student-combine and non-Student datasets.")
    try:
        return load_splits(dataset_name, target_mode)
    except FileNotFoundError:
        spec = DATASETS[dataset_name]
        raw = pd.read_csv(RAW_DIR / spec.raw_file, sep=spec.csv_sep)
        create_and_save_locked_test(raw, dataset_name, target_mode)
        return load_splits(dataset_name, target_mode)


def apply_student_scenario(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Expected one of {sorted(SCENARIOS)}.")
    out = df.copy()
    drop_cols = [col for col in SCENARIOS[scenario]["drop"] if col in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    # Feature engineering must run after dropping unavailable grades so derived grade
    # features cannot smuggle G1/G2 into early or G2 into midterm.
    return apply_feature_engineering(out, "student")


def scenario_sequence_columns(scenario: str) -> list[str]:
    return list(SCENARIOS[scenario]["sequence"])


class TechnicalStudentDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        numerical_cols: list[str],
        categorical_cols: list[str],
        sequence_cols: list[str],
    ):
        self.y = df[target_col].astype(int).to_numpy() if target_col in df.columns else np.zeros(len(df), dtype=int)
        self.reg_label = (
            df["G3_raw"].astype(float).to_numpy(dtype=np.float32)
            if "G3_raw" in df.columns
            else POINT_CENTERS_3CLASS[self.y].astype(np.float32)
        )
        self.seq_cols = [col for col in sequence_cols if col in df.columns]
        if self.seq_cols:
            self.seq_x = df[self.seq_cols].to_numpy(dtype=np.float32)[..., np.newaxis]
        else:
            self.seq_x = np.zeros((len(df), 1, 1), dtype=np.float32)

        self.num_cols = [
            col for col in numerical_cols
            if col in df.columns and col not in self.seq_cols and col != "G3_raw"
        ]
        self.cat_cols = [col for col in categorical_cols if col in df.columns and col != "G3_raw"]
        self.num_x = df[self.num_cols].to_numpy(dtype=np.float32) if self.num_cols else np.zeros((len(df), 1), dtype=np.float32)
        self.cat_x = df[self.cat_cols].to_numpy(dtype=int) if self.cat_cols else np.zeros((len(df), 1), dtype=int)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.seq_x[idx], dtype=torch.float32),
            torch.tensor(self.num_x[idx], dtype=torch.float32),
            torch.tensor(self.cat_x[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.long),
            idx,
            torch.tensor(self.reg_label[idx], dtype=torch.float32),
        )


@dataclass
class PreparedFold:
    train: pd.DataFrame
    validation: pd.DataFrame
    preprocessor: DataPreprocessor
    selector: FeatureSelector
    sequence_cols: list[str]


def oversampling_method_for_strategy(strategy: str) -> str:
    if strategy in {"smotenc", "smotenc_focal_loss"}:
        return "smotenc"
    if strategy == "random_oversampling":
        return "random_oversampling"
    return "none"


def prepare_fold(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    target_col: str,
    scenario: str,
    strategy: str,
    config: ExperimentConfig,
) -> PreparedFold:
    train_scenario = apply_student_scenario(train_df, scenario)
    validation_scenario = apply_student_scenario(validation_df, scenario)
    preprocessor = DataPreprocessor(
        target_col=target_col,
        oversample_method=oversampling_method_for_strategy(strategy),
        smote_ratio=config.smote_ratio,
        resampling_k_neighbors=config.resampling_k_neighbors,
    )
    train_prep = preprocessor.fit_transform(train_scenario, apply_oversampling=False)
    validation_prep = preprocessor.transform(validation_scenario)
    sequence_cols = scenario_sequence_columns(scenario)
    selector = FeatureSelector(
        target_col=target_col,
        use_feature_selection=config.use_feature_selection,
        required_features=sequence_cols,
    )
    train_selected = selector.fit_transform(
        train_prep,
        preprocessor.numerical_cols,
        preprocessor.categorical_cols,
    )
    validation_selected = selector.transform(validation_prep)
    if oversampling_method_for_strategy(strategy) != "none":
        train_selected = preprocessor.apply_oversampling(train_selected)
    return PreparedFold(train_selected, validation_selected, preprocessor, selector, sequence_cols)


def transform_with_prepared(
    df: pd.DataFrame,
    prepared: PreparedFold,
    target_col: str,
    scenario: str,
) -> pd.DataFrame:
    scenario_df = apply_student_scenario(df, scenario)
    prepped = prepared.preprocessor.transform(scenario_df)
    return prepared.selector.transform(prepped)


def stratified_folds(labels: Iterable[int], config: ExperimentConfig) -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)


def class_points(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    return POINT_CENTERS_3CLASS[np.clip(labels, 0, len(POINT_CENTERS_3CLASS) - 1)]


def compute_required_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: np.ndarray | None = None,
    y_reg_true: Iterable[float] | None = None,
    y_reg_pred: Iterable[float] | None = None,
    regression_source: str = "mapped_class",
) -> dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_pred_arr = np.asarray(list(y_pred), dtype=int)
    if y_reg_true is None:
        reg_true = class_points(y_true_arr)
    else:
        reg_true = np.asarray(list(y_reg_true), dtype=float)
    if regression_source == "regression_head" and y_reg_pred is not None:
        reg_pred = np.asarray(list(y_reg_pred), dtype=float)
    else:
        reg_pred = class_points(y_pred_arr)

    rmse = float(math.sqrt(np.mean((reg_true - reg_pred) ** 2)))
    r2 = float(r2_score(reg_true, reg_pred)) if len(np.unique(reg_true)) > 1 else 0.0
    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_precision": float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "recall_low": float(recall_score(y_true_arr, y_pred_arr, labels=[0], average="macro", zero_division=0)),
        "f1_low": float(f1_score(y_true_arr, y_pred_arr, labels=[0], average="macro", zero_division=0)),
        "rmse": rmse,
        "r2": r2,
    }
    if y_score is not None:
        metrics["mean_confidence"] = float(np.max(y_score, axis=1).mean())
    if y_reg_pred is not None:
        head_pred = np.asarray(list(y_reg_pred), dtype=float)
        metrics["regression_head_rmse"] = float(math.sqrt(np.mean((reg_true - head_pred) ** 2)))
        metrics["regression_head_r2"] = float(r2_score(reg_true, head_pred)) if len(np.unique(reg_true)) > 1 else 0.0
    return metrics


def summarize_cv(rows: list[dict]) -> dict[str, float]:
    metric_keys = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "recall_low",
        "f1_low",
        "rmse",
        "r2",
    ]
    summary: dict[str, float] = {}
    for key in metric_keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            summary[f"cv_{key}_mean"] = float(np.mean(values))
            summary[f"cv_{key}_std"] = float(np.std(values, ddof=0))
    return summary


def tune_low_threshold_from_oof(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float | list[float]]:
    best = {"threshold_low": 0.0, "score": -1.0, "macro_f1": 0.0, "recall_low": 0.0}
    for threshold in np.linspace(0.0, 0.95, 40):
        preds = predict_with_low_threshold(probabilities, float(threshold))
        metrics = compute_required_metrics(y_true, preds)
        score = 0.65 * metrics["recall_low"] + 0.35 * metrics["macro_f1"]
        if score > best["score"]:
            best = {
                "threshold_low": float(threshold),
                "score": float(score),
                "macro_f1": metrics["macro_f1"],
                "recall_low": metrics["recall_low"],
            }
    return best


def predict_with_low_threshold(probabilities: np.ndarray, threshold_low: float) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    fallback = np.argmax(probs[:, 1:], axis=1) + 1
    return np.where(probs[:, 0] >= threshold_low, 0, fallback)
