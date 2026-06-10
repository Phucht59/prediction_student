from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.paper_replication.advanced_experiments import (
    ExactPaperCNNBiLSTM,
    ImprovedCNNBiLSTM,
    PAPER_BENCHMARKS,
    apply_binning,
    model_dataset_type,
    set_seed,
)
from src.paper_replication.pipeline import DATASETS, PROJECT_ROOT, XAPI_CLASS_MAPPING, build_preprocessor, dense_array, read_raw, resolve_columns


RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
V6_RESULTS_PATH = RESULTS_DIR / "v6_case_sweep_results.json"
V6_REPORT_PATH = REPORTS_DIR / "v6_case_sweep_report.md"

STUDENT_DATASETS = {"student-mat", "student-por"}
ALL_DATASETS = ["student-mat", "student-por", "xapi"]
DEFAULT_SEEDS = [42, 123, 456, 789, 2024, 314, 999]
BINNING_KEY = "H1_portuguese_scale"
CLASS_NAMES = {
    "student-mat": ["0", "1", "2", "3", "4"],
    "student-por": ["0", "1", "2", "3", "4"],
    "xapi": ["L", "M", "H"],
}
DEFAULT_OVERSAMPLING = {"student-mat": "smote", "student-por": "none", "xapi": "adasyn"}
PAPER_REFERENCES = {
    **PAPER_BENCHMARKS,
}
V1_REFERENCES = {
    "student-mat": {"accuracy": 0.8101, "f1_macro": 0.8044},
    "student-por": {"accuracy": 0.7385, "f1_macro": 0.7368},
    "xapi": {"accuracy": 0.7396, "f1_macro": 0.7474},
}


@dataclass(frozen=True)
class SweepCase:
    name: str
    feature_set: str
    oversampling: str
    conv_filters: int
    kernel_size: int
    bilstm_hidden: int
    dense_hidden: int
    dropout: float
    lr: float
    batch_size: int
    label_smoothing: float = 0.0
    loss: str = "ce"
    datasets: tuple[str, ...] = tuple(ALL_DATASETS)
    binning_key: str = BINNING_KEY
    split_policy: str = "clean_train_only"
    selection_metric: str = "f1_macro"
    model_variant: str = "exact"
    num_conv_layers: int = 4
    num_bilstm_layers: int = 1
    use_batchnorm: bool = False
    use_residual: bool = False
    use_attention: bool = False
    weight_decay: float = 0.0
    patience: int = 0
    test_size: float = 0.20


CASES: list[SweepCase] = [
    SweepCase("paper_base_default", "paper", "default", 64, 3, 64, 64, 0.20, 1e-3, 32),
    SweepCase("paper_kernel5_h128", "paper", "default", 64, 5, 128, 128, 0.20, 1e-3, 32),
    SweepCase("paper_kernel7_h128_smooth", "paper", "default", 64, 7, 128, 128, 0.35, 3e-4, 16, 0.10),
    SweepCase("paper_wide128_dropout01", "paper", "default", 128, 3, 128, 128, 0.10, 1e-3, 16),
    SweepCase("paper_low_lr_dropout035", "paper", "default", 64, 5, 128, 64, 0.35, 3e-4, 32),
    SweepCase("top3_base_default", "top3", "default", 64, 3, 64, 64, 0.20, 1e-3, 32, datasets=("student-mat", "student-por")),
    SweepCase("top5_base_default", "top5", "default", 64, 3, 64, 64, 0.20, 1e-3, 32),
    SweepCase("top5_wide_smooth", "top5", "default", 128, 5, 128, 128, 0.25, 3e-4, 16, 0.10),
    SweepCase("student_top5_classw", "top5", "default", 64, 3, 64, 64, 0.20, 1e-3, 32, 0.0, "class_weight", ("student-mat", "student-por")),
    SweepCase("student_top5_wide_smooth_classw", "top5", "default", 128, 5, 128, 128, 0.25, 3e-4, 16, 0.05, "class_weight", ("student-mat", "student-por")),
    SweepCase("student_academic10_classw", "academic10", "default", 64, 3, 64, 64, 0.20, 1e-3, 32, 0.0, "class_weight", ("student-mat", "student-por")),
    SweepCase("student_academic10_wide_smooth", "academic10", "default", 128, 5, 128, 128, 0.25, 3e-4, 16, 0.05, "class_weight", ("student-mat", "student-por")),
    SweepCase("student_full_classw", "full", "default", 64, 3, 64, 64, 0.20, 1e-3, 32, 0.0, "class_weight", ("student-mat", "student-por")),
    SweepCase("student_full_wide_smooth", "full", "default", 128, 5, 128, 128, 0.25, 3e-4, 16, 0.05, "class_weight", ("student-mat", "student-por")),
    SweepCase("xapi_paper_none", "paper", "none", 64, 3, 64, 64, 0.20, 1e-3, 32, datasets=("xapi",)),
    SweepCase("xapi_paper_smote", "paper", "smote", 64, 3, 64, 64, 0.20, 1e-3, 32, datasets=("xapi",)),
    SweepCase("xapi_top4_adasyn", "top4", "adasyn", 64, 3, 64, 64, 0.20, 1e-3, 32, datasets=("xapi",)),
    SweepCase("xapi_top4_smote_wide", "top4", "smote", 128, 5, 128, 128, 0.20, 5e-4, 16, datasets=("xapi",)),
    SweepCase("xapi_top5_none_classw", "top5", "none", 64, 3, 64, 64, 0.20, 1e-3, 32, 0.0, "class_weight", ("xapi",)),
    SweepCase("xapi_top5_adasyn_smooth", "top5", "adasyn", 128, 5, 128, 128, 0.25, 3e-4, 16, 0.10, datasets=("xapi",)),
    SweepCase("xapi_top5_classw_dropout01", "top5", "none", 64, 3, 64, 64, 0.10, 1e-3, 32, 0.0, "class_weight", ("xapi",)),
    SweepCase("xapi_top5_classw_kernel5_h128", "top5", "none", 64, 5, 128, 128, 0.20, 1e-3, 32, 0.0, "class_weight", ("xapi",)),
    SweepCase("xapi_top5_classw_wide128", "top5", "none", 128, 3, 128, 128, 0.10, 1e-3, 16, 0.0, "class_weight", ("xapi",)),
    SweepCase("xapi_top5_classw_lr0003", "top5", "none", 64, 3, 64, 64, 0.20, 3e-4, 32, 0.0, "class_weight", ("xapi",)),
    SweepCase("xapi_top5_classw_smooth005", "top5", "none", 64, 3, 64, 64, 0.20, 1e-3, 32, 0.05, "class_weight", ("xapi",)),
    SweepCase("xapi_numeric5_classw", "numeric5", "none", 64, 3, 64, 64, 0.20, 1e-3, 32, 0.0, "class_weight", ("xapi",)),
    SweepCase("xapi_numeric5_classw_smooth005", "numeric5", "none", 64, 3, 64, 64, 0.20, 1e-3, 32, 0.05, "class_weight", ("xapi",)),
    SweepCase("xapi_behavior8_classw", "behavior8", "none", 64, 3, 64, 64, 0.20, 1e-3, 32, 0.0, "class_weight", ("xapi",)),
    SweepCase("xapi_behavior8_wide_smooth", "behavior8", "none", 128, 5, 128, 128, 0.25, 3e-4, 16, 0.05, "class_weight", ("xapi",)),
    SweepCase("xapi_full_classw", "full", "none", 64, 3, 64, 64, 0.20, 1e-3, 32, 0.0, "class_weight", ("xapi",)),
    SweepCase("xapi_full_wide_smooth", "full", "none", 128, 5, 128, 128, 0.25, 3e-4, 16, 0.05, "class_weight", ("xapi",)),
    SweepCase("v12_h1_acc_select", "paper", "default", 64, 3, 64, 64, 0.20, 1e-3, 32, datasets=("student-mat", "student-por"), selection_metric="accuracy"),
    SweepCase(
        "v12_h3_acc_select",
        "paper",
        "default",
        64,
        3,
        64,
        64,
        0.20,
        1e-3,
        32,
        datasets=("student-mat", "student-por"),
        binning_key="H3_paper_percentile",
        selection_metric="accuracy",
    ),
    SweepCase(
        "v12_h1_global_scaler_acc",
        "paper",
        "default",
        64,
        3,
        64,
        64,
        0.20,
        1e-3,
        32,
        datasets=("student-mat", "student-por"),
        split_policy="global_scaler",
        selection_metric="accuracy",
    ),
    SweepCase(
        "v12_h3_global_scaler_acc",
        "paper",
        "default",
        64,
        3,
        64,
        64,
        0.20,
        1e-3,
        32,
        datasets=("student-mat", "student-por"),
        binning_key="H3_paper_percentile",
        split_policy="global_scaler",
        selection_metric="accuracy",
    ),
    SweepCase(
        "v12_h1_presplit_smote_acc",
        "paper",
        "smote",
        64,
        3,
        64,
        64,
        0.20,
        1e-3,
        32,
        datasets=("student-mat", "student-por"),
        split_policy="presplit_oversample_global_scaler",
        selection_metric="accuracy",
    ),
    SweepCase(
        "v12_h3_presplit_smote_acc",
        "paper",
        "smote",
        64,
        3,
        64,
        64,
        0.20,
        1e-3,
        32,
        datasets=("student-mat", "student-por"),
        binning_key="H3_paper_percentile",
        split_policy="presplit_oversample_global_scaler",
        selection_metric="accuracy",
    ),
    SweepCase(
        "v13_engineered_h1_acc",
        "paper_engineered",
        "default",
        64,
        3,
        64,
        64,
        0.20,
        1e-3,
        32,
        datasets=("student-mat", "student-por"),
        selection_metric="accuracy",
    ),
    SweepCase(
        "v13_engineered_h1_wide_acc",
        "paper_engineered",
        "default",
        128,
        3,
        128,
        128,
        0.10,
        1e-3,
        16,
        datasets=("student-mat", "student-por"),
        selection_metric="accuracy",
    ),
    SweepCase(
        "v13_engineered_h1_classw",
        "paper_engineered",
        "default",
        64,
        3,
        64,
        64,
        0.20,
        1e-3,
        32,
        0.0,
        "class_weight",
        ("student-mat", "student-por"),
        selection_metric="accuracy",
    ),
    SweepCase(
        "v13_engineered_h1_presplit_smote",
        "paper_engineered",
        "smote",
        64,
        3,
        64,
        64,
        0.20,
        1e-3,
        32,
        datasets=("student-mat", "student-por"),
        split_policy="presplit_oversample_global_scaler",
        selection_metric="accuracy",
    ),
    SweepCase(
        "v14_engineered_improved_attn",
        "paper_engineered",
        "default",
        64,
        3,
        128,
        128,
        0.25,
        7e-4,
        32,
        datasets=("student-mat", "student-por"),
        selection_metric="accuracy",
        model_variant="improved",
        num_conv_layers=4,
        num_bilstm_layers=1,
        use_batchnorm=True,
        use_residual=True,
        use_attention=True,
        weight_decay=1e-4,
        patience=25,
    ),
    SweepCase(
        "v14_engineered_improved_wide",
        "paper_engineered",
        "default",
        128,
        3,
        128,
        128,
        0.15,
        7e-4,
        16,
        datasets=("student-mat", "student-por"),
        selection_metric="accuracy",
        model_variant="improved",
        num_conv_layers=4,
        num_bilstm_layers=1,
        use_batchnorm=True,
        use_residual=True,
        use_attention=True,
        weight_decay=1e-4,
        patience=25,
    ),
    SweepCase(
        "v14_top5_improved_classw",
        "top5",
        "default",
        64,
        3,
        128,
        128,
        0.20,
        7e-4,
        32,
        0.0,
        "class_weight",
        ("student-mat", "student-por"),
        selection_metric="accuracy",
        model_variant="improved",
        num_conv_layers=4,
        num_bilstm_layers=1,
        use_batchnorm=True,
        use_residual=True,
        use_attention=True,
        weight_decay=1e-4,
        patience=25,
    ),
    SweepCase(
        "v14_xapi_full_improved",
        "full",
        "none",
        128,
        5,
        128,
        128,
        0.20,
        5e-4,
        16,
        0.05,
        "class_weight",
        ("xapi",),
        selection_metric="accuracy",
        model_variant="improved",
        num_conv_layers=4,
        num_bilstm_layers=1,
        use_batchnorm=True,
        use_residual=True,
        use_attention=False,
        weight_decay=1e-4,
        patience=25,
    ),
    SweepCase(
        "v14_xapi_behavior8_improved",
        "behavior8",
        "none",
        128,
        5,
        128,
        128,
        0.20,
        5e-4,
        16,
        0.05,
        "class_weight",
        ("xapi",),
        selection_metric="accuracy",
        model_variant="improved",
        num_conv_layers=4,
        num_bilstm_layers=1,
        use_batchnorm=True,
        use_residual=True,
        use_attention=False,
        weight_decay=1e-4,
        patience=25,
    ),
    SweepCase(
        "v16_full_engineered_exact",
        "full_engineered",
        "default",
        64,
        3,
        64,
        64,
        0.20,
        1e-3,
        32,
        datasets=("student-mat", "student-por"),
        selection_metric="accuracy",
        patience=25,
    ),
    SweepCase(
        "v16_full_engineered_wide_smooth",
        "full_engineered",
        "default",
        128,
        5,
        128,
        128,
        0.25,
        3e-4,
        16,
        0.05,
        "class_weight",
        ("student-mat", "student-por"),
        selection_metric="accuracy",
        patience=25,
    ),
    SweepCase(
        "v16_full_engineered_improved",
        "full_engineered",
        "default",
        64,
        3,
        128,
        128,
        0.25,
        7e-4,
        32,
        datasets=("student-mat", "student-por"),
        selection_metric="accuracy",
        model_variant="improved",
        num_conv_layers=4,
        num_bilstm_layers=1,
        use_batchnorm=True,
        use_residual=True,
        use_attention=True,
        weight_decay=1e-4,
        patience=25,
    ),
    SweepCase(
        "v17_grade_seq_paper",
        "paper",
        "default",
        64,
        3,
        64,
        64,
        0.20,
        1e-3,
        32,
        datasets=("student-mat", "student-por"),
        selection_metric="accuracy",
        model_variant="grade_seq",
        patience=25,
    ),
    SweepCase(
        "v17_grade_seq_engineered",
        "paper_engineered",
        "default",
        64,
        3,
        64,
        64,
        0.20,
        1e-3,
        32,
        datasets=("student-mat", "student-por"),
        selection_metric="accuracy",
        model_variant="grade_seq",
        patience=25,
    ),
    SweepCase(
        "v17_grade_seq_engineered_wide",
        "paper_engineered",
        "default",
        128,
        5,
        128,
        128,
        0.20,
        5e-4,
        16,
        0.05,
        "ce",
        ("student-mat", "student-por"),
        selection_metric="accuracy",
        model_variant="grade_seq",
        weight_decay=1e-4,
        patience=25,
    ),
]


@dataclass
class PreparedSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    feature_columns: list[str]
    processed_feature_names: list[str]
    n_features: int
    n_classes: int
    pre_split_oversampling_effective: str | None = None


@dataclass
class TrainResult:
    row: dict[str, Any]
    state_dict: dict[str, torch.Tensor]


def ensure_dirs() -> None:
    for path in (RESULTS_DIR, MODELS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def artifact_paths(output_tag: str) -> tuple[Path, Path, str]:
    tag = output_tag.strip()
    if not tag:
        return V6_RESULTS_PATH, V6_REPORT_PATH, "v6_best"
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in tag.lower()).strip("_")
    if not safe:
        return V6_RESULTS_PATH, V6_REPORT_PATH, "v6_best"
    return RESULTS_DIR / f"{safe}_results.json", REPORTS_DIR / f"{safe}_report.md", f"{safe}_best"


def checkpoint_path(output_tag: str) -> Path:
    result_path, _, _ = artifact_paths(output_tag)
    return result_path.with_name(result_path.stem.replace("_results", "") + "_checkpoint.json")


def seed_state_path(model_prefix: str, dataset: str, case_name: str, seed: int) -> Path:
    safe_case = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in case_name).strip("_")
    return MODELS_DIR / "seed_checkpoints" / f"{model_prefix}_{dataset}_{safe_case}_seed{int(seed)}.pt"


def row_key(dataset: str, case_name: str, seed: int) -> tuple[str, str, int]:
    return dataset, case_name, int(seed)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass
    return f"{float(value):.{digits}f}"


def get_feature_names(preprocessor: Any, fallback: list[str]) -> list[str]:
    try:
        return [str(value) for value in preprocessor.get_feature_names_out().tolist()]
    except Exception:
        return fallback


def resolve_one(raw: pd.DataFrame, alias: str) -> str:
    return resolve_columns(raw.columns.tolist(), [alias])[0]


def read_dataset(dataset: str) -> pd.DataFrame:
    return read_raw(DATASETS[dataset])


def model_type_for(dataset: str) -> str:
    return model_dataset_type(dataset)


def student_features(raw: pd.DataFrame, dataset: str, feature_set: str, binning_key: str = BINNING_KEY) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if feature_set == "paper":
        columns = ["G1", "G2"]
    elif feature_set == "paper_engineered":
        g1 = pd.to_numeric(raw["G1"], errors="raise").astype(float)
        g2 = pd.to_numeric(raw["G2"], errors="raise").astype(float)
        data = pd.DataFrame(
            {
                "G1": g1,
                "G2": g2,
                "G2_minus_G1": g2 - g1,
                "G_mean": (g1 + g2) / 2.0,
                "G_min": np.minimum(g1, g2),
                "G_max": np.maximum(g1, g2),
                "G1_bin": apply_binning(g1, binning_key, dataset).astype(float),
                "G2_bin": apply_binning(g2, binning_key, dataset).astype(float),
            },
            index=raw.index,
        )
        y = apply_binning(raw["G3"], binning_key, dataset).to_numpy(dtype=np.int64)
        return data, y, data.columns.tolist()
    elif feature_set == "full_engineered":
        g1 = pd.to_numeric(raw["G1"], errors="raise").astype(float)
        g2 = pd.to_numeric(raw["G2"], errors="raise").astype(float)
        data = raw[[column for column in raw.columns.tolist() if column != "G3"]].copy()
        engineered = pd.DataFrame(
            {
                "eng_G2_minus_G1": g2 - g1,
                "eng_G_mean": (g1 + g2) / 2.0,
                "eng_G_min": np.minimum(g1, g2),
                "eng_G_max": np.maximum(g1, g2),
                "eng_G1_bin": apply_binning(g1, binning_key, dataset).astype(float),
                "eng_G2_bin": apply_binning(g2, binning_key, dataset).astype(float),
            },
            index=raw.index,
        )
        data = pd.concat([data, engineered], axis=1)
        y = apply_binning(raw["G3"], binning_key, dataset).to_numpy(dtype=np.int64)
        return data, y, data.columns.tolist()
    elif feature_set == "top3":
        columns = ["G1", "G2", "failures"]
    elif feature_set == "top5":
        if dataset == "student-mat":
            columns = ["G2", "G1", "failures", "Medu", "schoolsup"]
        elif dataset == "student-por":
            columns = ["G2", "G1", "failures", "higher", "Medu"]
        else:
            columns = ["G2", "G1", "failures", "higher", "Medu", "schoolsup"]
    elif feature_set == "academic10":
        columns = ["G2", "G1", "failures", "absences", "studytime", "schoolsup", "higher", "Medu", "Fedu", "goout"]
    elif feature_set == "full":
        columns = [column for column in raw.columns.tolist() if column != "G3"]
    else:
        raise ValueError(f"Unsupported student feature set: {feature_set}")
    missing = [column for column in columns if column not in raw.columns]
    if missing:
        raise ValueError(f"{dataset} missing V6 feature columns: {missing}")
    y = apply_binning(raw["G3"], binning_key, dataset).to_numpy(dtype=np.int64)
    return raw[columns].copy(), y, columns


def xapi_features(raw: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    raised = resolve_one(raw, "raisedhands")
    visited = resolve_one(raw, "VisitedResources")
    absence = resolve_one(raw, "StudentAbsenceDays")
    announcements = resolve_one(raw, "AnnouncementsView")
    discussion = resolve_one(raw, "Discussion")
    parent_survey = resolve_one(raw, "ParentAnsweringSurvey")
    parent_satisfaction = resolve_one(raw, "ParentschoolSatisfaction")
    relation = resolve_one(raw, "Relation")
    y = raw["Class"].map(XAPI_CLASS_MAPPING).to_numpy(dtype=np.int64)
    if feature_set == "full":
        columns = [column for column in raw.columns.tolist() if column != "Class"]
        return raw[columns].copy(), y, columns
    absence_map = {"Under-7": 0.0, "Above-7": 1.0}
    absence_values = raw[absence].map(absence_map)
    if absence_values.isna().any():
        unknown = sorted(raw.loc[absence_values.isna(), absence].dropna().unique().tolist())
        raise ValueError(f"Unsupported StudentAbsenceDays values: {unknown}")
    data: dict[str, Any] = {
        "raisedhands": pd.to_numeric(raw[raised], errors="raise").astype(float),
        "VisitedResources": pd.to_numeric(raw[visited], errors="raise").astype(float),
        "StudentAbsenceDays": absence_values.astype(float),
    }
    if feature_set in {"top4", "top5"}:
        data["AnnouncementsView"] = pd.to_numeric(raw[announcements], errors="raise").astype(float)
    if feature_set == "top5":
        data["ParentAnsweringSurvey"] = raw[parent_survey].astype(str)
    if feature_set in {"numeric5", "behavior8"}:
        data["AnnouncementsView"] = pd.to_numeric(raw[announcements], errors="raise").astype(float)
        data["Discussion"] = pd.to_numeric(raw[discussion], errors="raise").astype(float)
    if feature_set == "behavior8":
        data["ParentAnsweringSurvey"] = raw[parent_survey].astype(str)
        data["ParentschoolSatisfaction"] = raw[parent_satisfaction].astype(str)
        data["Relation"] = raw[relation].astype(str)
    if feature_set not in {"paper", "top4", "top5", "numeric5", "behavior8"}:
        raise ValueError(f"Unsupported xAPI feature set: {feature_set}")
    frame = pd.DataFrame(data, index=raw.index)
    return frame, y, frame.columns.tolist()


def prepare_split(dataset: str, raw: pd.DataFrame, case: SweepCase, seed: int) -> PreparedSplit:
    if dataset in STUDENT_DATASETS:
        frame, y, feature_columns = student_features(raw, dataset, case.feature_set, case.binning_key)
        n_classes = 5
    else:
        frame, y, feature_columns = xapi_features(raw, case.feature_set)
        n_classes = 3
    if case.selection_metric not in {"f1_macro", "accuracy"}:
        raise ValueError(f"Unsupported selection metric: {case.selection_metric}")
    if case.split_policy == "presplit_oversample_global_scaler":
        preprocessor, _, _ = build_preprocessor(frame)
        X_all = dense_array(preprocessor.fit_transform(frame))
        feature_names = get_feature_names(preprocessor, feature_columns)
        X_res, y_res, effective = oversample_train(X_all, y, effective_oversampling(dataset, case), seed)
        indices = np.arange(len(X_res))
        train_idx, test_idx, y_train, y_test = train_test_split(
            indices,
            y_res,
            test_size=0.20,
            random_state=seed,
            stratify=y_res,
        )
        return PreparedSplit(
            X_train=np.asarray(X_res[train_idx], dtype=np.float32),
            y_train=np.asarray(y_train, dtype=np.int64),
            X_test=np.asarray(X_res[test_idx], dtype=np.float32),
            y_test=np.asarray(y_test, dtype=np.int64),
            train_indices=np.asarray(train_idx, dtype=np.int64),
            test_indices=np.asarray(test_idx, dtype=np.int64),
            feature_columns=feature_columns,
            processed_feature_names=feature_names,
            n_features=int(X_res.shape[1]),
            n_classes=n_classes,
            pre_split_oversampling_effective=effective,
        )
    indices = np.arange(len(frame))
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices,
        y,
        test_size=getattr(case, "test_size", 0.20),
        random_state=seed,
        stratify=y,
    )
    if case.split_policy == "clean_train_only":
        preprocessor, _, _ = build_preprocessor(frame.iloc[train_idx])
        X_train = dense_array(preprocessor.fit_transform(frame.iloc[train_idx]))
        X_test = dense_array(preprocessor.transform(frame.iloc[test_idx]))
    elif case.split_policy == "global_scaler":
        preprocessor, _, _ = build_preprocessor(frame)
        X_all = dense_array(preprocessor.fit_transform(frame))
        X_train = X_all[train_idx]
        X_test = X_all[test_idx]
    else:
        raise ValueError(f"Unsupported split policy: {case.split_policy}")
    feature_names = get_feature_names(preprocessor, feature_columns)
    return PreparedSplit(
        X_train=X_train,
        y_train=np.asarray(y_train, dtype=np.int64),
        X_test=X_test,
        y_test=np.asarray(y_test, dtype=np.int64),
        train_indices=np.asarray(train_idx, dtype=np.int64),
        test_indices=np.asarray(test_idx, dtype=np.int64),
        feature_columns=feature_columns,
        processed_feature_names=feature_names,
        n_features=int(X_train.shape[1]),
        n_classes=n_classes,
    )


def effective_oversampling(dataset: str, case: SweepCase) -> str:
    return DEFAULT_OVERSAMPLING[dataset] if case.oversampling == "default" else case.oversampling


def oversample_train(X: np.ndarray, y: np.ndarray, mode: str, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    if mode == "none":
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), "none"
    counts = np.bincount(np.asarray(y, dtype=np.int64))
    present = counts[counts > 0]
    if len(present) == 0:
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), mode
    if present.min() < 2:
        sampler = RandomOverSampler(random_state=seed)
    else:
        k = int(min(5, present.min() - 1))
        if mode == "smote":
            sampler = SMOTE(random_state=seed, k_neighbors=k)
        elif mode == "adasyn":
            sampler = ADASYN(random_state=seed, n_neighbors=k)
        else:
            raise ValueError(f"Unsupported oversampling mode: {mode}")
    try:
        X_res, y_res = sampler.fit_resample(X, y)
        return np.asarray(X_res, dtype=np.float32), np.asarray(y_res, dtype=np.int64), mode
    except Exception as exc:
        fallback = RandomOverSampler(random_state=seed)
        X_res, y_res = fallback.fit_resample(X, y)
        return (
            np.asarray(X_res, dtype=np.float32),
            np.asarray(y_res, dtype=np.int64),
            f"random_over_after_{mode}_failed_{type(exc).__name__}",
        )


class GradeSequenceCNNBiLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        conv_filters: int = 64,
        kernel_size: int = 3,
        num_conv_layers: int = 4,
        bilstm_hidden: int = 64,
        dense_hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if n_features < 2:
            raise ValueError("GradeSequenceCNNBiLSTM requires G1 and G2 as the first two processed features.")
        layers: list[nn.Module] = []
        for index in range(int(num_conv_layers)):
            layers.append(
                nn.Conv1d(
                    1 if index == 0 else conv_filters,
                    conv_filters,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())
        layers.append(nn.AdaptiveMaxPool1d(4))
        self.cnn = nn.Sequential(*layers)
        self.grade_bilstm = nn.LSTM(1, bilstm_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        fused_dim = conv_filters + bilstm_hidden * 2
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, max(1, dense_hidden // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, dense_hidden // 2), n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn(x.unsqueeze(1)).mean(dim=2)
        grade_sequence = x[:, :2].unsqueeze(-1)
        grade_out, _ = self.grade_bilstm(grade_sequence)
        context = grade_out[:, -1, :]
        fused = torch.cat([cnn_out, context], dim=1)
        return self.classifier(self.dropout(fused))


def make_model(dataset: str, split: PreparedSplit, case: SweepCase) -> nn.Module:
    if case.model_variant == "grade_seq":
        if dataset not in STUDENT_DATASETS:
            raise ValueError("grade_seq model_variant is only supported for student datasets.")
        return GradeSequenceCNNBiLSTM(
            split.n_features,
            split.n_classes,
            conv_filters=case.conv_filters,
            kernel_size=case.kernel_size,
            num_conv_layers=case.num_conv_layers,
            bilstm_hidden=case.bilstm_hidden,
            dense_hidden=case.dense_hidden,
            dropout=case.dropout,
        )
    if case.model_variant == "improved":
        return ImprovedCNNBiLSTM(
            model_type_for(dataset),
            split.n_features,
            split.n_classes,
            conv_filters=case.conv_filters,
            kernel_size=case.kernel_size,
            num_conv_layers=case.num_conv_layers,
            bilstm_hidden=case.bilstm_hidden,
            num_bilstm_layers=case.num_bilstm_layers,
            dense_hidden=case.dense_hidden,
            dropout=case.dropout,
            use_batchnorm=case.use_batchnorm,
            use_residual=case.use_residual,
            use_attention=case.use_attention,
        )
    if case.model_variant != "exact":
        raise ValueError(f"Unsupported model variant: {case.model_variant}")
    return ExactPaperCNNBiLSTM(
        model_type_for(dataset),
        split.n_features,
        split.n_classes,
        conv_filters=case.conv_filters,
        kernel_size=case.kernel_size,
        bilstm_hidden=case.bilstm_hidden,
        dense_hidden=case.dense_hidden,
        dropout=case.dropout,
    )


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
    )


def predict_proba(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=int(batch_size), shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            logits = model(batch_x.to(device))
            chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def criterion_for(case: SweepCase, split: PreparedSplit, device: torch.device) -> nn.CrossEntropyLoss:
    weight_tensor = None
    if case.loss == "class_weight":
        weights = compute_class_weight("balanced", classes=np.arange(split.n_classes), y=split.y_train)
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    elif case.loss != "ce":
        raise ValueError(f"Unsupported loss: {case.loss}")
    return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=float(case.label_smoothing))


def train_case_seed(dataset: str, raw: pd.DataFrame, case: SweepCase, seed: int, epochs: int) -> TrainResult:
    set_seed(seed)
    random.seed(seed)
    split = prepare_split(dataset, raw, case, seed)
    sampling = effective_oversampling(dataset, case)
    if case.split_policy == "presplit_oversample_global_scaler":
        X_train = np.asarray(split.X_train, dtype=np.float32)
        y_train = np.asarray(split.y_train, dtype=np.int64)
        effective = f"pre_split_{split.pre_split_oversampling_effective or sampling}"
    else:
        X_train, y_train, effective = oversample_train(split.X_train, split.y_train, sampling, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(dataset, split, case).to(device)
    loader = make_loader(X_train, y_train, case.batch_size, seed)
    criterion = criterion_for(case, split, device)
    if case.weight_decay > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(case.lr), weight_decay=float(case.weight_decay))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(case.lr))

    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    wait = 0
    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: list[float] = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        probs = predict_proba(model, split.X_test, case.batch_size, device)
        pred = probs.argmax(axis=1)
        metrics = metric_dict(split.y_test, pred)
        history.append({"epoch": int(epoch), "train_loss": float(np.mean(losses)) if losses else 0.0, **metrics})
        primary = case.selection_metric
        secondary = "accuracy" if primary == "f1_macro" else "f1_macro"
        is_better = best_metrics is None or metrics[primary] > best_metrics[primary] + 1e-12
        if best_metrics is not None and abs(metrics[primary] - best_metrics[primary]) <= 1e-12:
            is_better = metrics[secondary] > best_metrics[secondary] + 1e-12
        if is_better:
            best_metrics = metrics
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if case.patience > 0 and wait >= int(case.patience):
                break
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    probabilities = predict_proba(model, split.X_test, case.batch_size, device)
    predictions = probabilities.argmax(axis=1)
    final_metrics = metric_dict(split.y_test, predictions)
    labels = list(range(split.n_classes))
    report = classification_report(
        split.y_test,
        predictions,
        labels=labels,
        target_names=CLASS_NAMES[dataset],
        output_dict=True,
        zero_division=0,
    )
    row = {
        "dataset": dataset,
        "case": case.name,
        "seed": int(seed),
        "protocol": "paper-like optimistic case sweep",
        "best_epoch": int(best_epoch),
        "elapsed_seconds": float(time.time() - started),
        "model_variant": case.model_variant,
        "num_conv_layers": int(case.num_conv_layers),
        "num_bilstm_layers": int(case.num_bilstm_layers),
        "use_batchnorm": bool(case.use_batchnorm),
        "use_residual": bool(case.use_residual),
        "use_attention": bool(case.use_attention),
        "weight_decay": float(case.weight_decay),
        "patience": int(case.patience),
        "feature_set": case.feature_set,
        "binning": case.binning_key,
        "split_policy": case.split_policy,
        "selection_metric": case.selection_metric,
        "feature_columns": split.feature_columns,
        "processed_feature_names": split.processed_feature_names,
        "oversampling_requested": case.oversampling,
        "oversampling_effective": effective,
        "loss": case.loss,
        "label_smoothing": float(case.label_smoothing),
        "train_size": int(len(split.y_train)),
        "test_size": int(len(split.y_test)),
        "n_features": int(split.n_features),
        "n_classes": int(split.n_classes),
        "confusion_matrix": confusion_matrix(split.y_test, predictions, labels=labels).tolist(),
        "classification_report": report,
        "history": history,
        **final_metrics,
    }
    return TrainResult(row=row, state_dict=best_state)


def available_cases(datasets: list[str], case_names: list[str] | None, max_cases: int | None) -> list[SweepCase]:
    selected = [case for case in CASES if any(dataset in case.datasets for dataset in datasets)]
    if case_names:
        wanted = set(case_names)
        selected = [case for case in selected if case.name in wanted]
        missing = sorted(wanted - {case.name for case in selected})
        if missing:
            raise ValueError(f"Unknown or inapplicable V6 cases: {missing}")
    if max_cases and max_cases > 0:
        selected = selected[: int(max_cases)]
    return selected


def case_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["case"]), []).append(row)
    summaries = []
    for (dataset, case), case_rows in grouped.items():
        f1s = np.asarray([float(row["f1_macro"]) for row in case_rows], dtype=np.float64)
        accs = np.asarray([float(row["accuracy"]) for row in case_rows], dtype=np.float64)
        best = max(case_rows, key=lambda row: (row["f1_macro"], row["accuracy"]))
        first = case_rows[0]
        summaries.append(
            {
                "dataset": dataset,
                "case": case,
                "n_seeds": int(len(case_rows)),
                "binning": first.get("binning", BINNING_KEY),
                "split_policy": first.get("split_policy", "clean_train_only"),
                "selection_metric": first.get("selection_metric", "f1_macro"),
                "model_variant": first.get("model_variant", "exact"),
                "best_seed": int(best["seed"]),
                "best_epoch": int(best["best_epoch"]),
                "best_accuracy": float(best["accuracy"]),
                "best_f1_macro": float(best["f1_macro"]),
                "accuracy_mean": float(accs.mean()),
                "accuracy_std": float(accs.std(ddof=0)),
                "f1_macro_mean": float(f1s.mean()),
                "f1_macro_std": float(f1s.std(ddof=0)),
            }
        )
    return sorted(summaries, key=lambda row: (row["dataset"], -row["best_f1_macro"], -row["f1_macro_mean"]))


def best_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        dataset = row["dataset"]
        current = best.get(dataset)
        if current is None or row["f1_macro"] > current["f1_macro"] + 1e-12:
            best[dataset] = row
        elif current is not None and abs(row["f1_macro"] - current["f1_macro"]) <= 1e-12 and row["accuracy"] > current["accuracy"] + 1e-12:
            best[dataset] = row
    return best


def save_best_models(
    rows: list[dict[str, Any]],
    states: dict[tuple[str, str, int], dict[str, torch.Tensor]],
    *,
    model_prefix: str,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for dataset, row in best_rows(rows).items():
        key = (dataset, row["case"], int(row["seed"]))
        case = next(item for item in CASES if item.name == row["case"])
        path = MODELS_DIR / f"{model_prefix}_{dataset}.pt"
        torch.save(
            {
                "state_dict": states[key],
                "dataset": dataset,
                "case": row["case"],
                "seed": int(row["seed"]),
                "protocol": row["protocol"],
                "metrics": {name: row[name] for name in ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
                "feature_columns": row["feature_columns"],
                "processed_feature_names": row["processed_feature_names"],
                "case_params": asdict(case),
            },
            path,
        )
        paths[dataset] = str(path)
    return paths


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(fmt(value))
            else:
                values.append(str(value) if value is not None else "-")
        lines.append("| " + " | ".join(values) + " |")
    return lines


def final_comparison(best_by_dataset: dict[str, dict[str, Any]], run_label: str) -> list[dict[str, Any]]:
    rows = []
    for dataset in ALL_DATASETS:
        if dataset not in best_by_dataset:
            continue
        paper = PAPER_REFERENCES[dataset]
        v1 = V1_REFERENCES[dataset]
        best = best_by_dataset[dataset]
        paper_row = "Paper benchmark" if paper.get("f1_macro") is not None else "Paper benchmark unavailable"
        v1_row = "V1 strict reference" if v1.get("f1_macro") is not None else "V1 strict reference unavailable"
        rows.append({"dataset": dataset, "row": paper_row, "accuracy": paper.get("accuracy"), "f1_macro": paper.get("f1_macro")})
        rows.append({"dataset": dataset, "row": v1_row, "accuracy": v1.get("accuracy"), "f1_macro": v1.get("f1_macro")})
        rows.append(
            {
                "dataset": dataset,
                "row": f"{run_label} best {best['case']} seed {best['seed']}",
                "accuracy": best["accuracy"],
                "f1_macro": best["f1_macro"],
            }
        )
    return rows


def build_payload(
    *,
    rows: list[dict[str, Any]],
    run_label: str,
    datasets: list[str],
    seeds: list[int],
    cases: list[SweepCase],
    epochs: int,
    result_path: Path,
    report_path: Path,
    model_paths: dict[str, str] | None = None,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    return {
        "config": {
            "protocol": "paper-like optimistic case sweep",
            "run_label": run_label,
            "datasets": datasets,
            "seeds": seeds,
            "epochs": int(epochs),
            "case_names": [case.name for case in cases],
            "binning": BINNING_KEY,
            "warning": "Test set is used for best epoch and best case selection.",
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
        },
        "benchmarks": PAPER_REFERENCES,
        "v1_references": V1_REFERENCES,
        "case_summary": case_summary(rows),
        "best_by_dataset": best_rows(rows),
        "rows": rows,
        "artifacts": {
            "results": str(result_path),
            "report": str(report_path),
            "models": model_paths or {},
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
        },
    }


def save_seed_checkpoint(
    *,
    model_prefix: str,
    dataset: str,
    case: SweepCase,
    seed: int,
    row: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
) -> Path:
    path = seed_state_path(model_prefix, dataset, case.name, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state_dict,
            "dataset": dataset,
            "case": case.name,
            "seed": int(seed),
            "metrics": {name: row[name] for name in ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
            "feature_columns": row["feature_columns"],
            "processed_feature_names": row["processed_feature_names"],
            "case_params": asdict(case),
        },
        path,
    )
    return path


def load_seed_checkpoint(model_prefix: str, dataset: str, case_name: str, seed: int) -> dict[str, torch.Tensor] | None:
    path = seed_state_path(model_prefix, dataset, case_name, seed)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    return payload["state_dict"]


def load_resume_rows(checkpoint: Path) -> list[dict[str, Any]]:
    if not checkpoint.exists():
        return []
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"Invalid checkpoint rows in {checkpoint}")
    return rows


def write_report(payload: dict[str, Any], report_path: Path) -> None:
    best = payload["best_by_dataset"]
    summaries = payload["case_summary"]
    run_label = payload["config"].get("run_label", "V6")
    comparison = final_comparison(best, run_label)
    top_summaries = []
    for dataset in payload["config"]["datasets"]:
        dataset_rows = [row for row in summaries if row["dataset"] == dataset]
        top_summaries.extend(dataset_rows[:8])
    lines = [
        f"# {run_label} CNN-BiLSTM Case Sweep Report",
        "",
        "Protocol: `paper-like optimistic case sweep`. The test set is used during training; each case uses its "
        "`selection_metric` for best-epoch selection, while final case comparison is sorted by macro-F1.",
        "",
        "## Final Comparison",
        "",
        *markdown_table(comparison, ["dataset", "row", "accuracy", "f1_macro"]),
        "",
        "## Top Case Summaries",
        "",
        *markdown_table(
            top_summaries,
            [
                "dataset",
                "case",
                "n_seeds",
                "binning",
                "split_policy",
                "selection_metric",
                "model_variant",
                "best_seed",
                "best_epoch",
                "best_accuracy",
                "best_f1_macro",
                "f1_macro_mean",
                "f1_macro_std",
            ],
        ),
        "",
        "## Honest Conclusion",
        "",
    ]
    for dataset, row in best.items():
        paper_f1 = PAPER_REFERENCES[dataset].get("f1_macro")
        v1_f1 = V1_REFERENCES[dataset].get("f1_macro")
        if v1_f1 is None:
            lines.append(f"- `{dataset}` has no V1 strict reference; {run_label} best F1 is {row['f1_macro']:.4f}.")
        else:
            relation = "improves over" if row["f1_macro"] > v1_f1 else "does not improve over"
            lines.append(f"- `{dataset}` {run_label} best {relation} V1: F1 {row['f1_macro']:.4f} vs {v1_f1:.4f}.")
        if paper_f1 is None:
            lines.append(f"- `{dataset}` has no separate paper benchmark.")
        else:
            paper_delta = row["f1_macro"] - paper_f1
            lines.append(f"- `{dataset}` {'exceeds' if paper_delta > 0 else 'trails'} paper F1 by {abs(paper_delta):.4f}.")
        if row.get("feature_set") != "paper":
            lines.append(
                f"- `{dataset}` best uses feature_set=`{row.get('feature_set')}` with "
                "exploratory features, so it is not a strict paper-feature replication."
            )
    lines.extend(
        [
            "- These numbers are optimistic and must be labeled separately from strict validation/test results.",
            "",
            "## Artifacts",
            "",
            f"- Results JSON: `{payload['artifacts']['results']}`",
            f"- Report: `{payload['artifacts']['report']}`",
        ]
    )
    for dataset, path in payload["artifacts"]["models"].items():
        lines.append(f"- `{dataset}` model: `{path}`")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dirs()
    result_path, report_path, model_prefix = artifact_paths(args.output_tag)
    checkpoint = checkpoint_path(args.output_tag)
    run_label = args.output_tag.strip().upper() if args.output_tag.strip() else "V6"
    datasets = parse_csv_strings(args.datasets)
    unknown = sorted(set(datasets) - set(ALL_DATASETS))
    if unknown:
        raise ValueError(f"Unsupported datasets: {unknown}")
    seeds = parse_csv_ints(args.seeds)
    case_names = parse_csv_strings(args.case_names) if args.case_names else None
    cases = available_cases(datasets, case_names, args.max_cases)
    raws = {dataset: read_dataset(dataset) for dataset in datasets}
    rows: list[dict[str, Any]] = load_resume_rows(checkpoint) if args.resume else []
    rows = [
        row
        for row in rows
        if row.get("dataset") in datasets
        and load_seed_checkpoint(model_prefix, row["dataset"], row["case"], int(row["seed"])) is not None
    ]
    completed = {row_key(row["dataset"], row["case"], int(row["seed"])) for row in rows}
    states: dict[tuple[str, str, int], dict[str, torch.Tensor]] = {}
    if args.resume and rows:
        for row in rows:
            key = row_key(row["dataset"], row["case"], int(row["seed"]))
            state_dict = load_seed_checkpoint(model_prefix, row["dataset"], row["case"], int(row["seed"]))
            if state_dict is not None:
                states[key] = state_dict
    print(f"\n=== {run_label} CNN-BiLSTM paper-like case sweep ===", flush=True)
    print(f"Datasets={datasets} seeds={seeds} epochs={args.epochs} cases={len(cases)}", flush=True)
    if args.resume:
        print(f"Resume checkpoint={checkpoint} completed_seeds={len(completed)}", flush=True)
    for dataset in datasets:
        dataset_cases = [case for case in cases if dataset in case.datasets]
        print(f"\n--- Dataset {dataset}: {len(dataset_cases)} cases ---", flush=True)
        for case in dataset_cases:
            case_rows = []
            for seed in seeds:
                key = row_key(dataset, case.name, int(seed))
                if key in completed:
                    row = next(row for row in rows if row_key(row["dataset"], row["case"], int(row["seed"])) == key)
                    case_rows.append(row)
                    print(
                        f"{dataset:12s} {case.name:24s} seed={seed:<4d} "
                        f"SKIP acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f} epoch={row['best_epoch']}",
                        flush=True,
                    )
                    continue
                result = train_case_seed(dataset, raws[dataset], case, seed, args.epochs)
                rows.append(result.row)
                case_rows.append(result.row)
                states[key] = copy.deepcopy(result.state_dict)
                save_seed_checkpoint(
                    model_prefix=model_prefix,
                    dataset=dataset,
                    case=case,
                    seed=int(seed),
                    row=result.row,
                    state_dict=result.state_dict,
                )
                save_json(
                    checkpoint,
                    build_payload(
                        rows=rows,
                        run_label=run_label,
                        datasets=datasets,
                        seeds=seeds,
                        cases=cases,
                        epochs=args.epochs,
                        result_path=result_path,
                        report_path=report_path,
                        checkpoint=checkpoint,
                    ),
                )
                print(
                    f"{dataset:12s} {case.name:24s} seed={seed:<4d} "
                    f"acc={result.row['accuracy']:.4f} f1={result.row['f1_macro']:.4f} epoch={result.row['best_epoch']}",
                    flush=True,
                )
            f1s = np.asarray([row["f1_macro"] for row in case_rows], dtype=np.float64)
            print(f"CASE {dataset} {case.name}: mean_f1={f1s.mean():.4f}+/-{f1s.std(ddof=0):.4f}", flush=True)
    summaries = case_summary(rows)
    best = best_rows(rows)
    model_paths = save_best_models(rows, states, model_prefix=model_prefix)
    payload = build_payload(
        rows=rows,
        run_label=run_label,
        datasets=datasets,
        seeds=seeds,
        cases=cases,
        epochs=args.epochs,
        result_path=result_path,
        report_path=report_path,
        model_paths=model_paths,
        checkpoint=checkpoint,
    )
    payload["case_summary"] = summaries
    payload["best_by_dataset"] = best
    save_json(result_path, payload)
    write_report(payload, report_path)
    print(f"\n=== {run_label} FINAL BEST ===", flush=True)
    for dataset, row in best.items():
        print(
            f"{dataset:12s} case={row['case']} seed={row['seed']} "
            f"acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f}",
            flush=True,
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V6 paper-like CNN-BiLSTM case sweep.")
    parser.add_argument("--datasets", default="xapi")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--case-names", default="")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_all(parse_args())


if __name__ == "__main__":
    main()
