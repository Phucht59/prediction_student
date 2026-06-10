from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.paper_replication.pipeline import (
    DATASETS,
    PROJECT_ROOT,
    RAW_DIR,
    XAPI_CLASS_MAPPING,
    dense_array,
    make_one_hot_encoder,
    read_raw,
    resolve_columns,
)


ROOT_RESULTS_DIR = PROJECT_ROOT / "results"
ROOT_MODELS_DIR = PROJECT_ROOT / "models"
ROOT_OPTUNA_DIR = PROJECT_ROOT / "optuna"
ALL_EXPERIMENTS_PATH = ROOT_RESULTS_DIR / "all_experiments.json"

PAPER_DISTRIBUTION = np.asarray([32.91, 26.08, 15.70, 15.19, 10.13], dtype=np.float64)
PAPER_BENCHMARKS = {
    "student-mat": {"accuracy": 1.0000, "f1_macro": 0.9400},
    "student-por": {"accuracy": 0.9231, "f1_macro": 0.9000},
    "xapi": {"accuracy": 0.8438, "precision_macro": 0.8426, "recall_macro": 0.8521, "f1_macro": 0.8447},
}

STUDENT_DATASETS = ["student-mat", "student-por"]
ALL_DATASETS = ["student-mat", "student-por", "xapi"]

BINNING_STRATEGIES: dict[str, dict[str, Any]] = {
    "H1_portuguese_scale": {
        "mat": [-0.1, 9, 11, 13, 15, 20],
        "por": [-0.1, 9, 11, 13, 15, 20],
        "description": "Portuguese school grade scale",
    },
    "H2_quintile": {
        "description": "Equal-frequency quintile cut",
    },
    "H3_paper_percentile": {
        "mat_quantiles": [0.0, 0.1013, 0.3621, 0.5191, 0.6910, 1.0],
        "por_quantiles": [0.0, 0.1263, 0.4375, 0.6744, 0.8469, 1.0],
        "description": "Custom quantiles reverse-engineered from Figure 1 distribution",
    },
    "H4_equal_width": {
        "mat": [-0.1, 4, 8, 12, 16, 20],
        "por": [-0.1, 4, 8, 12, 16, 20],
        "description": "Equal-width bins",
    },
    "H5_pass_fail_3class": {
        "mat": [-0.1, 9, 14, 20],
        "por": [-0.1, 9, 14, 20],
        "description": "3-class fail/average/excellent",
    },
}

BINNING_ALIASES = {
    "H1_portuguese": "H1_portuguese_scale",
    "H1": "H1_portuguese_scale",
    "H2": "H2_quintile",
    "H3": "H3_paper_percentile",
    "H4": "H4_equal_width",
    "H5": "H5_pass_fail_3class",
}


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    n_features: int
    n_classes: int
    feature_columns: list[str]


@dataclass
class TrainResult:
    metrics: dict[str, float]
    val_metrics: dict[str, float]
    best_epoch: int
    best_val_f1: float
    best_val_accuracy: float
    elapsed_seconds: float
    state_dict: dict[str, torch.Tensor]
    probabilities: np.ndarray
    predictions: np.ndarray


def ensure_advanced_dirs() -> None:
    for path in (ROOT_RESULTS_DIR, ROOT_MODELS_DIR, ROOT_OPTUNA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dataset_short(dataset: str) -> str:
    if dataset == "student-mat":
        return "mat"
    if dataset == "student-por":
        return "por"
    return "xapi"


def normalize_binning_key(key: str) -> str:
    return BINNING_ALIASES.get(key, key)


def apply_binning(series: pd.Series, strategy_key: str, dataset: str) -> pd.Series:
    key = normalize_binning_key(strategy_key)
    strategy = BINNING_STRATEGIES[key]
    values = pd.to_numeric(series, errors="raise")
    short = dataset_short(dataset)
    if key == "H2_quintile":
        return pd.qcut(values, 5, labels=False, duplicates="drop").astype("int64")
    if key == "H3_paper_percentile":
        quantiles = strategy[f"{short}_quantiles"]
        edges = values.quantile(quantiles).to_numpy(dtype=np.float64)
        edges[0] = min(edges[0], values.min()) - 0.1
        edges[-1] = max(edges[-1], values.max())
        edges = np.maximum.accumulate(edges)
        for index in range(1, len(edges)):
            if edges[index] <= edges[index - 1]:
                edges[index] = edges[index - 1] + 1e-6
        return pd.cut(values, bins=edges, labels=False, include_lowest=True).astype("int64")
    bins = strategy[short]
    return pd.cut(values, bins=bins, labels=False, include_lowest=True).astype("int64")


def class_distribution_percent(y: pd.Series | np.ndarray, n_classes: int | None = None) -> list[float]:
    values = np.asarray(y, dtype=np.int64)
    if n_classes is None:
        n_classes = int(values.max()) + 1
    counts = np.bincount(values, minlength=n_classes).astype(np.float64)
    return (counts / counts.sum() * 100.0).round(4).tolist()


def l1_to_paper(distribution: list[float]) -> float | None:
    if len(distribution) != len(PAPER_DISTRIBUTION):
        return None
    return float(np.abs(np.asarray(distribution) - PAPER_DISTRIBUTION).sum())


def spearman_binned(df: pd.DataFrame, feature: str, target: str, strategy_key: str, dataset: str) -> float:
    feature_binned = apply_binning(df[feature], strategy_key, dataset)
    return float(feature_binned.corr(df[target], method="spearman"))


def test_all_binnings(df_mat: pd.DataFrame, df_por: pd.DataFrame) -> dict[str, Any]:
    print("\n=== STAGE 1A: BINNING HYPOTHESIS TESTING ===", flush=True)
    rows: list[dict[str, Any]] = []
    frames = {"student-mat": df_mat, "student-por": df_por}
    for strategy_key, strategy in BINNING_STRATEGIES.items():
        for dataset, frame in frames.items():
            target = apply_binning(frame["G3"], strategy_key, dataset)
            n_classes = int(target.max()) + 1
            distribution = class_distribution_percent(target, n_classes)
            frame_eval = frame.copy()
            frame_eval["G3_binned"] = target
            g2_corr = spearman_binned(frame_eval, "G2", "G3_binned", strategy_key, dataset)
            g1_corr = spearman_binned(frame_eval, "G1", "G3_binned", strategy_key, dataset)
            row = {
                "strategy": strategy_key,
                "dataset": dataset,
                "description": strategy["description"],
                "n_classes": n_classes,
                "distribution_percent": distribution,
                "l1_to_paper_distribution": l1_to_paper(distribution),
                "spearman_g2_binned_g3_binned": g2_corr,
                "spearman_g1_binned_g3_binned": g1_corr,
            }
            rows.append(row)
            print(
                f"{dataset:12s} {strategy_key:22s} classes={n_classes} "
                f"dist={distribution} L1={row['l1_to_paper_distribution']} "
                f"rho(G2,G3)={g2_corr:.4f} rho(G1,G3)={g1_corr:.4f}",
                flush=True,
            )
    scored = [row for row in rows if row["n_classes"] == 5 and row["l1_to_paper_distribution"] is not None]
    by_strategy: dict[str, dict[str, Any]] = {}
    for strategy_key in BINNING_STRATEGIES:
        strategy_rows = [row for row in scored if row["strategy"] == strategy_key]
        if not strategy_rows:
            continue
        by_strategy[strategy_key] = {
            "strategy": strategy_key,
            "mean_l1_to_paper": float(np.mean([row["l1_to_paper_distribution"] for row in strategy_rows])),
            "mean_spearman_g2": float(np.mean([row["spearman_g2_binned_g3_binned"] for row in strategy_rows])),
        }
    ranked = sorted(by_strategy.values(), key=lambda item: (item["mean_l1_to_paper"], -item["mean_spearman_g2"]))
    best = ranked[0]["strategy"] if ranked else "H1_portuguese_scale"
    print(f"Selected binning: {best} (lowest mean L1 to paper, tie by Spearman G2)", flush=True)
    return {"rows": rows, "strategy_scores": ranked, "selected_binning": best}


def verify_pearson_selection(raw: pd.DataFrame, dataset: str, binning_key: str, threshold: float = 0.6) -> dict[str, Any]:
    data = raw.copy()
    data["target_class"] = apply_binning(data["G3"], binning_key, dataset)
    numeric = data.select_dtypes(include=[np.number]).copy()
    if "G3" in numeric.columns:
        numeric = numeric.drop(columns=["G3"])
    correlations = numeric.corr(numeric_only=True)["target_class"].abs().sort_values(ascending=False)
    correlations = correlations.drop(index=["target_class"], errors="ignore")
    kept = correlations[correlations >= threshold].index.tolist()
    dropped = correlations[correlations < threshold].index.tolist()
    paper_features = ["G1", "G2"]
    selected_for_training = [feature for feature in paper_features if feature in numeric.columns]
    print(f"\n=== STAGE 1B: Pearson Selection {dataset} (|r| >= {threshold}) ===", flush=True)
    print(f"KEPT ({len(kept)}): {kept}", flush=True)
    print(f"DROPPED ({len(dropped)}): {dropped}", flush=True)
    print(f"G1 in kept: {'yes' if 'G1' in kept else 'NO'}", flush=True)
    print(f"G2 in kept: {'yes' if 'G2' in kept else 'NO'}", flush=True)
    print(f"Training features retained for paper replication: {selected_for_training}", flush=True)
    return {
        "dataset": dataset,
        "threshold": threshold,
        "kept": kept,
        "dropped": dropped,
        "correlations": {key: float(value) for key, value in correlations.items()},
        "g1_in_kept": "G1" in kept,
        "g2_in_kept": "G2" in kept,
        "selected_for_training": selected_for_training,
    }


def verify_xapi_split(raw_xapi: pd.DataFrame, seed: int = 42, test_size: float = 0.2, val_size: float = 0.2) -> dict[str, Any]:
    y = raw_xapi["Class"].map(XAPI_CLASS_MAPPING).astype("int64")
    indices = np.arange(len(raw_xapi))
    trainval_idx, test_idx, y_trainval, y_test = train_test_split(
        indices, y.to_numpy(), test_size=test_size, random_state=seed, stratify=y
    )
    val_ratio = val_size / (1.0 - test_size)
    train_idx, val_idx, y_train, y_val = train_test_split(
        trainval_idx, y_trainval, test_size=val_ratio, random_state=seed, stratify=y_trainval
    )
    train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)
    overlap = {
        "train_val": sorted(train_set & val_set),
        "train_test": sorted(train_set & test_set),
        "val_test": sorted(val_set & test_set),
    }
    assert not overlap["train_val"] and not overlap["train_test"] and not overlap["val_test"]
    result = {
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "zero_overlap": True,
        "class_distribution_train": class_distribution_percent(y_train, 3),
        "class_distribution_val": class_distribution_percent(y_val, 3),
        "class_distribution_test": class_distribution_percent(y_test, 3),
        "overlap": overlap,
    }
    print("\n=== STAGE 1C: xAPI clean split verification ===", flush=True)
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)
    return result


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in X_train.columns if column not in numeric_columns]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_one_hot_encoder())]),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def prepare_split(
    raw: pd.DataFrame,
    dataset: str,
    feature_columns: list[str],
    y: np.ndarray,
    *,
    seed: int,
    train_indices: np.ndarray | None = None,
    val_indices: np.ndarray | None = None,
    test_indices: np.ndarray | None = None,
) -> SplitData:
    indices = np.arange(len(raw))
    if train_indices is None or val_indices is None or test_indices is None:
        trainval_idx, test_indices, y_trainval, _ = train_test_split(
            indices, y, test_size=0.2, random_state=seed, stratify=y
        )
        train_indices, val_indices, _, _ = train_test_split(
            trainval_idx,
            y_trainval,
            test_size=0.25,
            random_state=seed,
            stratify=y_trainval,
        )
    train_set, val_set, test_set = set(train_indices), set(val_indices), set(test_indices)
    assert not (train_set & val_set), "Data leakage: train/val overlap."
    assert not (train_set & test_set), "Data leakage: train/test overlap."
    assert not (val_set & test_set), "Data leakage: val/test overlap."

    X = raw[feature_columns].copy()
    preprocessor = build_preprocessor(X.iloc[train_indices])
    X_train = dense_array(preprocessor.fit_transform(X.iloc[train_indices]))
    X_val = dense_array(preprocessor.transform(X.iloc[val_indices]))
    X_test = dense_array(preprocessor.transform(X.iloc[test_indices]))
    return SplitData(
        X_train=X_train,
        y_train=np.asarray(y[train_indices], dtype=np.int64),
        X_val=X_val,
        y_val=np.asarray(y[val_indices], dtype=np.int64),
        X_test=X_test,
        y_test=np.asarray(y[test_indices], dtype=np.int64),
        train_indices=np.asarray(train_indices, dtype=np.int64),
        val_indices=np.asarray(val_indices, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64),
        n_features=int(X_train.shape[1]),
        n_classes=int(np.max(y)) + 1,
        feature_columns=feature_columns,
    )


def resample_training(X: np.ndarray, y: np.ndarray, mode: str, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    if mode in {"none", "class_weight"}:
        return X, y, mode
    counts = np.bincount(y)
    present = counts[counts > 0]
    if len(present) == 0:
        return X, y, mode
    if present.min() < 2:
        sampler = RandomOverSampler(random_state=seed)
    else:
        n_neighbors = int(min(5, present.min() - 1))
        if mode == "smote":
            sampler = SMOTE(random_state=seed, k_neighbors=n_neighbors)
        elif mode == "adasyn":
            sampler = ADASYN(random_state=seed, n_neighbors=n_neighbors)
        else:
            return X, y, mode
    try:
        X_res, y_res = sampler.fit_resample(X, y)
        return np.asarray(X_res, dtype=np.float32), np.asarray(y_res, dtype=np.int64), mode
    except Exception as exc:
        fallback = RandomOverSampler(random_state=seed)
        X_res, y_res = fallback.fit_resample(X, y)
        return np.asarray(X_res, dtype=np.float32), np.asarray(y_res, dtype=np.int64), f"random_over_after_{mode}_failed_{type(exc).__name__}"


def loader_for(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


class ExactPaperCNNBiLSTM(nn.Module):
    def __init__(
        self,
        dataset_type: str,
        n_features: int,
        n_classes: int,
        input_layout: str = "feature_steps",
        conv_filters: int = 64,
        kernel_size: int = 3,
        bilstm_hidden: int = 64,
        dense_hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dataset_type = dataset_type
        self.n_features = n_features
        self.input_layout = input_layout
        conv_in = 1 if input_layout == "feature_steps" else n_features

        def conv_stack(in_channels: int, out_channels: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            for index in range(4):
                layers.append(
                    nn.Conv1d(
                        in_channels if index == 0 else out_channels,
                        out_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                    )
                )
                layers.append(nn.ReLU())
            layers.append(nn.AdaptiveMaxPool1d(4))
            return nn.Sequential(*layers)

        if dataset_type in {"mat", "por"}:
            self.cnn = conv_stack(conv_in, conv_filters)
            lstm_input = conv_filters
        elif dataset_type == "xapi":
            self.cnn = nn.Sequential(
                conv_stack(conv_in, conv_filters),
                conv_stack(conv_filters, conv_filters * 2),
            )
            lstm_input = conv_filters * 2
        else:
            raise ValueError(f"Unsupported dataset_type={dataset_type}")

        self.bilstm = nn.LSTM(lstm_input, bilstm_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(bilstm_hidden * 2, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, max(1, dense_hidden // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, dense_hidden // 2), n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_layout == "feature_steps":
            out = x.unsqueeze(1)
        elif self.input_layout == "channel_steps":
            out = x.unsqueeze(2)
        else:
            raise ValueError(f"Unsupported input_layout={self.input_layout}")
        out = self.cnn(out)
        out = out.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(out)
        out = self.dropout(lstm_out[:, -1, :])
        return self.classifier(out)


class ConvResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_layers: int, use_batchnorm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(n_layers):
            ch_in = in_channels if index == 0 else out_channels
            layers.append(nn.Conv1d(ch_in, out_channels, kernel_size, padding=kernel_size // 2))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)
        self.skip = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        skip = self.skip(x)
        return out + skip if out.shape == skip.shape else out


class ImprovedCNNBiLSTM(nn.Module):
    def __init__(
        self,
        dataset_type: str,
        n_features: int,
        n_classes: int,
        conv_filters: int = 64,
        kernel_size: int = 3,
        num_conv_layers: int = 4,
        bilstm_hidden: int = 128,
        num_bilstm_layers: int = 1,
        dense_hidden: int = 128,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        use_residual: bool = True,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_type = dataset_type
        self.use_residual = use_residual
        self.use_attention = use_attention and dataset_type != "xapi"
        self.conv_block1 = ConvResidualBlock(1, conv_filters, kernel_size, num_conv_layers, use_batchnorm)
        self.pool1 = nn.AdaptiveMaxPool1d(4)
        self.drop1 = nn.Dropout(dropout)
        if dataset_type == "xapi":
            self.conv_block2 = ConvResidualBlock(conv_filters, conv_filters * 2, kernel_size, num_conv_layers, use_batchnorm)
            self.pool2 = nn.AdaptiveMaxPool1d(4)
            self.drop2 = nn.Dropout(dropout)
            lstm_input = conv_filters * 2
        else:
            self.conv_block2 = None
            lstm_input = conv_filters
        self.bilstm = nn.LSTM(
            lstm_input,
            bilstm_hidden,
            num_layers=num_bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_bilstm_layers > 1 else 0.0,
        )
        self.lstm_drop = nn.Dropout(dropout)
        if self.use_attention:
            self.attention = nn.Linear(bilstm_hidden * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(bilstm_hidden * 2, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, max(1, dense_hidden // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, dense_hidden // 2), n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.unsqueeze(1)
        out = self.conv_block1(out)
        out = self.pool1(out)
        out = self.drop1(out)
        if self.conv_block2 is not None:
            out = self.conv_block2(out)
            out = self.pool2(out)
            out = self.drop2(out)
        out = out.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(out)
        lstm_out = self.lstm_drop(lstm_out)
        if self.use_attention:
            weights = torch.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * weights).sum(dim=1)
        else:
            context = lstm_out[:, -1, :]
        return self.classifier(context)


def predict_proba(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for (batch_x,) in DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size):
            logits = model(batch_x.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def train_model_once(
    model_factory: Callable[[], nn.Module],
    split: SplitData,
    *,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    oversampling: str,
    label_smoothing: float = 0.0,
    grad_clip: float | None = None,
    use_cosine: bool = False,
) -> TrainResult:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, effective_sampling = resample_training(split.X_train, split.y_train, oversampling, seed)
    model = model_factory().to(device)
    train_loader = loader_for(X_train, y_train, batch_size, True, seed)
    weights = None
    if oversampling == "class_weight":
        classes = np.arange(split.n_classes)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=split.y_train)
        weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)
        if use_cosine
        else None
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0
    best_val_accuracy = -1.0
    best_epoch = 0
    wait = 0
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        val_probs = predict_proba(model, split.X_val, batch_size, device)
        val_pred = val_probs.argmax(axis=1)
        val_metrics = metric_dict(split.y_val, val_pred)
        val_f1 = val_metrics["f1_macro"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_probs = predict_proba(model, split.X_test, batch_size, device)
    test_pred = test_probs.argmax(axis=1)
    metrics = metric_dict(split.y_test, test_pred)
    metrics["effective_oversampling"] = effective_sampling
    return TrainResult(
        metrics=metrics,
        val_metrics={"accuracy": best_val_accuracy, "f1_macro": best_val_f1},
        best_epoch=int(best_epoch),
        best_val_f1=float(best_val_f1),
        best_val_accuracy=float(best_val_accuracy),
        elapsed_seconds=float(time.time() - started),
        state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        probabilities=test_probs,
        predictions=test_pred,
    )


def student_y(raw: pd.DataFrame, dataset: str, binning_key: str) -> np.ndarray:
    return apply_binning(raw["G3"], binning_key, dataset).to_numpy(dtype=np.int64)


def xapi_y(raw: pd.DataFrame) -> np.ndarray:
    return raw["Class"].map(XAPI_CLASS_MAPPING).to_numpy(dtype=np.int64)


def paper_feature_columns(dataset: str, raw: pd.DataFrame, pearson_results: dict[str, Any]) -> list[str]:
    if dataset in STUDENT_DATASETS:
        return list(pearson_results[dataset]["selected_for_training"])
    spec = DATASETS["xapi"]
    return resolve_columns(raw.columns.tolist(), spec.paper_feature_aliases)


def model_dataset_type(dataset: str) -> str:
    return {"student-mat": "mat", "student-por": "por", "xapi": "xapi"}[dataset]


def run_holdout_replication(
    dataset: str,
    raw: pd.DataFrame,
    y: np.ndarray,
    feature_columns: list[str],
    *,
    binning: str,
    oversampling: str,
    epochs: int,
    seed: int,
    input_layout: str = "feature_steps",
) -> dict[str, Any]:
    split = prepare_split(raw, dataset, feature_columns, y, seed=seed)
    dtype = model_dataset_type(dataset)

    def factory() -> nn.Module:
        return ExactPaperCNNBiLSTM(dtype, split.n_features, split.n_classes, input_layout=input_layout)

    result = train_model_once(
        factory,
        split,
        seed=seed,
        epochs=epochs,
        patience=10,
        batch_size=16 if dataset == "xapi" else 32,
        lr=1e-3,
        weight_decay=1e-4,
        oversampling=oversampling,
    )
    return {
        "dataset": dataset,
        "stage": "replication",
        "model": "ExactPaperCNNBiLSTM",
        "binning": binning,
        "eval_mode": "holdout",
        "oversampling": oversampling,
        "input_layout": input_layout,
        "epochs": epochs,
        "best_epoch": result.best_epoch,
        "best_val_f1": result.best_val_f1,
        "best_val_accuracy": result.best_val_accuracy,
        "feature_columns": feature_columns,
        **result.metrics,
    }


def run_cv_replication(
    dataset: str,
    raw: pd.DataFrame,
    y: np.ndarray,
    feature_columns: list[str],
    *,
    binning: str,
    oversampling: str,
    epochs: int,
    seed: int,
    eval_mode: str,
    input_layout: str = "feature_steps",
) -> dict[str, Any]:
    counts = np.bincount(y)
    n_splits = max(2, min(5, int(counts[counts > 0].min())))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rows = []
    indices = np.arange(len(raw))
    dtype = model_dataset_type(dataset)
    for fold, (trainval_idx, test_idx) in enumerate(folds.split(indices, y), start=1):
        y_trainval = y[trainval_idx]
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=0.2,
            random_state=seed + fold,
            stratify=y_trainval,
        )
        split = prepare_split(
            raw,
            dataset,
            feature_columns,
            y,
            seed=seed,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
        )

        def factory() -> nn.Module:
            return ExactPaperCNNBiLSTM(dtype, split.n_features, split.n_classes, input_layout=input_layout)

        result = train_model_once(
            factory,
            split,
            seed=seed + fold,
            epochs=epochs,
            patience=10,
            batch_size=16 if dataset == "xapi" else 32,
            lr=1e-3,
            weight_decay=1e-4,
            oversampling=oversampling,
        )
        fold_rows.append(
            {
                "fold": fold,
                "best_epoch": result.best_epoch,
                "best_val_accuracy": result.best_val_accuracy,
                "best_val_f1": result.best_val_f1,
                **result.metrics,
            }
        )
        print(
            f"  {dataset} {binning}+{eval_mode}+{oversampling} fold={fold}/{n_splits} "
            f"acc={result.metrics['accuracy']:.4f} f1={result.metrics['f1_macro']:.4f}",
            flush=True,
        )
    if eval_mode == "best_fold":
        chosen = sorted(fold_rows, key=lambda row: row["best_val_accuracy"], reverse=True)[0]
        aggregate = dict(chosen)
    elif eval_mode == "avg_fold":
        metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "best_val_accuracy", "best_val_f1"]
        aggregate = {key: float(np.mean([row[key] for row in fold_rows])) for key in metric_keys}
        aggregate.update({f"{key}_std": float(np.std([row[key] for row in fold_rows])) for key in metric_keys})
        aggregate["best_epoch"] = float(np.mean([row["best_epoch"] for row in fold_rows]))
    else:
        raise ValueError(f"Unsupported CV eval_mode={eval_mode}")
    return {
        "dataset": dataset,
        "stage": "replication",
        "model": "ExactPaperCNNBiLSTM",
        "binning": binning,
        "eval_mode": eval_mode,
        "oversampling": oversampling,
        "input_layout": input_layout,
        "epochs": epochs,
        "n_splits": n_splits,
        "feature_columns": feature_columns,
        "folds": fold_rows,
        **aggregate,
    }


def run_stage2_replication(
    raws: dict[str, pd.DataFrame],
    selected_binning: str,
    pearson_results: dict[str, Any],
    *,
    replication_epochs: int,
    seed: int,
) -> list[dict[str, Any]]:
    print("\n=== STAGE 2: BASELINE/PAPER REPLICATION ===", flush=True)
    results: list[dict[str, Any]] = []
    experiments_student = [
        ("H1_portuguese_scale", "best_fold", "smote", 50),
        ("H1_portuguese_scale", "best_fold", "none", 50),
        ("H1_portuguese_scale", "avg_fold", "smote", 50),
        ("H1_portuguese_scale", "holdout", "smote", 50),
        ("H2_quintile", "best_fold", "smote", 50),
        ("H2_quintile", "avg_fold", "smote", 50),
        (selected_binning, "holdout", "smote", 50),
    ]
    seen = set()
    for binning, eval_mode, oversampling, default_epochs in experiments_student:
        for dataset in STUDENT_DATASETS:
            key = (dataset, binning, eval_mode, oversampling)
            if key in seen:
                continue
            seen.add(key)
            raw = raws[dataset]
            y = student_y(raw, dataset, binning)
            feature_columns = paper_feature_columns(dataset, raw, pearson_results)
            epochs = min(replication_epochs, default_epochs)
            print(f"Running {dataset} {binning}+{eval_mode}+{oversampling} epochs={epochs}", flush=True)
            if eval_mode == "holdout":
                row = run_holdout_replication(
                    dataset,
                    raw,
                    y,
                    feature_columns,
                    binning=binning,
                    oversampling=oversampling,
                    epochs=epochs,
                    seed=seed,
                )
            else:
                row = run_cv_replication(
                    dataset,
                    raw,
                    y,
                    feature_columns,
                    binning=binning,
                    oversampling=oversampling,
                    epochs=epochs,
                    seed=seed,
                    eval_mode=eval_mode,
                )
            print(
                f"RESULT {dataset} {binning}+{eval_mode}+{oversampling}: "
                f"acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f}",
                flush=True,
            )
            results.append(row)

    xapi_experiments = [
        ("holdout", "adasyn", 100),
        ("holdout", "smote", 100),
        ("best_fold", "adasyn", 100),
        ("holdout", "class_weight", 100),
    ]
    raw_xapi = raws["xapi"]
    y_xapi = xapi_y(raw_xapi)
    xapi_features = paper_feature_columns("xapi", raw_xapi, pearson_results)
    for eval_mode, oversampling, default_epochs in xapi_experiments:
        epochs = min(replication_epochs, default_epochs)
        print(f"Running xapi {eval_mode}+{oversampling} epochs={epochs}", flush=True)
        if eval_mode == "holdout":
            row = run_holdout_replication(
                "xapi",
                raw_xapi,
                y_xapi,
                xapi_features,
                binning="xapi",
                oversampling=oversampling,
                epochs=epochs,
                seed=seed,
            )
        else:
            row = run_cv_replication(
                "xapi",
                raw_xapi,
                y_xapi,
                xapi_features,
                binning="xapi",
                oversampling=oversampling,
                epochs=epochs,
                seed=seed,
                eval_mode=eval_mode,
            )
        print(f"RESULT xapi {eval_mode}+{oversampling}: acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f}", flush=True)
        results.append(row)
    return results


def best_replication_by_dataset(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        dataset = row["dataset"]
        current = best.get(dataset)
        if current is None or (row["f1_macro"], row["accuracy"]) > (current["f1_macro"], current["accuracy"]):
            best[dataset] = row
    return best


def build_default_split_for_dataset(
    dataset: str,
    raws: dict[str, pd.DataFrame],
    best_binning: str,
    pearson_results: dict[str, Any],
    seed: int,
) -> tuple[SplitData, str]:
    raw = raws[dataset]
    if dataset in STUDENT_DATASETS:
        y = student_y(raw, dataset, best_binning)
        binning = best_binning
    else:
        y = xapi_y(raw)
        binning = "xapi"
    feature_columns = paper_feature_columns(dataset, raw, pearson_results)
    return prepare_split(raw, dataset, feature_columns, y, seed=seed), binning


def run_improved_default(
    raws: dict[str, pd.DataFrame],
    best_binning: str,
    pearson_results: dict[str, Any],
    *,
    train_epochs: int,
    seed: int,
) -> list[dict[str, Any]]:
    print("\n=== STAGE 3: IMPROVED CNN-BiLSTM DEFAULT ===", flush=True)
    rows = []
    for dataset in ALL_DATASETS:
        split, binning = build_default_split_for_dataset(dataset, raws, best_binning, pearson_results, seed)
        dtype = model_dataset_type(dataset)

        def factory() -> nn.Module:
            return ImprovedCNNBiLSTM(
                dtype,
                split.n_features,
                split.n_classes,
                conv_filters=64,
                bilstm_hidden=128,
                dense_hidden=128,
                dropout=0.3,
                use_attention=(dataset != "xapi"),
            )

        oversampling = "adasyn" if dataset == "xapi" else "smote"
        result = train_model_once(
            factory,
            split,
            seed=seed,
            epochs=train_epochs,
            patience=15,
            batch_size=16 if dataset == "xapi" else 32,
            lr=1e-3,
            weight_decay=1e-4,
            oversampling=oversampling,
            label_smoothing=0.1,
            grad_clip=1.0,
            use_cosine=True,
        )
        row = {
            "dataset": dataset,
            "stage": "improved_default",
            "model": "ImprovedCNNBiLSTM",
            "binning": binning,
            "oversampling": oversampling,
            "use_attention": dataset != "xapi",
            "epochs": train_epochs,
            "best_epoch": result.best_epoch,
            "best_val_f1": result.best_val_f1,
            "best_val_accuracy": result.best_val_accuracy,
            **result.metrics,
        }
        rows.append(row)
        print(f"IMPROVED DEFAULT {dataset}: acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f}", flush=True)
    return rows


def optuna_objective_factory(
    dataset: str,
    split: SplitData,
    oversampling: str,
    *,
    seed: int,
    tune_epochs: int,
):
    def objective(trial: Any) -> float:
        params = {
            "conv_filters": trial.suggest_categorical("conv_filters", [32, 64, 128]),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
            "num_conv_layers": trial.suggest_int("num_conv_layers", 2, 4),
            "bilstm_hidden": trial.suggest_categorical("bilstm_hidden", [64, 128, 256]),
            "num_bilstm_layers": trial.suggest_int("num_bilstm_layers", 1, 2),
            "dense_hidden": trial.suggest_categorical("dense_hidden", [64, 128, 256]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.05),
            "use_attention": trial.suggest_categorical("use_attention", [True, False] if dataset != "xapi" else [False]),
        }
        dtype = model_dataset_type(dataset)

        def factory() -> nn.Module:
            return ImprovedCNNBiLSTM(
                dtype,
                split.n_features,
                split.n_classes,
                conv_filters=params["conv_filters"],
                kernel_size=params["kernel_size"],
                num_conv_layers=params["num_conv_layers"],
                bilstm_hidden=params["bilstm_hidden"],
                num_bilstm_layers=params["num_bilstm_layers"],
                dense_hidden=params["dense_hidden"],
                dropout=params["dropout"],
                use_attention=params["use_attention"],
            )

        result = train_model_once(
            factory,
            split,
            seed=seed + trial.number,
            epochs=tune_epochs,
            patience=10,
            batch_size=params["batch_size"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            oversampling=oversampling,
            label_smoothing=params["label_smoothing"],
            grad_clip=1.0,
            use_cosine=True,
        )
        trial.set_user_attr("test_accuracy", result.metrics["accuracy"])
        trial.set_user_attr("test_f1_macro", result.metrics["f1_macro"])
        trial.set_user_attr("best_epoch", result.best_epoch)
        return result.best_val_f1

    return objective


def run_improved_optuna_and_ensemble(
    raws: dict[str, pd.DataFrame],
    best_binning: str,
    pearson_results: dict[str, Any],
    *,
    trials: int,
    tune_epochs: int,
    train_epochs: int,
    ensemble_seeds: list[int],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print("\n=== STAGE 4: OPTUNA + ENSEMBLE ===", flush=True)
    try:
        import optuna
    except Exception as exc:
        raise RuntimeError("Optuna is required for stage 4.") from exc

    optuna_payload: dict[str, Any] = {}
    final_rows: list[dict[str, Any]] = []
    for dataset in ALL_DATASETS:
        split, binning = build_default_split_for_dataset(dataset, raws, best_binning, pearson_results, seed)
        oversampling = "adasyn" if dataset == "xapi" else "smote"
        print(f"Optuna start {dataset}: trials={trials}, tune_epochs={tune_epochs}", flush=True)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=min(10, max(1, trials // 5))),
        )
        study.optimize(optuna_objective_factory(dataset, split, oversampling, seed=seed, tune_epochs=tune_epochs), n_trials=trials)
        study_path = ROOT_OPTUNA_DIR / f"study_{dataset}.pkl"
        with study_path.open("wb") as handle:
            pickle.dump(study, handle)
        best_params = dict(study.best_params)
        optuna_payload[dataset] = {
            "best_value": float(study.best_value),
            "best_params": best_params,
            "study_path": str(study_path),
            "trials": trials,
            "tune_epochs": tune_epochs,
        }
        print(f"Optuna best {dataset}: val_f1={study.best_value:.4f} params={best_params}", flush=True)

        final_result, split = train_final_improved(dataset, split, best_params, oversampling, train_epochs=train_epochs, seed=seed)
        model_path = ROOT_MODELS_DIR / f"best_{dataset}.pt"
        torch.save(
            {
                "state_dict": final_result.state_dict,
                "dataset": dataset,
                "params": best_params,
                "binning": binning,
                "metrics": final_result.metrics,
                "best_epoch": final_result.best_epoch,
            },
            model_path,
        )
        final_row = {
            "dataset": dataset,
            "stage": "improved_optuna",
            "model": "ImprovedCNNBiLSTM",
            "binning": binning,
            "oversampling": oversampling,
            "epochs": train_epochs,
            "best_epoch": final_result.best_epoch,
            "best_val_f1": final_result.best_val_f1,
            "best_val_accuracy": final_result.best_val_accuracy,
            "model_path": str(model_path),
            **final_result.metrics,
        }
        final_rows.append(final_row)
        print(f"IMPROVED+OPTUNA {dataset}: acc={final_row['accuracy']:.4f} f1={final_row['f1_macro']:.4f}", flush=True)

        ensemble_row = run_ensemble(
            dataset,
            split,
            best_params,
            oversampling,
            seeds=ensemble_seeds,
            train_epochs=train_epochs,
        )
        final_rows.append(ensemble_row)
        print(f"ENSEMBLE {dataset}: acc={ensemble_row['accuracy']:.4f} f1={ensemble_row['f1_macro']:.4f}", flush=True)
    return optuna_payload, final_rows


def train_final_improved(
    dataset: str,
    split: SplitData,
    params: dict[str, Any],
    oversampling: str,
    *,
    train_epochs: int,
    seed: int,
) -> tuple[TrainResult, SplitData]:
    dtype = model_dataset_type(dataset)
    params = dict(params)
    batch_size = int(params.pop("batch_size"))
    lr = float(params.pop("lr"))
    weight_decay = float(params.pop("weight_decay"))
    label_smoothing = float(params.pop("label_smoothing"))
    if dataset == "xapi":
        params["use_attention"] = False

    def factory() -> nn.Module:
        return ImprovedCNNBiLSTM(dtype, split.n_features, split.n_classes, **params)

    result = train_model_once(
        factory,
        split,
        seed=seed,
        epochs=train_epochs,
        patience=15,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        oversampling=oversampling,
        label_smoothing=label_smoothing,
        grad_clip=1.0,
        use_cosine=True,
    )
    return result, split


def run_ensemble(
    dataset: str,
    split: SplitData,
    params: dict[str, Any],
    oversampling: str,
    *,
    seeds: list[int],
    train_epochs: int,
) -> dict[str, Any]:
    all_probs = []
    seed_metrics = []
    for seed in seeds:
        result, _ = train_final_improved(dataset, split, params, oversampling, train_epochs=train_epochs, seed=seed)
        all_probs.append(result.probabilities)
        seed_metrics.append({"seed": seed, **result.metrics, "best_epoch": result.best_epoch})
        print(
            f"  ensemble seed={seed} {dataset}: acc={result.metrics['accuracy']:.4f} f1={result.metrics['f1_macro']:.4f}",
            flush=True,
        )
    avg_probs = np.mean(all_probs, axis=0)
    y_pred = avg_probs.argmax(axis=1)
    metrics = metric_dict(split.y_test, y_pred)
    return {
        "dataset": dataset,
        "stage": "ensemble",
        "model": "ImprovedCNNBiLSTM softmax average",
        "n_seeds": len(seeds),
        "seeds": seeds,
        "seed_metrics": seed_metrics,
        **metrics,
    }


def final_table(all_results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_rep = best_replication_by_dataset(all_results["stage2_replication"])
    candidates = all_results["stage3_improved"] + all_results["stage4_final"]
    for dataset in ALL_DATASETS:
        bench = PAPER_BENCHMARKS[dataset]
        rows.append(
            {
                "dataset": dataset,
                "model": "Paper benchmark",
                "accuracy": bench.get("accuracy"),
                "precision_macro": bench.get("precision_macro"),
                "recall_macro": bench.get("recall_macro"),
                "f1_macro": bench.get("f1_macro"),
            }
        )
        if dataset in best_rep:
            rows.append({"dataset": dataset, "model": "Best replication", **metric_subset(best_rep[dataset])})
        for stage_name in ("improved_optuna", "ensemble"):
            stage_rows = [row for row in candidates if row["dataset"] == dataset and row["stage"] == stage_name]
            if stage_rows:
                best = sorted(stage_rows, key=lambda row: (row["f1_macro"], row["accuracy"]), reverse=True)[0]
                rows.append({"dataset": dataset, "model": stage_name, **metric_subset(best)})
    return rows


def metric_subset(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": row.get("accuracy"),
        "precision_macro": row.get("precision_macro"),
        "recall_macro": row.get("recall_macro"),
        "f1_macro": row.get("f1_macro"),
    }


def print_final_report(all_results: dict[str, Any]) -> None:
    rows = final_table(all_results)
    print("\n=== FINAL RESULTS - CNN-BiLSTM Experiment Report ===", flush=True)
    print(f"{'Dataset':12s} {'Model':24s} {'Acc':>8s} {'Prec':>8s} {'Recall':>8s} {'F1-Mac':>8s}", flush=True)
    print("-" * 76, flush=True)
    for row in rows:
        print(
            f"{row['dataset']:12s} {row['model']:24s} "
            f"{fmt(row.get('accuracy')):>8s} {fmt(row.get('precision_macro')):>8s} "
            f"{fmt(row.get('recall_macro')):>8s} {fmt(row.get('f1_macro')):>8s}",
            flush=True,
        )
    print("\nBest binning found:", all_results["stage1"]["binning"]["selected_binning"], flush=True)
    best_rep = best_replication_by_dataset(all_results["stage2_replication"])
    for dataset, row in best_rep.items():
        print(
            f"Best replication {dataset}: binning={row.get('binning')} eval={row.get('eval_mode')} "
            f"oversampling={row.get('oversampling')} acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f}",
            flush=True,
        )
    print("\nOptuna best params per dataset:", flush=True)
    for dataset, payload in all_results["stage4_optuna"].items():
        print(f"  {dataset}: {payload['best_params']}", flush=True)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass
    return f"{float(value):.4f}"


def load_raws() -> dict[str, pd.DataFrame]:
    return {dataset: read_raw(DATASETS[dataset]) for dataset in ALL_DATASETS}


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    ensure_advanced_dirs()
    set_seed(args.seed)
    raws = load_raws()
    all_results: dict[str, Any] = {
        "config": vars(args),
        "benchmarks": PAPER_BENCHMARKS,
        "stage1": {},
        "stage2_replication": [],
        "stage3_improved": [],
        "stage4_optuna": {},
        "stage4_final": [],
    }

    binning = test_all_binnings(raws["student-mat"], raws["student-por"])
    pearson = {
        "student-mat": verify_pearson_selection(raws["student-mat"], "student-mat", binning["selected_binning"]),
        "student-por": verify_pearson_selection(raws["student-por"], "student-por", binning["selected_binning"]),
    }
    xapi_split = verify_xapi_split(raws["xapi"], seed=args.seed)
    all_results["stage1"] = {"binning": binning, "pearson": pearson, "xapi_split": xapi_split}
    save_json(ALL_EXPERIMENTS_PATH, all_results)

    all_results["stage2_replication"] = run_stage2_replication(
        raws,
        binning["selected_binning"],
        pearson,
        replication_epochs=args.replication_epochs,
        seed=args.seed,
    )
    save_json(ALL_EXPERIMENTS_PATH, all_results)

    all_results["stage3_improved"] = run_improved_default(
        raws,
        binning["selected_binning"],
        pearson,
        train_epochs=args.train_epochs,
        seed=args.seed,
    )
    save_json(ALL_EXPERIMENTS_PATH, all_results)

    optuna_payload, final_rows = run_improved_optuna_and_ensemble(
        raws,
        binning["selected_binning"],
        pearson,
        trials=args.trials,
        tune_epochs=args.tune_epochs,
        train_epochs=args.train_epochs,
        ensemble_seeds=[int(value.strip()) for value in args.ensemble_seeds.split(",") if value.strip()],
        seed=args.seed,
    )
    all_results["stage4_optuna"] = optuna_payload
    all_results["stage4_final"] = final_rows
    all_results["final_table"] = final_table(all_results)
    save_json(ALL_EXPERIMENTS_PATH, all_results)
    print_final_report(all_results)
    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced CNN-BiLSTM replication/improvement experiments.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replication-epochs", type=int, default=50)
    parser.add_argument("--tune-epochs", type=int, default=50)
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--ensemble-seeds", default="42,123,456,789,2024")
    return parser.parse_args()


def main() -> None:
    run_all(parse_args())


if __name__ == "__main__":
    main()

