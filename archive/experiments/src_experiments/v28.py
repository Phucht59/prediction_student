from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader

from src.config import DATASETS, DEFAULT_SEED, RAW_DIR, REPORTS_DIR
from src.data_pipeline import (
    DataPreprocessor,
    FeatureSelector,
    apply_feature_engineering,
    create_and_save_locked_test,
    get_sequence_columns,
    load_splits,
)
from src.experiments.common import (
    ExperimentConfig,
    TechnicalStudentDataset,
    apply_student_scenario,
    compute_required_metrics,
    save_json,
    scenario_sequence_columns,
)
from src.losses_v27 import ClassBalancedFocalLoss, FocalLoss, OrdinalLoss
from src.train_pipeline import calculate_class_weights
from src.utils import set_seed, setup_logger

logger = setup_logger("v28_experiments")

V28_DIR = REPORTS_DIR / "v28"
REQUIRED_TASKS = (
    ("student-mat", "late"),
    ("student-por", "late"),
    ("student-por", "midterm"),
    ("xapi", "xapi"),
)
OPTIONAL_TASKS = (
    ("student-mat", "midterm"),
    ("student-mat", "early"),
    ("student-por", "early"),
)
SEED_ENSEMBLE_5 = (42, 123, 155, 156, 2025)
SEED_ENSEMBLE_11 = (42, 123, 155, 156, 2025, 7, 99, 200, 300, 500, 1337)


@dataclass(frozen=True)
class V28RunConfig:
    seed: int = DEFAULT_SEED
    cv_folds: int = 2
    max_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 6
    ensemble_seeds: tuple[int, ...] = SEED_ENSEMBLE_5
    use_optional_tasks: bool = False
    use_feature_selection: bool = True
    smote_ratio: float = 1.0
    resampling_k_neighbors: int = 5


@dataclass(frozen=True)
class V28Candidate:
    candidate_id: str
    variant: str
    model_family: str
    cnn_kernel_size: int
    cnn_channels: int
    lstm_hidden_dim: int
    pooling: str
    loss_name: str
    imbalance_strategy: str
    dropout: float = 0.20
    context_hidden_dim: int = 64
    ordinal_weight: float = 0.30


@dataclass
class V28Prepared:
    train: pd.DataFrame
    validation: pd.DataFrame
    preprocessor: DataPreprocessor
    selector: FeatureSelector
    sequence_cols: list[str]
    kind: str
    target_col: str


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(8, hidden_dim // 2), 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(sequence), dim=1)
        return torch.sum(sequence * weights, dim=1)


class SequenceCNNBiLSTMV28(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cnn_kernel_size: int,
        cnn_channels: int,
        lstm_hidden_dim: int,
        pooling: str,
        dropout: float,
        ordinal_head: bool = False,
    ):
        super().__init__()
        self.pooling = pooling
        self.has_ordinal_head = ordinal_head
        self.sequence_cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.sequence_norm = nn.LayerNorm(cnn_channels)
        self.sequence_bilstm = nn.LSTM(cnn_channels, lstm_hidden_dim, batch_first=True, bidirectional=True)
        out_dim = lstm_hidden_dim * 2
        self.attention = AttentionPooling(out_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )
        self.ordinal_head = nn.Linear(out_dim, num_classes - 1) if ordinal_head else None

    def _pool(self, sequence: torch.Tensor) -> torch.Tensor:
        if self.pooling == "last":
            return sequence[:, -1, :]
        return self.attention(sequence)

    def forward(self, seq_x, num_x=None, cat_x=None):
        seq = self.sequence_cnn(seq_x.float().transpose(1, 2)).transpose(1, 2)
        seq = self.sequence_norm(seq)
        seq, _ = self.sequence_bilstm(seq)
        pooled = self._pool(seq)
        logits = self.classifier(pooled)
        ordinal_logits = self.ordinal_head(pooled) if self.ordinal_head is not None else None
        return logits, ordinal_logits


class GatedFusionV28(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_numerical: int,
        cat_cardinalities: list[int],
        cnn_kernel_size: int,
        cnn_channels: int,
        lstm_hidden_dim: int,
        pooling: str,
        context_hidden_dim: int,
        dropout: float,
        ordinal_head: bool = False,
    ):
        super().__init__()
        self.sequence = SequenceCNNBiLSTMV28(
            num_classes=num_classes,
            cnn_kernel_size=cnn_kernel_size,
            cnn_channels=cnn_channels,
            lstm_hidden_dim=lstm_hidden_dim,
            pooling=pooling,
            dropout=dropout,
            ordinal_head=False,
        )
        seq_dim = lstm_hidden_dim * 2
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities
        self.embeddings = nn.ModuleList()
        embedding_total_dim = 0
        for cardinality in cat_cardinalities:
            dim = max(2, min(32, (cardinality + 1) // 2))
            self.embeddings.append(nn.Embedding(cardinality, dim))
            embedding_total_dim += dim
        context_input_dim = max(1, num_numerical + embedding_total_dim)
        self.context = nn.Sequential(
            nn.Linear(context_input_dim, context_hidden_dim),
            nn.LayerNorm(context_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_hidden_dim, context_hidden_dim),
            nn.ReLU(),
        )
        self.seq_proj = nn.Linear(seq_dim, context_hidden_dim)
        self.gate = nn.Sequential(nn.Linear(seq_dim + context_hidden_dim, context_hidden_dim), nn.Sigmoid())
        self.classifier = nn.Linear(context_hidden_dim, num_classes)
        self.ordinal_head = nn.Linear(context_hidden_dim, num_classes - 1) if ordinal_head else None

    def _context_input(self, num_x: torch.Tensor, cat_x: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.num_numerical > 0:
            parts.append(num_x[:, : self.num_numerical].float())
        if self.cat_cardinalities:
            embedded = []
            for idx, layer in enumerate(self.embeddings):
                values = torch.clamp(cat_x[:, idx].long(), 0, self.cat_cardinalities[idx] - 1)
                embedded.append(layer(values))
            parts.append(torch.cat(embedded, dim=1))
        if not parts:
            return torch.zeros((num_x.shape[0], 1), device=num_x.device)
        return torch.cat(parts, dim=1)

    def forward(self, seq_x, num_x, cat_x):
        seq = self.sequence.sequence_cnn(seq_x.float().transpose(1, 2)).transpose(1, 2)
        seq = self.sequence.sequence_norm(seq)
        seq, _ = self.sequence.sequence_bilstm(seq)
        seq_vec = self.sequence._pool(seq)
        ctx_vec = self.context(self._context_input(num_x, cat_x))
        gate = self.gate(torch.cat([seq_vec, ctx_vec], dim=1))
        fused = gate * self.seq_proj(seq_vec) + (1.0 - gate) * ctx_vec
        logits = self.classifier(fused)
        ordinal_logits = self.ordinal_head(fused) if self.ordinal_head is not None else None
        return logits, ordinal_logits


def ensure_v28_dir() -> None:
    V28_DIR.mkdir(parents=True, exist_ok=True)


def _load_or_create_splits(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return load_splits(dataset_name, "3class")
    except FileNotFoundError:
        spec = DATASETS[dataset_name]
        raw = pd.read_csv(RAW_DIR / spec.raw_file, sep=spec.csv_sep)
        create_and_save_locked_test(raw, dataset_name, "3class")
        return load_splits(dataset_name, "3class")


def _apply_scenario(df: pd.DataFrame, dataset_name: str, scenario: str) -> pd.DataFrame:
    spec = DATASETS[dataset_name]
    if spec.kind == "student":
        return apply_student_scenario(df, scenario)
    return apply_feature_engineering(df.copy(), spec.kind)


def _sequence_columns(dataset_name: str, scenario: str) -> list[str]:
    spec = DATASETS[dataset_name]
    if spec.kind == "student":
        return scenario_sequence_columns(scenario)
    return get_sequence_columns(spec.kind)


def _oversample_method(imbalance_strategy: str) -> str:
    if imbalance_strategy.startswith("smotenc"):
        return "smotenc"
    if imbalance_strategy.startswith("random_oversampling"):
        return "random_oversampling"
    return "none"


def prepare_v28_fold(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    dataset_name: str,
    scenario: str,
    candidate: V28Candidate,
    config: V28RunConfig,
) -> V28Prepared:
    spec = DATASETS[dataset_name]
    train_scenario = _apply_scenario(train_df, dataset_name, scenario)
    validation_scenario = _apply_scenario(validation_df, dataset_name, scenario)
    preprocessor = DataPreprocessor(
        target_col=spec.target_col,
        oversample_method=_oversample_method(candidate.imbalance_strategy),
        smote_ratio=config.smote_ratio,
        resampling_k_neighbors=config.resampling_k_neighbors,
    )
    train_prep = preprocessor.fit_transform(train_scenario, apply_oversampling=False)
    validation_prep = preprocessor.transform(validation_scenario)
    sequence_cols = _sequence_columns(dataset_name, scenario)
    selector = FeatureSelector(
        target_col=spec.target_col,
        use_feature_selection=config.use_feature_selection,
        required_features=sequence_cols,
    )
    train_selected = selector.fit_transform(train_prep, preprocessor.numerical_cols, preprocessor.categorical_cols)
    validation_selected = selector.transform(validation_prep)
    if _oversample_method(candidate.imbalance_strategy) != "none":
        train_selected = preprocessor.apply_oversampling(train_selected)
    return V28Prepared(
        train=train_selected,
        validation=validation_selected,
        preprocessor=preprocessor,
        selector=selector,
        sequence_cols=sequence_cols,
        kind=spec.kind,
        target_col=spec.target_col,
    )


def transform_v28(df: pd.DataFrame, prepared: V28Prepared, dataset_name: str, scenario: str) -> pd.DataFrame:
    scenario_df = _apply_scenario(df, dataset_name, scenario)
    prepped = prepared.preprocessor.transform(scenario_df)
    return prepared.selector.transform(prepped)


def make_dataset(frame: pd.DataFrame, prepared: V28Prepared) -> TechnicalStudentDataset:
    return TechnicalStudentDataset(
        frame,
        target_col=prepared.target_col,
        numerical_cols=prepared.preprocessor.numerical_cols,
        categorical_cols=prepared.preprocessor.categorical_cols,
        sequence_cols=prepared.sequence_cols,
    )


def make_model(candidate: V28Candidate, dataset: TechnicalStudentDataset, prepared: V28Prepared) -> nn.Module:
    ordinal = candidate.variant == "sequence_cnn_bilstm_v28_ordinal"
    if candidate.variant == "gated_fusion_v28":
        cat_cardinalities = [len(prepared.preprocessor.label_encoders[column].classes_) for column in dataset.cat_cols]
        return GatedFusionV28(
            num_classes=3,
            num_numerical=len(dataset.num_cols),
            cat_cardinalities=cat_cardinalities,
            cnn_kernel_size=candidate.cnn_kernel_size,
            cnn_channels=candidate.cnn_channels,
            lstm_hidden_dim=candidate.lstm_hidden_dim,
            pooling=candidate.pooling,
            context_hidden_dim=candidate.context_hidden_dim,
            dropout=candidate.dropout,
            ordinal_head=ordinal,
        )
    return SequenceCNNBiLSTMV28(
        num_classes=3,
        cnn_kernel_size=candidate.cnn_kernel_size,
        cnn_channels=candidate.cnn_channels,
        lstm_hidden_dim=candidate.lstm_hidden_dim,
        pooling=candidate.pooling,
        dropout=candidate.dropout,
        ordinal_head=ordinal,
    )


def candidate_grid(dataset_name: str, scenario: str) -> list[V28Candidate]:
    spec = DATASETS[dataset_name]
    if spec.kind == "student" and scenario == "early":
        return []
    kernels = (2, 3) if dataset_name == "xapi" else (1, 2)
    primary_kernel = kernels[0]
    alternate_kernel = kernels[-1]
    return [
        V28Candidate(f"seq_k{primary_kernel}_c16_h32_last_cw", "sequence_cnn_bilstm_v28", "sequence", primary_kernel, 16, 32, "last", "class_weight", "none_class_weight"),
        V28Candidate(f"seq_k{primary_kernel}_c32_h64_attn_cw", "sequence_cnn_bilstm_v28", "sequence", primary_kernel, 32, 64, "attention", "class_weight", "none_class_weight"),
        V28Candidate(f"seq_k{alternate_kernel}_c32_h64_attn_focal", "sequence_cnn_bilstm_v28_focal", "sequence", alternate_kernel, 32, 64, "attention", "focal_loss_class_weight", "focal_loss_class_weight"),
        V28Candidate(f"seq_k{alternate_kernel}_c64_h96_attn_cbf", "sequence_cnn_bilstm_v28_focal", "sequence", alternate_kernel, 64, 96, "attention", "class_balanced_focal_loss", "class_balanced_focal_loss"),
        V28Candidate(f"seq_k{alternate_kernel}_c32_h64_last_random_focal", "sequence_cnn_bilstm_v28_focal", "sequence", alternate_kernel, 32, 64, "last", "focal_loss_class_weight", "random_oversampling_focal_loss"),
        V28Candidate(f"seq_k{primary_kernel}_c32_h64_attn_smotenc_cw", "sequence_cnn_bilstm_v28", "sequence", primary_kernel, 32, 64, "attention", "class_weight", "smotenc_class_weight"),
        V28Candidate(f"seq_k{alternate_kernel}_c32_h64_attn_smotenc_focal", "sequence_cnn_bilstm_v28_focal", "sequence", alternate_kernel, 32, 64, "attention", "focal_loss_class_weight", "smotenc_focal_loss"),
        V28Candidate(f"seq_k{alternate_kernel}_c64_h64_attn_ordinal", "sequence_cnn_bilstm_v28_ordinal", "sequence", alternate_kernel, 64, 64, "attention", "ordinal_class_weight", "none_class_weight"),
        V28Candidate(f"gated_k{alternate_kernel}_c32_h64_attn_cw", "gated_fusion_v28", "fusion", alternate_kernel, 32, 64, "attention", "class_weight", "none_class_weight"),
    ]


def class_loss(candidate: V28Candidate, labels: np.ndarray, device: torch.device) -> Callable:
    if candidate.loss_name == "class_balanced_focal_loss":
        counts = np.bincount(labels.astype(int), minlength=3)
        return ClassBalancedFocalLoss(counts, beta=0.99, gamma=2.0).to(device)
    weights = calculate_class_weights(labels, num_classes=3).to(device)
    if candidate.loss_name == "focal_loss_class_weight":
        return FocalLoss(weight=weights, gamma=2.0).to(device)
    return nn.CrossEntropyLoss(weight=weights)


def ordinal_targets(labels: torch.Tensor, num_thresholds: int = 2) -> torch.Tensor:
    targets = torch.zeros((labels.shape[0], num_thresholds), device=labels.device)
    for idx in range(num_thresholds):
        targets[:, idx] = (labels > idx).float()
    return targets


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    candidate: V28Candidate,
    original_train_labels: np.ndarray,
    config: V28RunConfig,
    device: torch.device,
) -> nn.Module:
    ce_loss = class_loss(candidate, original_train_labels, device)
    ord_loss = OrdinalLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = None
    best_f1 = -1.0
    stale = 0
    for _ in range(config.max_epochs):
        model.train()
        for seq_x, num_x, cat_x, labels, _, _ in train_loader:
            seq_x = seq_x.to(device)
            num_x = num_x.to(device)
            cat_x = cat_x.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, ordinal_logits = model(seq_x, num_x, cat_x)
            loss = ce_loss(logits, labels)
            if ordinal_logits is not None:
                loss = loss + candidate.ordinal_weight * ord_loss(ordinal_logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        f1 = validate_macro_f1(model, val_loader, device)
        if f1 > best_f1 + 1e-6:
            best_f1 = f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def validate_macro_f1(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    probs, labels, _ = predict_proba(model, loader, device)
    return float(f1_score(labels, np.argmax(probs, axis=1), average="macro", zero_division=0))


def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    reg_true: list[np.ndarray] = []
    with torch.no_grad():
        for seq_x, num_x, cat_x, y, _, y_reg in loader:
            logits, _ = model(seq_x.to(device), num_x.to(device), cat_x.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(y.numpy())
            reg_true.append(y_reg.numpy())
    return np.vstack(probs), np.concatenate(labels), np.concatenate(reg_true)


def predict_with_threshold(probabilities: np.ndarray, threshold: float | None) -> np.ndarray:
    if threshold is None or math.isnan(float(threshold)):
        return np.argmax(probabilities, axis=1)
    non_low = np.argmax(probabilities[:, 1:], axis=1) + 1
    return np.where(probabilities[:, 0] >= float(threshold), 0, non_low)


def tune_thresholds(y_true: np.ndarray, probabilities: np.ndarray, reg_true: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    argmax_preds = np.argmax(probabilities, axis=1)
    rows.append({"prediction_mode": "argmax", "threshold_low": np.nan, "selection_score": 0.0, **compute_required_metrics(y_true, argmax_preds, probabilities, y_reg_true=reg_true)})
    objectives = {
        "low_f1_tuned": lambda m: m["f1_low"] + 1e-3 * m["macro_f1"],
        "low_recall_priority": lambda m: 0.80 * m["recall_low"] + 0.20 * m["f1_low"],
        "balanced_low_macro": lambda m: 0.50 * m["macro_f1"] + 0.25 * m["recall_low"] + 0.25 * m["f1_low"],
    }
    for mode, objective in objectives.items():
        best = None
        for threshold in np.linspace(0.05, 0.95, 37):
            preds = predict_with_threshold(probabilities, float(threshold))
            metrics = compute_required_metrics(y_true, preds, probabilities, y_reg_true=reg_true)
            score = float(objective(metrics))
            row = {"prediction_mode": mode, "threshold_low": float(threshold), "selection_score": score, **metrics}
            if best is None or score > best["selection_score"]:
                best = row
        rows.append(best)
    return rows


def rank_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["macro_f1", "recall_low", "f1_low"], ascending=[False, False, False])


def train_candidate_fold(
    train_pool: pd.DataFrame,
    dataset_name: str,
    scenario: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    candidate: V28Candidate,
    config: V28RunConfig,
    fold_seed: int,
    device: torch.device,
):
    spec = DATASETS[dataset_name]
    prepared = prepare_v28_fold(
        train_pool.iloc[train_idx].copy(),
        train_pool.iloc[val_idx].copy(),
        dataset_name,
        scenario,
        candidate,
        config,
    )
    train_ds = make_dataset(prepared.train, prepared)
    val_ds = make_dataset(prepared.validation, prepared)
    drop_last = len(train_ds) > config.batch_size and len(train_ds) % config.batch_size == 1
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    set_seed(fold_seed)
    model = make_model(candidate, train_ds, prepared).to(device)
    original_labels = train_pool.iloc[train_idx][spec.target_col].astype(int).to_numpy()
    model = train_model(model, train_loader, val_loader, candidate, original_labels, config, device)
    probs, labels, reg_true = predict_proba(model, val_loader, device)
    return probs, labels, reg_true


def run_candidate_cv(
    dataset_name: str,
    scenario: str,
    candidate: V28Candidate,
    config: V28RunConfig,
    train_pool: pd.DataFrame,
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    spec = DATASETS[dataset_name]
    labels = train_pool[spec.target_col].astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    oof_probs = np.zeros((len(train_pool), 3), dtype=float)
    oof_targets = np.zeros(len(train_pool), dtype=int)
    oof_reg_true = np.zeros(len(train_pool), dtype=float)
    fold_rows = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_pool, labels), start=1):
        probs, y_val, reg_true = train_candidate_fold(
            train_pool,
            dataset_name,
            scenario,
            train_idx,
            val_idx,
            candidate,
            config,
            config.seed + fold,
            device,
        )
        oof_probs[val_idx] = probs
        oof_targets[val_idx] = y_val
        oof_reg_true[val_idx] = reg_true
        preds = np.argmax(probs, axis=1)
        fold_rows.append(
            {
                "dataset": dataset_name,
                "scenario": scenario,
                "candidate_id": candidate.candidate_id,
                "variant": candidate.variant,
                "model_family": candidate.model_family,
                "fold": fold,
                **asdict(candidate),
                **compute_required_metrics(y_val, preds, probs, y_reg_true=reg_true),
            }
        )
    threshold_rows = []
    for row in tune_thresholds(oof_targets, oof_probs, oof_reg_true):
        threshold_rows.append(
            {
                "dataset": dataset_name,
                "scenario": scenario,
                "candidate_id": candidate.candidate_id,
                "variant": candidate.variant,
                "model_family": candidate.model_family,
                **asdict(candidate),
                **row,
            }
        )
    return fold_rows, threshold_rows


def evaluate_locked_candidate(
    dataset_name: str,
    scenario: str,
    candidate: V28Candidate,
    threshold_row: dict,
    config: V28RunConfig,
    train_pool: pd.DataFrame,
    locked_test: pd.DataFrame,
    seed: int,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    spec = DATASETS[dataset_name]
    prepared = prepare_v28_fold(train_pool.copy(), locked_test.copy(), dataset_name, scenario, candidate, config)
    locked_selected = transform_v28(locked_test.copy(), prepared, dataset_name, scenario)
    train_ds = make_dataset(prepared.train, prepared)
    locked_ds = make_dataset(locked_selected, prepared)
    drop_last = len(train_ds) > config.batch_size and len(train_ds) % config.batch_size == 1
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=drop_last)
    train_eval_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False)
    locked_loader = DataLoader(locked_ds, batch_size=config.batch_size, shuffle=False)
    set_seed(seed)
    model = make_model(candidate, train_ds, prepared).to(device)
    original_labels = train_pool[spec.target_col].astype(int).to_numpy()
    model = train_model(model, train_loader, train_eval_loader, candidate, original_labels, config, device)
    probs, y_true, reg_true = predict_proba(model, locked_loader, device)
    preds = predict_with_threshold(probs, threshold_row["threshold_low"])
    metrics = compute_required_metrics(y_true, preds, probs, y_reg_true=reg_true)
    row = {
        "dataset": dataset_name,
        "scenario": scenario,
        "candidate_id": candidate.candidate_id,
        "variant": candidate.variant,
        "prediction_mode": threshold_row["prediction_mode"],
        "threshold_low": threshold_row["threshold_low"],
        "seed": seed,
        **asdict(candidate),
        **metrics,
    }
    return row, probs, y_true, reg_true


def evaluate_ensemble(
    dataset_name: str,
    scenario: str,
    candidate: V28Candidate,
    threshold_row: dict,
    config: V28RunConfig,
    train_pool: pd.DataFrame,
    locked_test: pd.DataFrame,
    device: torch.device,
) -> dict:
    member_probs = []
    y_true = None
    reg_true = None
    for seed in config.ensemble_seeds:
        _, probs, labels, regs = evaluate_locked_candidate(
            dataset_name,
            scenario,
            candidate,
            threshold_row,
            config,
            train_pool,
            locked_test,
            seed,
            device,
        )
        member_probs.append(probs)
        y_true = labels
        reg_true = regs
    mean_probs = np.mean(np.stack(member_probs, axis=0), axis=0)
    preds = predict_with_threshold(mean_probs, threshold_row["threshold_low"])
    metrics = compute_required_metrics(y_true, preds, mean_probs, y_reg_true=reg_true)
    return {
        "dataset": dataset_name,
        "scenario": scenario,
        "candidate_id": candidate.candidate_id,
        "variant": "sequence_cnn_bilstm_v28_ensemble" if candidate.model_family == "sequence" else f"{candidate.variant}_ensemble",
        "base_variant": candidate.variant,
        "prediction_mode": threshold_row["prediction_mode"],
        "threshold_low": threshold_row["threshold_low"],
        "ensemble_seeds": "|".join(str(seed) for seed in config.ensemble_seeds),
        **asdict(candidate),
        **metrics,
    }


def build_baseline_models(seed: int):
    models = {
        "logistic_regression": LogisticRegression(max_iter=300, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample", random_state=seed, n_jobs=1),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=250, early_stopping=True, random_state=seed),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=1,
        )
    except Exception:
        models["hist_gradient_boosting"] = HistGradientBoostingClassifier(max_iter=150, random_state=seed)
    return models


def split_xy(frame: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    drop_cols = [target_col]
    if "G3_raw" in frame.columns:
        drop_cols.append("G3_raw")
    return frame.drop(columns=drop_cols, errors="ignore"), frame[target_col].astype(int).to_numpy()


def fit_baseline(model, x_train: pd.DataFrame, y_train: np.ndarray):
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    try:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    except TypeError:
        model.fit(x_train, y_train)
    return model


def run_baseline_for_task(
    dataset_name: str,
    scenario: str,
    config: V28RunConfig,
    train_pool: pd.DataFrame,
    locked_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    spec = DATASETS[dataset_name]
    dummy = V28Candidate("baseline_preprocess", "baseline", "baseline", 1, 16, 32, "attention", "class_weight", "none_class_weight")
    labels = train_pool[spec.target_col].astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    cv_rows = []
    fallback_note = "xgboost" if "xgboost" in build_baseline_models(config.seed) else "hist_gradient_boosting"
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_pool, labels), start=1):
        prepared = prepare_v28_fold(
            train_pool.iloc[train_idx].copy(),
            train_pool.iloc[val_idx].copy(),
            dataset_name,
            scenario,
            dummy,
            config,
        )
        x_train, y_train = split_xy(prepared.train, spec.target_col)
        x_val, y_val = split_xy(prepared.validation, spec.target_col)
        reg_true = prepared.validation["G3_raw"].to_numpy(dtype=float) if "G3_raw" in prepared.validation else None
        for model_name, model in build_baseline_models(config.seed + fold).items():
            fitted = fit_baseline(model, x_train, y_train)
            preds = fitted.predict(x_val)
            probs = fitted.predict_proba(x_val) if hasattr(fitted, "predict_proba") else None
            cv_rows.append(
                {
                    "dataset": dataset_name,
                    "scenario": scenario,
                    "model": model_name,
                    "fold": fold,
                    **compute_required_metrics(y_val, preds, probs, y_reg_true=reg_true),
                }
            )
    cv_df = pd.DataFrame(cv_rows)
    cv_summary = cv_df.groupby(["dataset", "scenario", "model"], as_index=False).agg(
        macro_f1=("macro_f1", "mean"),
        recall_low=("recall_low", "mean"),
        f1_low=("f1_low", "mean"),
    )
    selected = rank_rows(cv_summary).iloc[0]
    prepared = prepare_v28_fold(train_pool.copy(), locked_test.copy(), dataset_name, scenario, dummy, config)
    locked_selected = transform_v28(locked_test.copy(), prepared, dataset_name, scenario)
    x_train, y_train = split_xy(prepared.train, spec.target_col)
    x_locked, y_locked = split_xy(locked_selected, spec.target_col)
    reg_true = locked_selected["G3_raw"].to_numpy(dtype=float) if "G3_raw" in locked_selected else None
    model = build_baseline_models(config.seed)[selected["model"]]
    fitted = fit_baseline(model, x_train, y_train)
    preds = fitted.predict(x_locked)
    probs = fitted.predict_proba(x_locked) if hasattr(fitted, "predict_proba") else None
    locked_df = pd.DataFrame([
        {
            "dataset": dataset_name,
            "scenario": scenario,
            "model": selected["model"],
            "selected_by": "cv_macro_f1_recall_low_f1_low",
            **compute_required_metrics(y_locked, preds, probs, y_reg_true=reg_true),
        }
    ])
    return cv_df, locked_df, fallback_note


def old_champion_rows() -> pd.DataFrame:
    path = REPORTS_DIR / "ablation" / "deep_ablation_locked_test.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    return rank_rows(frame).groupby(["dataset", "scenario"], as_index=False).head(1)


def run_v28_experiments(tasks: list[tuple[str, str]], config: V28RunConfig) -> dict[str, pd.DataFrame]:
    ensure_v28_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running V28 on device=%s config=%s", device, config)
    cv_rows: list[dict] = []
    threshold_rows: list[dict] = []
    locked_rows: list[dict] = []
    ensemble_rows: list[dict] = []
    selection_rows: list[dict] = []
    baseline_cv_frames = []
    baseline_locked_frames = []
    fallback_notes = set()

    for dataset_name, scenario in tasks:
        if dataset_name == "student-combine":
            raise ValueError("student-combine is not allowed in V28 experiments.")
        train_pool, locked_test = _load_or_create_splits(dataset_name)
        candidates = candidate_grid(dataset_name, scenario)
        if not candidates:
            logger.info("Skipping sequence V28 for %s/%s because no real sequence is available.", dataset_name, scenario)
        task_thresholds = []
        for candidate in candidates:
            logger.info("CV task=%s/%s candidate=%s", dataset_name, scenario, candidate.candidate_id)
            fold_rows, candidate_thresholds = run_candidate_cv(dataset_name, scenario, candidate, config, train_pool, device)
            cv_rows.extend(fold_rows)
            threshold_rows.extend(candidate_thresholds)
            task_thresholds.extend(candidate_thresholds)

        baseline_cv, baseline_locked, fallback_note = run_baseline_for_task(dataset_name, scenario, config, train_pool, locked_test)
        baseline_cv_frames.append(baseline_cv)
        baseline_locked_frames.append(baseline_locked)
        fallback_notes.add(fallback_note)

        if not task_thresholds:
            continue
        task_threshold_df = pd.DataFrame(task_thresholds)
        selected = rank_rows(task_threshold_df).iloc[0].to_dict()
        candidate_lookup = {candidate.candidate_id: candidate for candidate in candidates}
        selected_candidate = candidate_lookup[selected["candidate_id"]]

        best_sequence = rank_rows(task_threshold_df[task_threshold_df["model_family"] == "sequence"]).iloc[0].to_dict()
        if selected_candidate.model_family == "fusion":
            sequence_margin = selected["macro_f1"] - best_sequence["macro_f1"]
            if sequence_margin < 1e-9:
                selected = best_sequence
                selected_candidate = candidate_lookup[selected["candidate_id"]]

        selection_rows.append(
            {
                "dataset": dataset_name,
                "scenario": scenario,
                "selected_by": "oof_macro_f1_recall_low_f1_low",
                **{key: selected[key] for key in selected.keys() if key in {"candidate_id", "variant", "model_family", "prediction_mode", "threshold_low", "macro_f1", "recall_low", "f1_low"}},
            }
        )
        locked_row, _, _, _ = evaluate_locked_candidate(
            dataset_name,
            scenario,
            selected_candidate,
            selected,
            config,
            train_pool,
            locked_test,
            config.seed,
            device,
        )
        locked_rows.append(locked_row)
        if selected_candidate.model_family == "sequence":
            ensemble_rows.append(evaluate_ensemble(dataset_name, scenario, selected_candidate, selected, config, train_pool, locked_test, device))

    cv_df = pd.DataFrame(cv_rows)
    thresholds_df = pd.DataFrame(threshold_rows)
    locked_df = pd.DataFrame(locked_rows)
    ensemble_df = pd.DataFrame(ensemble_rows)
    selection_df = pd.DataFrame(selection_rows)
    baseline_cv_df = pd.concat(baseline_cv_frames, ignore_index=True) if baseline_cv_frames else pd.DataFrame()
    baseline_locked_df = pd.concat(baseline_locked_frames, ignore_index=True) if baseline_locked_frames else pd.DataFrame()

    vs_rows = []
    comparison_deep = locked_df.copy()
    for _, deep_row in comparison_deep.iterrows():
        baseline = baseline_locked_df[
            (baseline_locked_df["dataset"] == deep_row["dataset"])
            & (baseline_locked_df["scenario"] == deep_row["scenario"])
        ]
        if baseline.empty:
            continue
        baseline_row = baseline.iloc[0]
        vs_rows.append(
            {
                "dataset": deep_row["dataset"],
                "scenario": deep_row["scenario"],
                "deep_variant": deep_row["variant"],
                "deep_candidate_id": deep_row["candidate_id"],
                "deep_result_type": "selected_seed",
                "deep_prediction_mode": deep_row["prediction_mode"],
                "deep_macro_f1": deep_row["macro_f1"],
                "deep_recall_low": deep_row["recall_low"],
                "deep_f1_low": deep_row["f1_low"],
                "baseline_model": baseline_row["model"],
                "baseline_macro_f1": baseline_row["macro_f1"],
                "baseline_recall_low": baseline_row["recall_low"],
                "baseline_f1_low": baseline_row["f1_low"],
                "macro_f1_gap_deep_minus_baseline": deep_row["macro_f1"] - baseline_row["macro_f1"],
                "recall_low_gap_deep_minus_baseline": deep_row["recall_low"] - baseline_row["recall_low"],
                "f1_low_gap_deep_minus_baseline": deep_row["f1_low"] - baseline_row["f1_low"],
            }
        )
    vs_df = pd.DataFrame(vs_rows)

    outputs = {
        "cv": cv_df,
        "thresholds": thresholds_df,
        "locked": locked_df,
        "ensemble": ensemble_df,
        "selection": selection_df,
        "baseline_cv": baseline_cv_df,
        "baseline_locked": baseline_locked_df,
        "vs_baseline": vs_df,
    }
    cv_df.to_csv(V28_DIR / "deep_v28_cv_results.csv", index=False)
    locked_df.to_csv(V28_DIR / "deep_v28_locked_test_results.csv", index=False)
    thresholds_df.to_csv(V28_DIR / "deep_v28_thresholds.csv", index=False)
    ensemble_df.to_csv(V28_DIR / "deep_v28_ensemble_results.csv", index=False)
    vs_df.to_csv(V28_DIR / "deep_v28_vs_baseline.csv", index=False)
    baseline_cv_df.to_csv(V28_DIR / "baseline_v28_cv_results.csv", index=False)
    baseline_locked_df.to_csv(V28_DIR / "baseline_v28_locked_test_results.csv", index=False)
    selection_df.to_csv(V28_DIR / "deep_v28_selection.csv", index=False)
    save_json(
        V28_DIR / "deep_v28_config.json",
        {
            "config": asdict(config),
            "tasks": tasks,
            "fallback_baseline": sorted(fallback_notes),
            "selection_rule": "Macro F1, then Recall Low, then F1 Low on CV/OOF only. Locked test is final evaluation only.",
            "regression_head": "not claimed; V28 reports classification metrics and mapped-class RMSE/R2 only.",
        },
    )
    write_reports(outputs, tasks, config, sorted(fallback_notes))
    return outputs


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    out = frame[columns].head(limit).copy()
    for col in out.select_dtypes(include=["float"]).columns:
        out[col] = out[col].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(out.columns) + " |"
    separator = "| " + " | ".join("---" for _ in out.columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in out.to_numpy()]
    return "\n".join([header, separator, *rows])


def write_reports(outputs: dict[str, pd.DataFrame], tasks: list[tuple[str, str]], config: V28RunConfig, fallback_notes: list[str]) -> None:
    old = old_champion_rows()
    champion_new = rank_rows(outputs["selection"]) if not outputs["selection"].empty else pd.DataFrame()
    selection = outputs["selection"]
    vs = outputs["vs_baseline"]

    selection_lines = [
        "# Final Deep Model Selection V28",
        "",
        "- Selection uses CV/OOF only: Macro F1, then Recall Low, then F1 Low.",
        "- Locked test is used only after selection for final evaluation.",
        "- Gated fusion is not selected unless it wins CV/OOF over sequence-only.",
        "- Regression head is not claimed in V28.",
        "",
        "## CV/OOF Selected Models",
        "",
        markdown_table(selection, ["dataset", "scenario", "candidate_id", "variant", "prediction_mode", "macro_f1", "recall_low", "f1_low"], limit=20),
    ]
    (V28_DIR / "final_deep_model_selection.md").write_text("\n".join(selection_lines), encoding="utf-8")

    summary_lines = [
        "# Final V28 Summary",
        "",
        "## Protocol",
        "",
        "- No ADASYN.",
        "- No student-combine.",
        "- Thresholds are tuned from OOF train-pool probabilities, never locked test.",
        "- Model selection uses CV/OOF only. Locked test is final evaluation only.",
        "- Regression head is not claimed because prior regression-head RMSE remained high; V28 reports classification-first results.",
        f"- Runtime config: cv_folds={config.cv_folds}, max_epochs={config.max_epochs}, ensemble_seeds={list(config.ensemble_seeds)}.",
        f"- Baseline package path used: {', '.join(fallback_notes) if fallback_notes else 'not recorded'}.",
        "",
        "## Old Deep Champions From deep_debug_summary",
        "",
        markdown_table(old, ["dataset", "scenario", "variant", "prediction_mode", "macro_f1", "recall_low", "f1_low"], limit=12),
        "",
        "## New V28 Locked-Test Results",
        "",
        markdown_table(rank_rows(outputs["locked"]), ["dataset", "scenario", "variant", "candidate_id", "prediction_mode", "macro_f1", "recall_low", "f1_low"], limit=20),
        "",
        "## Seed Ensemble Locked-Test Check",
        "",
        markdown_table(rank_rows(outputs["ensemble"]), ["dataset", "scenario", "variant", "candidate_id", "prediction_mode", "macro_f1", "recall_low", "f1_low"], limit=20),
        "",
        "## Deep Vs Baseline Same Scenario",
        "",
        markdown_table(rank_rows(vs.rename(columns={"deep_macro_f1": "macro_f1", "deep_recall_low": "recall_low", "deep_f1_low": "f1_low"})), ["dataset", "scenario", "deep_variant", "deep_result_type", "deep_prediction_mode", "macro_f1", "recall_low", "f1_low", "baseline_model", "baseline_macro_f1", "macro_f1_gap_deep_minus_baseline"], limit=20),
        "",
        "## Conclusion",
        "",
    ]
    if not champion_new.empty:
        top = champion_new.iloc[0]
        summary_lines.append(
            f"Selected deep architecture for thesis reporting by CV/OOF: `{top['variant']}` from `{top['dataset']}/{top['scenario']}` with CV/OOF Macro F1={top['macro_f1']:.4f}, Recall Low={top['recall_low']:.4f}, F1 Low={top['f1_low']:.4f}. For each dataset/scenario, use the corresponding CV-selected row in `deep_v28_selection.csv`; do not substitute models based on locked-test ranking."
        )
    else:
        summary_lines.append("No V28 sequence model was selected; inspect CV outputs.")
    (V28_DIR / "final_v28_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
