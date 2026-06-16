from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import DATASETS
from src.experiments.common import (
    ExperimentConfig,
    TechnicalStudentDataset,
    compute_required_metrics,
    ensure_technical_report_dirs,
    load_or_create_student_splits,
    prepare_fold,
    save_json,
    stratified_folds,
    summarize_cv,
    transform_with_prepared,
    write_config,
)
from src.models_v27 import AttentionPooling1D, StudentHybridV27
from src.utils import set_seed, setup_logger

logger = setup_logger("deep_debug")

DEBUG_VARIANTS = ("context_mlp_only", "sequence_cnn_bilstm_only", "fusion_cnn_bilstm_context")


@dataclass(frozen=True)
class DebugRunConfig:
    max_epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    overfit_sample_size: int = 64
    overfit_target_f1: float = 0.95
    overfit_use_feature_selection: bool = False


def _cat_cardinalities(dataset: TechnicalStudentDataset, prepared) -> list[int]:
    return [len(prepared.preprocessor.label_encoders[column].classes_) for column in dataset.cat_cols]


class ContextMLPOnly(nn.Module):
    def __init__(self, num_numerical: int, cat_cardinalities: list[int], hidden_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities
        self.embeddings = nn.ModuleList()
        embedding_total_dim = 0
        for cardinality in cat_cardinalities:
            dim = max(2, min(50, (cardinality + 1) // 2))
            self.embeddings.append(nn.Embedding(cardinality, dim))
            embedding_total_dim += dim
        input_dim = max(1, num_numerical + embedding_total_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.ordinal_head = nn.Linear(hidden_dim, num_classes - 1)
        self.reg_head = nn.Linear(hidden_dim, 1)

    def _context(self, num_x: torch.Tensor, cat_x: torch.Tensor) -> torch.Tensor:
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
        hidden = self.mlp(self._context(num_x, cat_x))
        return self.class_head(hidden), self.ordinal_head(hidden), self.reg_head(hidden).squeeze(-1)


class SequenceCNNBiLSTMOnly(nn.Module):
    def __init__(self, hidden_dim: int = 64, cnn_channels: int = 32, num_classes: int = 3):
        super().__init__()
        self.sequence_cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.sequence_bilstm = nn.LSTM(cnn_channels, hidden_dim, batch_first=True, bidirectional=True)
        self.pool = AttentionPooling1D(hidden_dim * 2)
        self.class_head = nn.Linear(hidden_dim * 2, num_classes)
        self.ordinal_head = nn.Linear(hidden_dim * 2, num_classes - 1)
        self.reg_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, seq_x, num_x, cat_x):
        seq = self.sequence_cnn(seq_x.float().transpose(1, 2)).transpose(1, 2)
        seq, _ = self.sequence_bilstm(seq)
        hidden, _ = self.pool(seq)
        return self.class_head(hidden), self.ordinal_head(hidden), self.reg_head(hidden).squeeze(-1)


def _make_dataset(frame: pd.DataFrame, prepared, target_col: str) -> TechnicalStudentDataset:
    return TechnicalStudentDataset(
        frame,
        target_col=target_col,
        numerical_cols=prepared.preprocessor.numerical_cols,
        categorical_cols=prepared.preprocessor.categorical_cols,
        sequence_cols=prepared.sequence_cols,
    )


def create_debug_model(variant: str, dataset: TechnicalStudentDataset, prepared, run_config: DebugRunConfig) -> nn.Module:
    cat_cardinalities = _cat_cardinalities(dataset, prepared)
    if variant == "context_mlp_only":
        return ContextMLPOnly(len(dataset.num_cols), cat_cardinalities, hidden_dim=run_config.hidden_dim)
    if variant == "sequence_cnn_bilstm_only":
        return SequenceCNNBiLSTMOnly(hidden_dim=run_config.hidden_dim)
    if variant == "fusion_cnn_bilstm_context":
        return StudentHybridV27(
            num_classes=3,
            seq_in_channels=1,
            num_numerical=len(dataset.num_cols),
            cat_cardinalities=cat_cardinalities,
            cnn_channels=32,
            lstm_hidden_dim=run_config.hidden_dim,
            context_hidden_dim=run_config.hidden_dim,
            fusion_hidden_dim=run_config.hidden_dim,
            dropout=0.15,
        )
    raise ValueError(f"Unsupported debug variant: {variant}")


def _train_classification(model, train_loader, device, run_config: DebugRunConfig):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=run_config.learning_rate, weight_decay=run_config.weight_decay)
    for _ in range(run_config.max_epochs):
        model.train()
        for seq_x, num_x, cat_x, labels, _, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(seq_x.to(device), num_x.to(device), cat_x.to(device))
            loss = criterion(outputs[0], labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


def _predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    reg_true: list[np.ndarray] = []
    reg_pred: list[np.ndarray] = []
    with torch.no_grad():
        for seq_x, num_x, cat_x, y, _, y_reg in loader:
            outputs = model(seq_x.to(device), num_x.to(device), cat_x.to(device))
            probs.append(torch.softmax(outputs[0], dim=1).cpu().numpy())
            labels.append(y.numpy())
            reg_true.append(y_reg.numpy())
            reg_pred.append(outputs[2].cpu().numpy())
    return np.vstack(probs), np.concatenate(labels), np.concatenate(reg_true), np.concatenate(reg_pred)


def run_overfit_sanity(
    dataset_name: str,
    scenario: str,
    variants: list[str],
    config: ExperimentConfig,
    run_config: DebugRunConfig | None = None,
) -> pd.DataFrame:
    run_config = run_config or DebugRunConfig()
    spec = DATASETS[dataset_name]
    train_pool, _ = load_or_create_student_splits(dataset_name, config.target_mode)
    sample_parts = []
    for _, group in train_pool.groupby(spec.target_col):
        sample_parts.append(
            group.sample(
                min(len(group), max(1, run_config.overfit_sample_size // 3)),
                random_state=config.seed,
            )
        )
    sample = pd.concat(sample_parts, ignore_index=True)
    if len(sample) > run_config.overfit_sample_size:
        sample = sample.sample(run_config.overfit_sample_size, random_state=config.seed)
    overfit_config = replace(config, use_feature_selection=run_config.overfit_use_feature_selection)
    prepared = prepare_fold(sample.copy(), sample.copy(), spec.target_col, scenario, "none", overfit_config)
    train_ds = _make_dataset(prepared.train, prepared, spec.target_col)
    loader = DataLoader(train_ds, batch_size=run_config.batch_size, shuffle=True)
    eval_loader = DataLoader(train_ds, batch_size=run_config.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    for variant in variants:
        if variant == "sequence_cnn_bilstm_only" and not prepared.sequence_cols:
            rows.append(
                {
                    "dataset": dataset_name,
                    "scenario": scenario,
                    "variant": variant,
                    "status": "skipped_no_true_sequence",
                }
            )
            continue
        set_seed(config.seed)
        model = create_debug_model(variant, train_ds, prepared, run_config).to(device)
        _train_classification(model, loader, device, run_config)
        probs, y_true, reg_true, reg_pred = _predict(model, eval_loader, device)
        preds = np.argmax(probs, axis=1)
        metrics = compute_required_metrics(y_true, preds, probs, y_reg_true=reg_true, y_reg_pred=reg_pred)
        rows.append(
            {
                "dataset": dataset_name,
                "scenario": scenario,
                "variant": variant,
                "status": "pass" if metrics["macro_f1"] >= run_config.overfit_target_f1 else "fail",
                "target_macro_f1": run_config.overfit_target_f1,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def run_branch_ablation(
    dataset_name: str,
    scenario: str,
    variants: list[str],
    config: ExperimentConfig,
    run_config: DebugRunConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_config = run_config or DebugRunConfig(max_epochs=20)
    spec = DATASETS[dataset_name]
    train_pool, locked_test = load_or_create_student_splits(dataset_name, config.target_mode)
    labels = train_pool[spec.target_col].astype(int).to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cv_rows: list[dict] = []
    locked_rows: list[dict] = []
    for variant in variants:
        if variant == "sequence_cnn_bilstm_only" and scenario == "early":
            cv_rows.append({"dataset": dataset_name, "scenario": scenario, "variant": variant, "fold": "all", "status": "skipped_no_true_sequence"})
            continue
        for fold, (train_idx, val_idx) in enumerate(stratified_folds(labels, config).split(train_pool, labels), start=1):
            prepared = prepare_fold(
                train_pool.iloc[train_idx].copy(),
                train_pool.iloc[val_idx].copy(),
                spec.target_col,
                scenario,
                "none",
                config,
            )
            train_ds = _make_dataset(prepared.train, prepared, spec.target_col)
            val_ds = _make_dataset(prepared.validation, prepared, spec.target_col)
            train_loader = DataLoader(train_ds, batch_size=run_config.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=run_config.batch_size, shuffle=False)
            set_seed(config.seed + fold)
            model = create_debug_model(variant, train_ds, prepared, run_config).to(device)
            _train_classification(model, train_loader, device, run_config)
            probs, y_true, reg_true, reg_pred = _predict(model, val_loader, device)
            preds = np.argmax(probs, axis=1)
            cv_rows.append(
                {
                    "dataset": dataset_name,
                    "scenario": scenario,
                    "variant": variant,
                    "fold": fold,
                    "status": "evaluated",
                    **compute_required_metrics(y_true, preds, probs, y_reg_true=reg_true, y_reg_pred=reg_pred),
                }
            )

        prepared = prepare_fold(train_pool.copy(), locked_test.copy(), spec.target_col, scenario, "none", config)
        locked_selected = transform_with_prepared(locked_test.copy(), prepared, spec.target_col, scenario)
        train_ds = _make_dataset(prepared.train, prepared, spec.target_col)
        locked_ds = _make_dataset(locked_selected, prepared, spec.target_col)
        train_loader = DataLoader(train_ds, batch_size=run_config.batch_size, shuffle=True)
        locked_loader = DataLoader(locked_ds, batch_size=run_config.batch_size, shuffle=False)
        set_seed(config.seed)
        model = create_debug_model(variant, train_ds, prepared, run_config).to(device)
        _train_classification(model, train_loader, device, run_config)
        probs, y_true, reg_true, reg_pred = _predict(model, locked_loader, device)
        preds = np.argmax(probs, axis=1)
        locked_rows.append(
            {
                "dataset": dataset_name,
                "scenario": scenario,
                "variant": variant,
                "fold": "locked_test",
                "status": "evaluated",
                **compute_required_metrics(y_true, preds, probs, y_reg_true=reg_true, y_reg_pred=reg_pred),
            }
        )
    return pd.DataFrame(cv_rows), pd.DataFrame(locked_rows)


def run_deep_debug_suite(
    datasets: list[str],
    scenarios: list[str],
    config: ExperimentConfig,
    run_config: DebugRunConfig | None = None,
) -> dict[str, pd.DataFrame]:
    run_config = run_config or DebugRunConfig()
    overfit_frames = []
    cv_frames = []
    locked_frames = []
    for dataset_name in datasets:
        for scenario in scenarios:
            logger.info("Deep debug suite: dataset=%s scenario=%s", dataset_name, scenario)
            overfit_frames.append(run_overfit_sanity(dataset_name, scenario, list(DEBUG_VARIANTS), config, run_config))
            cv_df, locked_df = run_branch_ablation(dataset_name, scenario, list(DEBUG_VARIANTS), config, run_config)
            cv_frames.append(cv_df)
            locked_frames.append(locked_df)
    outputs = {
        "overfit": pd.concat(overfit_frames, ignore_index=True),
        "branch_ablation_cv": pd.concat(cv_frames, ignore_index=True),
        "branch_ablation_locked": pd.concat(locked_frames, ignore_index=True),
    }
    dirs = ensure_technical_report_dirs()
    outputs["overfit"].to_csv(dirs["ablation"] / "deep_overfit_sanity.csv", index=False)
    outputs["branch_ablation_cv"].to_csv(dirs["ablation"] / "deep_branch_ablation_cv.csv", index=False)
    outputs["branch_ablation_locked"].to_csv(dirs["ablation"] / "deep_branch_ablation_locked_test.csv", index=False)

    cv_evaluated = outputs["branch_ablation_cv"][outputs["branch_ablation_cv"].get("status") == "evaluated"]
    summary = {
        "overfit": outputs["overfit"].to_dict("records"),
        "branch_ablation_cv_summary": [
            {**dict(zip(["dataset", "scenario", "variant"], keys)), **summarize_cv(group.to_dict("records"))}
            for keys, group in cv_evaluated.groupby(["dataset", "scenario", "variant"])
        ] if not cv_evaluated.empty else [],
        "branch_ablation_locked_test": outputs["branch_ablation_locked"].to_dict("records"),
        "notes": [
            "Early scenario has no real G1/G2 sequence; sequence-only is skipped.",
            "Main RMSE/R2 use class-to-point mapping. Regression-head metrics are reported separately.",
            "Overfit sanity should approach Macro F1 >= target before treating the deep model as thesis-ready.",
        ],
    }
    save_json(dirs["ablation"] / "deep_debug_summary.json", summary)
    write_config(dirs["ablation"] / "deep_debug_config.json", config, {"debug": run_config.__dict__})
    return outputs
