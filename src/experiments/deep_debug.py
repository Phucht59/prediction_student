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
    POINT_CENTERS_3CLASS,
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

BASE_DEBUG_VARIANTS = ("sequence_cnn_bilstm_only", "context_mlp_only", "fusion_cnn_bilstm_context")
CONTEXT_DEBUG_VARIANTS = ("context_mlp_only", "context_mlp_v2")


@dataclass(frozen=True)
class DebugRunConfig:
    max_epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    context_v2_hidden_dim: int = 96
    context_v2_grid: tuple[tuple[float, float, bool], ...] = (
        (0.10, 1e-4, True),
        (0.25, 1e-4, True),
        (0.25, 1e-3, False),
    )
    overfit_sample_size: int = 64
    overfit_target_f1: float = 0.95
    overfit_use_feature_selection: bool = False


@dataclass(frozen=True)
class VariantSpec:
    variant: str
    config_id: str
    dropout: float
    weight_decay: float
    use_feature_selection: bool


def variant_specs_for_scenario(scenario: str, config: ExperimentConfig, run_config: DebugRunConfig) -> list[VariantSpec]:
    specs = [
        VariantSpec(
            variant="context_mlp_only",
            config_id="default",
            dropout=0.15,
            weight_decay=run_config.weight_decay,
            use_feature_selection=config.use_feature_selection,
        )
    ]
    for dropout, weight_decay, use_feature_selection in run_config.context_v2_grid:
        specs.append(
            VariantSpec(
                variant="context_mlp_v2",
                config_id=f"dropout{dropout:g}_wd{weight_decay:g}_fs{int(use_feature_selection)}",
                dropout=float(dropout),
                weight_decay=float(weight_decay),
                use_feature_selection=bool(use_feature_selection),
            )
        )
    if scenario != "early":
        specs.insert(
            0,
            VariantSpec(
                variant="sequence_cnn_bilstm_only",
                config_id="default",
                dropout=0.15,
                weight_decay=run_config.weight_decay,
                use_feature_selection=config.use_feature_selection,
            ),
        )
        specs.append(
            VariantSpec(
                variant="fusion_cnn_bilstm_context",
                config_id="default",
                dropout=0.15,
                weight_decay=run_config.weight_decay,
                use_feature_selection=config.use_feature_selection,
            )
        )
    return specs


def _cat_cardinalities(dataset: TechnicalStudentDataset, prepared) -> list[int]:
    return [len(prepared.preprocessor.label_encoders[column].classes_) for column in dataset.cat_cols]


class ContextMLPOnly(nn.Module):
    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: list[int],
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.15,
    ):
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
            nn.Dropout(dropout),
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


class ContextMLPV2(ContextMLPOnly):
    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: list[int],
        hidden_dim: int = 96,
        num_classes: int = 3,
        dropout: float = 0.25,
    ):
        nn.Module.__init__(self)
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
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.ordinal_head = nn.Linear(hidden_dim, num_classes - 1)
        self.reg_head = nn.Linear(hidden_dim, 1)


class SequenceCNNBiLSTMOnly(nn.Module):
    def __init__(self, hidden_dim: int = 64, cnn_channels: int = 32, num_classes: int = 3, dropout: float = 0.15):
        super().__init__()
        self.sequence_cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
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


def create_debug_model(
    variant_spec: VariantSpec,
    dataset: TechnicalStudentDataset,
    prepared,
    run_config: DebugRunConfig,
) -> nn.Module:
    cat_cardinalities = _cat_cardinalities(dataset, prepared)
    if variant_spec.variant == "context_mlp_only":
        return ContextMLPOnly(
            len(dataset.num_cols),
            cat_cardinalities,
            hidden_dim=run_config.hidden_dim,
            dropout=variant_spec.dropout,
        )
    if variant_spec.variant == "context_mlp_v2":
        return ContextMLPV2(
            len(dataset.num_cols),
            cat_cardinalities,
            hidden_dim=run_config.context_v2_hidden_dim,
            dropout=variant_spec.dropout,
        )
    if variant_spec.variant == "sequence_cnn_bilstm_only":
        return SequenceCNNBiLSTMOnly(hidden_dim=run_config.hidden_dim, dropout=variant_spec.dropout)
    if variant_spec.variant == "fusion_cnn_bilstm_context":
        return StudentHybridV27(
            num_classes=3,
            seq_in_channels=1,
            num_numerical=len(dataset.num_cols),
            cat_cardinalities=cat_cardinalities,
            cnn_channels=32,
            lstm_hidden_dim=run_config.hidden_dim,
            context_hidden_dim=run_config.hidden_dim,
            fusion_hidden_dim=run_config.hidden_dim,
            dropout=variant_spec.dropout,
        )
    raise ValueError(f"Unsupported debug variant: {variant_spec.variant}")


def _train_classification(model, train_loader, device, run_config: DebugRunConfig, weight_decay: float):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=run_config.learning_rate, weight_decay=weight_decay)
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


def _classification_regression_truth(y_true: np.ndarray, reg_true: np.ndarray | None) -> np.ndarray:
    if reg_true is not None:
        return np.asarray(reg_true, dtype=float)
    labels = np.asarray(y_true, dtype=int)
    return POINT_CENTERS_3CLASS[np.clip(labels, 0, len(POINT_CENTERS_3CLASS) - 1)]


def predict_low_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    non_low = np.argmax(probs[:, 1:], axis=1) + 1
    return np.where(probs[:, 0] >= threshold, 0, non_low)


def tune_low_class_thresholds(y_true: np.ndarray, probabilities: np.ndarray, reg_true: np.ndarray | None = None) -> list[dict]:
    rows: list[dict] = []
    argmax_preds = np.argmax(probabilities, axis=1)
    argmax_metrics = compute_required_metrics(y_true, argmax_preds, probabilities, y_reg_true=_classification_regression_truth(y_true, reg_true))
    rows.append({"prediction_mode": "argmax", "threshold_low": np.nan, "selection_score": argmax_metrics["macro_f1"], **argmax_metrics})
    objectives = {
        "low_threshold_tuned": lambda m: 0.65 * m["recall_low"] + 0.35 * m["macro_f1"],
        "low_f1_tuned": lambda m: m["f1_low"] + 1e-3 * m["macro_f1"],
        "low_recall_priority": lambda m: 0.80 * m["recall_low"] + 0.20 * m["f1_low"],
    }
    for mode, objective in objectives.items():
        best_row: dict | None = None
        for threshold in np.linspace(0.05, 0.95, 37):
            preds = predict_low_threshold(probabilities, float(threshold))
            metrics = compute_required_metrics(
                y_true,
                preds,
                probabilities,
                y_reg_true=_classification_regression_truth(y_true, reg_true),
            )
            score = float(objective(metrics))
            candidate = {
                "prediction_mode": mode,
                "threshold_low": float(threshold),
                "selection_score": score,
                **metrics,
            }
            if best_row is None or score > best_row["selection_score"]:
                best_row = candidate
        if best_row is not None:
            rows.append(best_row)
    return rows


def _select_threshold_row(threshold_rows: list[dict], mode: str) -> dict:
    for row in threshold_rows:
        if row["prediction_mode"] == mode:
            return row
    raise ValueError(f"Missing threshold mode: {mode}")


def _predict_for_mode(probabilities: np.ndarray, threshold_row: dict) -> np.ndarray:
    if threshold_row["prediction_mode"] == "argmax":
        return np.argmax(probabilities, axis=1)
    return predict_low_threshold(probabilities, float(threshold_row["threshold_low"]))


def _rank_for_selection(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["macro_f1", "recall_low", "f1_low"], ascending=[False, False, False])


def _best_baseline_by_cv(dirs: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_cv_path = dirs["scenarios"] / "baseline_cv_all.csv"
    baseline_locked_path = dirs["scenarios"] / "baseline_locked_test_all.csv"
    if not baseline_cv_path.exists() or not baseline_locked_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    baseline_cv = pd.read_csv(baseline_cv_path)
    baseline_locked = pd.read_csv(baseline_locked_path)
    cv_summary = (
        baseline_cv.groupby(["dataset", "scenario", "strategy", "model"], as_index=False)
        .agg(
            macro_f1=("macro_f1", "mean"),
            recall_low=("recall_low", "mean"),
            f1_low=("f1_low", "mean"),
        )
    )
    selected_rows = []
    for _, group in cv_summary.groupby(["dataset", "scenario"]):
        selected_rows.append(_rank_for_selection(group).iloc[0].to_dict())
    selected = pd.DataFrame(selected_rows)
    return selected, baseline_locked


def build_baseline_vs_deep_same_scenario(
    deep_locked: pd.DataFrame,
    threshold_tuning: pd.DataFrame,
    dirs: dict[str, object],
) -> pd.DataFrame:
    selected_baseline, baseline_locked = _best_baseline_by_cv(dirs)
    if selected_baseline.empty or baseline_locked.empty or deep_locked.empty:
        return pd.DataFrame()

    selected_deep_rows = []
    for _, group in threshold_tuning.groupby(["dataset", "scenario"]):
        selected_deep_rows.append(_rank_for_selection(group).iloc[0].to_dict())
    selected_deep = pd.DataFrame(selected_deep_rows)

    rows: list[dict] = []
    for _, baseline_cv_row in selected_baseline.iterrows():
        dataset = baseline_cv_row["dataset"]
        scenario = baseline_cv_row["scenario"]
        baseline_match = baseline_locked[
            (baseline_locked["dataset"] == dataset)
            & (baseline_locked["scenario"] == scenario)
            & (baseline_locked["strategy"] == baseline_cv_row["strategy"])
            & (baseline_locked["model"] == baseline_cv_row["model"])
        ]
        if baseline_match.empty:
            continue
        baseline_row = baseline_match.iloc[0]
        deep_group = deep_locked[(deep_locked["dataset"] == dataset) & (deep_locked["scenario"] == scenario)]
        selected_deep_group = selected_deep[(selected_deep["dataset"] == dataset) & (selected_deep["scenario"] == scenario)]
        for _, deep_row in deep_group.iterrows():
            is_cv_selected = False
            if not selected_deep_group.empty:
                selected_row = selected_deep_group.iloc[0]
                is_cv_selected = (
                    deep_row["variant"] == selected_row["variant"]
                    and deep_row["config_id"] == selected_row["config_id"]
                    and deep_row["prediction_mode"] == selected_row["prediction_mode"]
                )
            rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "baseline_selected_by": "cv_macro_f1_recall_low_f1_low",
                    "baseline_model": baseline_cv_row["model"],
                    "baseline_strategy": baseline_cv_row["strategy"],
                    "baseline_locked_macro_f1": float(baseline_row["macro_f1"]),
                    "baseline_locked_recall_low": float(baseline_row["recall_low"]),
                    "baseline_locked_f1_low": float(baseline_row["f1_low"]),
                    "deep_variant": deep_row["variant"],
                    "deep_config_id": deep_row["config_id"],
                    "deep_prediction_mode": deep_row["prediction_mode"],
                    "deep_cv_selected": bool(is_cv_selected),
                    "deep_locked_macro_f1": float(deep_row["macro_f1"]),
                    "deep_locked_recall_low": float(deep_row["recall_low"]),
                    "deep_locked_f1_low": float(deep_row["f1_low"]),
                    "macro_f1_gap_deep_minus_baseline": float(deep_row["macro_f1"] - baseline_row["macro_f1"]),
                    "recall_low_gap_deep_minus_baseline": float(deep_row["recall_low"] - baseline_row["recall_low"]),
                    "f1_low_gap_deep_minus_baseline": float(deep_row["f1_low"] - baseline_row["f1_low"]),
                }
            )
    return pd.DataFrame(rows)


def run_overfit_sanity(
    dataset_name: str,
    scenario: str,
    variant_specs: list[VariantSpec],
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
    for variant_spec in variant_specs:
        set_seed(config.seed)
        model = create_debug_model(variant_spec, train_ds, prepared, run_config).to(device)
        _train_classification(model, loader, device, run_config, variant_spec.weight_decay)
        probs, y_true, reg_true, reg_pred = _predict(model, eval_loader, device)
        preds = np.argmax(probs, axis=1)
        metrics = compute_required_metrics(y_true, preds, probs, y_reg_true=reg_true, y_reg_pred=reg_pred)
        rows.append(
            {
                "dataset": dataset_name,
                "scenario": scenario,
                "variant": variant_spec.variant,
                "config_id": variant_spec.config_id,
                "dropout": variant_spec.dropout,
                "weight_decay": variant_spec.weight_decay,
                "use_feature_selection": variant_spec.use_feature_selection,
                "status": "pass" if metrics["macro_f1"] >= run_config.overfit_target_f1 else "fail",
                "target_macro_f1": run_config.overfit_target_f1,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def run_branch_ablation(
    dataset_name: str,
    scenario: str,
    variant_specs: list[VariantSpec],
    config: ExperimentConfig,
    run_config: DebugRunConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_config = run_config or DebugRunConfig(max_epochs=20)
    spec = DATASETS[dataset_name]
    train_pool, locked_test = load_or_create_student_splits(dataset_name, config.target_mode)
    labels = train_pool[spec.target_col].astype(int).to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cv_rows: list[dict] = []
    locked_rows: list[dict] = []
    threshold_rows: list[dict] = []
    for variant_spec in variant_specs:
        oof_probs = np.zeros((len(train_pool), 3), dtype=float)
        oof_targets = np.zeros(len(train_pool), dtype=int)
        oof_reg_true = np.zeros(len(train_pool), dtype=float)
        for fold, (train_idx, val_idx) in enumerate(stratified_folds(labels, config).split(train_pool, labels), start=1):
            fold_config = replace(config, use_feature_selection=variant_spec.use_feature_selection)
            prepared = prepare_fold(
                train_pool.iloc[train_idx].copy(),
                train_pool.iloc[val_idx].copy(),
                spec.target_col,
                scenario,
                "none",
                fold_config,
            )
            train_ds = _make_dataset(prepared.train, prepared, spec.target_col)
            val_ds = _make_dataset(prepared.validation, prepared, spec.target_col)
            drop_last = len(train_ds) > run_config.batch_size and len(train_ds) % run_config.batch_size == 1
            train_loader = DataLoader(train_ds, batch_size=run_config.batch_size, shuffle=True, drop_last=drop_last)
            val_loader = DataLoader(val_ds, batch_size=run_config.batch_size, shuffle=False)
            set_seed(config.seed + fold)
            model = create_debug_model(variant_spec, train_ds, prepared, run_config).to(device)
            _train_classification(model, train_loader, device, run_config, variant_spec.weight_decay)
            probs, y_true, reg_true, reg_pred = _predict(model, val_loader, device)
            preds = np.argmax(probs, axis=1)
            oof_probs[val_idx] = probs
            oof_targets[val_idx] = y_true
            oof_reg_true[val_idx] = reg_true
            cv_rows.append(
                {
                    "dataset": dataset_name,
                    "scenario": scenario,
                    "variant": variant_spec.variant,
                    "config_id": variant_spec.config_id,
                    "dropout": variant_spec.dropout,
                    "weight_decay": variant_spec.weight_decay,
                    "use_feature_selection": variant_spec.use_feature_selection,
                    "fold": fold,
                    "status": "evaluated",
                    **compute_required_metrics(y_true, preds, probs, y_reg_true=reg_true, y_reg_pred=reg_pred),
                }
            )

        tuned_rows = tune_low_class_thresholds(oof_targets, oof_probs, oof_reg_true)
        for row in tuned_rows:
            threshold_rows.append(
                {
                    "dataset": dataset_name,
                    "scenario": scenario,
                    "variant": variant_spec.variant,
                    "config_id": variant_spec.config_id,
                    "dropout": variant_spec.dropout,
                    "weight_decay": variant_spec.weight_decay,
                    "use_feature_selection": variant_spec.use_feature_selection,
                    **row,
                }
            )

        full_config = replace(config, use_feature_selection=variant_spec.use_feature_selection)
        prepared = prepare_fold(train_pool.copy(), locked_test.copy(), spec.target_col, scenario, "none", full_config)
        locked_selected = transform_with_prepared(locked_test.copy(), prepared, spec.target_col, scenario)
        train_ds = _make_dataset(prepared.train, prepared, spec.target_col)
        locked_ds = _make_dataset(locked_selected, prepared, spec.target_col)
        drop_last = len(train_ds) > run_config.batch_size and len(train_ds) % run_config.batch_size == 1
        train_loader = DataLoader(train_ds, batch_size=run_config.batch_size, shuffle=True, drop_last=drop_last)
        locked_loader = DataLoader(locked_ds, batch_size=run_config.batch_size, shuffle=False)
        set_seed(config.seed)
        model = create_debug_model(variant_spec, train_ds, prepared, run_config).to(device)
        _train_classification(model, train_loader, device, run_config, variant_spec.weight_decay)
        probs, y_true, reg_true, reg_pred = _predict(model, locked_loader, device)
        for threshold_row in tuned_rows:
            preds = _predict_for_mode(probs, threshold_row)
            locked_rows.append(
                {
                    "dataset": dataset_name,
                    "scenario": scenario,
                    "variant": variant_spec.variant,
                    "config_id": variant_spec.config_id,
                    "dropout": variant_spec.dropout,
                    "weight_decay": variant_spec.weight_decay,
                    "use_feature_selection": variant_spec.use_feature_selection,
                    "prediction_mode": threshold_row["prediction_mode"],
                    "threshold_low": threshold_row["threshold_low"],
                    "fold": "locked_test",
                    "status": "evaluated",
                    **compute_required_metrics(y_true, preds, probs, y_reg_true=reg_true, y_reg_pred=reg_pred),
                }
            )
    return pd.DataFrame(cv_rows), pd.DataFrame(locked_rows), pd.DataFrame(threshold_rows)


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
    threshold_frames = []
    for dataset_name in datasets:
        for scenario in scenarios:
            logger.info("Deep debug suite: dataset=%s scenario=%s", dataset_name, scenario)
            variant_specs = variant_specs_for_scenario(scenario, config, run_config)
            overfit_frames.append(run_overfit_sanity(dataset_name, scenario, variant_specs, config, run_config))
            cv_df, locked_df, threshold_df = run_branch_ablation(dataset_name, scenario, variant_specs, config, run_config)
            cv_frames.append(cv_df)
            locked_frames.append(locked_df)
            threshold_frames.append(threshold_df)
    outputs = {
        "overfit": pd.concat(overfit_frames, ignore_index=True),
        "deep_ablation_cv": pd.concat(cv_frames, ignore_index=True),
        "deep_ablation_locked_test": pd.concat(locked_frames, ignore_index=True),
        "low_class_threshold_tuning": pd.concat(threshold_frames, ignore_index=True),
    }
    dirs = ensure_technical_report_dirs()
    outputs["overfit"].to_csv(dirs["ablation"] / "deep_overfit_sanity.csv", index=False)
    outputs["deep_ablation_cv"].to_csv(dirs["ablation"] / "deep_ablation_cv.csv", index=False)
    outputs["deep_ablation_locked_test"].to_csv(dirs["ablation"] / "deep_ablation_locked_test.csv", index=False)
    outputs["low_class_threshold_tuning"].to_csv(dirs["ablation"] / "low_class_threshold_tuning.csv", index=False)
    # Backward-compatible debug file names.
    outputs["deep_ablation_cv"].to_csv(dirs["ablation"] / "deep_branch_ablation_cv.csv", index=False)
    outputs["deep_ablation_locked_test"].to_csv(dirs["ablation"] / "deep_branch_ablation_locked_test.csv", index=False)

    outputs["baseline_vs_deep_same_scenario"] = build_baseline_vs_deep_same_scenario(
        outputs["deep_ablation_locked_test"],
        outputs["low_class_threshold_tuning"],
        dirs,
    )
    outputs["baseline_vs_deep_same_scenario"].to_csv(dirs["ablation"] / "baseline_vs_deep_same_scenario.csv", index=False)

    cv_evaluated = outputs["deep_ablation_cv"][outputs["deep_ablation_cv"].get("status") == "evaluated"]
    summary = {
        "overfit": outputs["overfit"].to_dict("records"),
        "branch_ablation_cv_summary": [
            {**dict(zip(["dataset", "scenario", "variant", "config_id"], keys)), **summarize_cv(group.to_dict("records"))}
            for keys, group in cv_evaluated.groupby(["dataset", "scenario", "variant", "config_id"])
        ] if not cv_evaluated.empty else [],
        "branch_ablation_locked_test": outputs["deep_ablation_locked_test"].to_dict("records"),
        "low_class_threshold_tuning": outputs["low_class_threshold_tuning"].to_dict("records"),
        "baseline_vs_deep_same_scenario": outputs["baseline_vs_deep_same_scenario"].to_dict("records"),
        "notes": [
            "Early scenario has no real G1/G2 sequence; only context MLP variants are evaluated.",
            "Thresholds are tuned on out-of-fold train-pool probabilities and applied once to locked test.",
            "Baseline-vs-deep comparison uses CV-selected baselines and the same locked test per dataset/scenario.",
            "Main RMSE/R2 use class-to-point mapping. Regression-head metrics are reported separately.",
            "Do not claim the regression head while regression_head_rmse remains high.",
            "Overfit sanity should approach Macro F1 >= target before treating the deep model as thesis-ready.",
        ],
    }
    save_json(dirs["ablation"] / "deep_debug_summary.json", summary)
    write_config(dirs["ablation"] / "deep_debug_config.json", config, {"debug": run_config.__dict__})
    return outputs
