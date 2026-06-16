from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.config import DATASETS, DEFAULT_SEED, REPORTS_DIR
from src.experiments.common import compute_required_metrics, save_json
from src.experiments.v28 import (
    SEED_ENSEMBLE_5,
    _load_or_create_splits,
    make_dataset,
    prepare_v28_fold,
    transform_v28,
)
from src.losses_v27 import ClassBalancedFocalLoss, FocalLoss
from src.models_v27 import AttentionPooling1D, StudentHybridV27
from src.train_pipeline import calculate_class_weights
from src.utils import set_seed, setup_logger

logger = setup_logger("v29_experiments")

V29_DIR = REPORTS_DIR / "v29"
V28_DIR = REPORTS_DIR / "v28"
ABLATION_DIR = REPORTS_DIR / "ablation"
REQUIRED_TASKS = (
    ("student-mat", "late"),
    ("student-por", "late"),
    ("student-por", "midterm"),
    ("xapi", "xapi"),
)
THRESHOLD_MODES = ("argmax", "low_f1_tuned", "low_recall_priority", "balanced_low_macro")


@dataclass(frozen=True)
class V29RunConfig:
    seed: int = DEFAULT_SEED
    cv_folds: int = 5
    max_epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    ensemble_seeds: tuple[int, ...] = SEED_ENSEMBLE_5
    use_feature_selection: bool = True
    smote_ratio: float = 1.0
    resampling_k_neighbors: int = 5


@dataclass(frozen=True)
class V29Candidate:
    candidate_id: str
    variant: str
    model_family: str
    cnn_kernel_size: int
    cnn_channels: int
    lstm_hidden_dim: int
    loss_name: str
    use_class_weight: bool
    imbalance_strategy: str = "none_class_weight"
    dropout: float = 0.15
    context_hidden_dim: int = 64
    is_ensemble: bool = False
    base_candidate_id: str = ""


class SequenceCNNBiLSTMLight(nn.Module):
    """Old debug sequence branch with controlled kernel/loss changes only."""

    def __init__(
        self,
        num_classes: int = 3,
        cnn_kernel_size: int = 3,
        cnn_channels: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.sequence_cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.sequence_bilstm = nn.LSTM(cnn_channels, hidden_dim, batch_first=True, bidirectional=True)
        self.pool = AttentionPooling1D(hidden_dim * 2)
        self.class_head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, seq_x, num_x=None, cat_x=None):
        seq = self.sequence_cnn(seq_x.float().transpose(1, 2)).transpose(1, 2)
        seq, _ = self.sequence_bilstm(seq)
        hidden, _ = self.pool(seq)
        return self.class_head(hidden), None


def ensure_v29_dir() -> None:
    V29_DIR.mkdir(parents=True, exist_ok=True)


def candidate_grid(dataset_name: str) -> list[V29Candidate]:
    candidates = [
        V29Candidate("old_seq_default", "old_seq_default", "sequence", 3, 32, 64, "ce", False, "none"),
        V29Candidate(
            "old_seq_default_ensemble",
            "old_seq_default_ensemble",
            "sequence",
            3,
            32,
            64,
            "ce",
            False,
            "none",
            is_ensemble=True,
            base_candidate_id="old_seq_default",
        ),
        V29Candidate("seq_kernel1_small", "seq_kernel1_small", "sequence", 1, 32, 64, "ce", True, "none"),
        V29Candidate("seq_kernel2_small", "seq_kernel2_small", "sequence", 2, 32, 64, "ce", True, "none"),
        V29Candidate("seq_focal_light", "seq_focal_light", "sequence", 3, 32, 64, "focal_loss", False, "none"),
        V29Candidate("seq_cbf_light", "seq_cbf_light", "sequence", 3, 32, 64, "class_balanced_focal_loss", False, "none"),
    ]
    if dataset_name == "xapi":
        candidates.extend(
            [
                V29Candidate("xapi_sequence_light", "xapi_sequence_gated_light", "sequence", 3, 32, 64, "ce", True, "none"),
                V29Candidate("xapi_gated_light", "xapi_sequence_gated_light", "fusion", 3, 32, 64, "ce", True, "none"),
            ]
        )
    return candidates


def make_model(candidate: V29Candidate, dataset, prepared) -> nn.Module:
    if candidate.model_family == "fusion":
        cat_cardinalities = [len(prepared.preprocessor.label_encoders[column].classes_) for column in dataset.cat_cols]
        return StudentHybridV27(
            num_classes=3,
            seq_in_channels=1,
            num_numerical=len(dataset.num_cols),
            cat_cardinalities=cat_cardinalities,
            cnn_channels=candidate.cnn_channels,
            cnn_kernel_size=candidate.cnn_kernel_size,
            lstm_hidden_dim=candidate.lstm_hidden_dim,
            context_hidden_dim=candidate.context_hidden_dim,
            fusion_hidden_dim=candidate.context_hidden_dim,
            dropout=candidate.dropout,
        )
    return SequenceCNNBiLSTMLight(
        num_classes=3,
        cnn_kernel_size=candidate.cnn_kernel_size,
        cnn_channels=candidate.cnn_channels,
        hidden_dim=candidate.lstm_hidden_dim,
        dropout=candidate.dropout,
    )


def loss_for_candidate(candidate: V29Candidate, labels: np.ndarray, device: torch.device) -> nn.Module:
    if candidate.loss_name == "class_balanced_focal_loss":
        counts = np.bincount(labels.astype(int), minlength=3)
        return ClassBalancedFocalLoss(counts, beta=0.99, gamma=1.5).to(device)
    weight = calculate_class_weights(labels, num_classes=3).to(device) if candidate.use_class_weight else None
    if candidate.loss_name == "focal_loss":
        return FocalLoss(weight=weight, gamma=2.0).to(device)
    return nn.CrossEntropyLoss(weight=weight)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    candidate: V29Candidate,
    original_train_labels: np.ndarray,
    config: V29RunConfig,
    device: torch.device,
) -> nn.Module:
    criterion = loss_for_candidate(candidate, original_train_labels, device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = None
    best_f1 = -1.0
    stale = 0
    for _ in range(config.max_epochs):
        model.train()
        for seq_x, num_x, cat_x, labels, _, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(seq_x.to(device), num_x.to(device), cat_x.to(device))
            loss = criterion(outputs[0], labels.to(device))
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
            logits = model(seq_x.to(device), num_x.to(device), cat_x.to(device))[0]
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(y.numpy())
            reg_true.append(y_reg.numpy())
    return np.vstack(probs), np.concatenate(labels), np.concatenate(reg_true)


def predict_with_threshold(probabilities: np.ndarray, threshold: float | None) -> np.ndarray:
    if threshold is None or pd.isna(threshold):
        return np.argmax(probabilities, axis=1)
    non_low = np.argmax(probabilities[:, 1:], axis=1) + 1
    return np.where(probabilities[:, 0] >= float(threshold), 0, non_low)


def tune_thresholds(y_true: np.ndarray, probabilities: np.ndarray, reg_true: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    argmax_preds = np.argmax(probabilities, axis=1)
    rows.append(
        {
            "prediction_mode": "argmax",
            "threshold_low": np.nan,
            "selection_score": 0.0,
            **compute_required_metrics(y_true, argmax_preds, probabilities, y_reg_true=reg_true),
        }
    )
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
            row = {
                "prediction_mode": mode,
                "threshold_low": float(threshold),
                "selection_score": float(objective(metrics)),
                **metrics,
            }
            if best is None or row["selection_score"] > best["selection_score"]:
                best = row
        rows.append(best)
    return rows


def prepare_for_candidate(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    dataset_name: str,
    scenario: str,
    candidate: V29Candidate,
    config: V29RunConfig,
):
    return prepare_v28_fold(train_df, validation_df, dataset_name, scenario, candidate, config)


def train_candidate_fold(
    train_pool: pd.DataFrame,
    dataset_name: str,
    scenario: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    candidate: V29Candidate,
    config: V29RunConfig,
    fold_seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = DATASETS[dataset_name]
    prepared = prepare_for_candidate(
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
    set_seed(fold_seed)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    model = make_model(candidate, train_ds, prepared).to(device)
    original_labels = train_pool.iloc[train_idx][spec.target_col].astype(int).to_numpy()
    model = train_model(model, train_loader, val_loader, candidate, original_labels, config, device)
    return predict_proba(model, val_loader, device)


def fold_metric_summary(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    reg_true: np.ndarray,
    fold_ids: np.ndarray,
    threshold_row: dict,
) -> dict[str, float]:
    fold_rows = []
    for fold in sorted(np.unique(fold_ids)):
        mask = fold_ids == fold
        preds = predict_with_threshold(probabilities[mask], threshold_row["threshold_low"])
        fold_rows.append(compute_required_metrics(y_true[mask], preds, probabilities[mask], y_reg_true=reg_true[mask]))
    summary: dict[str, float] = {}
    for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "recall_low", "f1_low", "rmse", "r2"):
        values = np.asarray([row[metric] for row in fold_rows], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
    return summary


def run_candidate_oof(
    dataset_name: str,
    scenario: str,
    candidate: V29Candidate,
    config: V29RunConfig,
    train_pool: pd.DataFrame,
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    spec = DATASETS[dataset_name]
    labels = train_pool[spec.target_col].astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    folds = list(skf.split(train_pool, labels))
    seeds = config.ensemble_seeds if candidate.is_ensemble else (config.seed,)
    oof_probs = np.zeros((len(train_pool), 3), dtype=float)
    oof_targets = np.zeros(len(train_pool), dtype=int)
    oof_reg_true = np.zeros(len(train_pool), dtype=float)
    fold_ids = np.zeros(len(train_pool), dtype=int)

    for seed_index, seed in enumerate(seeds, start=1):
        logger.info(
            "OOF task=%s/%s candidate=%s seed=%s/%s",
            dataset_name,
            scenario,
            candidate.candidate_id,
            seed_index,
            len(seeds),
        )
        for fold, (train_idx, val_idx) in enumerate(folds, start=1):
            probs, y_val, reg_true = train_candidate_fold(
                train_pool,
                dataset_name,
                scenario,
                train_idx,
                val_idx,
                candidate,
                config,
                int(seed) + fold * 100,
                device,
            )
            oof_probs[val_idx] += probs / len(seeds)
            oof_targets[val_idx] = y_val
            oof_reg_true[val_idx] = reg_true
            fold_ids[val_idx] = fold

    cv_rows: list[dict] = []
    threshold_rows: list[dict] = []
    for threshold_row in tune_thresholds(oof_targets, oof_probs, oof_reg_true):
        fold_summary = fold_metric_summary(oof_targets, oof_probs, oof_reg_true, fold_ids, threshold_row)
        base = {
            "dataset": dataset_name,
            "scenario": scenario,
            "candidate_id": candidate.candidate_id,
            "variant": candidate.variant,
            "model_family": candidate.model_family,
            "is_ensemble": candidate.is_ensemble,
            "ensemble_seeds": "|".join(str(seed) for seed in seeds) if candidate.is_ensemble else "",
            **asdict(candidate),
            **threshold_row,
            **{f"oof_{key}": value for key, value in threshold_row.items() if key in {"accuracy", "macro_precision", "macro_recall", "macro_f1", "recall_low", "f1_low", "rmse", "r2", "mean_confidence"}},
            **fold_summary,
        }
        cv_rows.append(base)
        threshold_rows.append(base.copy())
    return cv_rows, threshold_rows


def near_best_select(frame: pd.DataFrame) -> pd.Series:
    ranked = frame.copy()
    best_macro = ranked["macro_f1_mean"].max()
    ranked = ranked[ranked["macro_f1_mean"] >= best_macro - 0.01].copy()
    ranked = ranked.sort_values(
        ["recall_low_mean", "f1_low_mean", "macro_f1_mean", "macro_f1_std"],
        ascending=[False, False, False, True],
    )
    return ranked.iloc[0]


def select_for_task(task_rows: pd.DataFrame) -> pd.Series:
    selected = near_best_select(task_rows)
    if selected["model_family"] == "fusion":
        sequence_rows = task_rows[task_rows["model_family"] == "sequence"]
        if not sequence_rows.empty:
            best_sequence_macro = sequence_rows["macro_f1_mean"].max()
            if selected["macro_f1_mean"] <= best_sequence_macro + 1e-9:
                selected = near_best_select(sequence_rows)
    return selected


def evaluate_locked_candidate(
    dataset_name: str,
    scenario: str,
    candidate: V29Candidate,
    selected: dict,
    config: V29RunConfig,
    train_pool: pd.DataFrame,
    locked_test: pd.DataFrame,
    seed: int,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    spec = DATASETS[dataset_name]
    prepared = prepare_for_candidate(train_pool.copy(), locked_test.copy(), dataset_name, scenario, candidate, config)
    locked_selected = transform_v28(locked_test.copy(), prepared, dataset_name, scenario)
    train_ds = make_dataset(prepared.train, prepared)
    locked_ds = make_dataset(locked_selected, prepared)
    drop_last = len(train_ds) > config.batch_size and len(train_ds) % config.batch_size == 1
    set_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=drop_last)
    train_eval_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False)
    locked_loader = DataLoader(locked_ds, batch_size=config.batch_size, shuffle=False)
    model = make_model(candidate, train_ds, prepared).to(device)
    original_labels = train_pool[spec.target_col].astype(int).to_numpy()
    model = train_model(model, train_loader, train_eval_loader, candidate, original_labels, config, device)
    probs, y_true, reg_true = predict_proba(model, locked_loader, device)
    preds = predict_with_threshold(probs, selected["threshold_low"])
    metrics = compute_required_metrics(y_true, preds, probs, y_reg_true=reg_true)
    row = {
        "dataset": dataset_name,
        "scenario": scenario,
        "candidate_id": candidate.candidate_id,
        "variant": candidate.variant,
        "prediction_mode": selected["prediction_mode"],
        "threshold_low": selected["threshold_low"],
        "seed": seed,
        "result_type": "selected_seed",
        **asdict(candidate),
        **metrics,
    }
    return row, probs, y_true, reg_true


def evaluate_locked_selected(
    dataset_name: str,
    scenario: str,
    candidate: V29Candidate,
    selected: dict,
    config: V29RunConfig,
    train_pool: pd.DataFrame,
    locked_test: pd.DataFrame,
    device: torch.device,
) -> dict:
    if not candidate.is_ensemble:
        row, _, _, _ = evaluate_locked_candidate(
            dataset_name,
            scenario,
            candidate,
            selected,
            config,
            train_pool,
            locked_test,
            config.seed,
            device,
        )
        return row

    member_probs = []
    y_true = None
    reg_true = None
    for seed in config.ensemble_seeds:
        _, probs, labels, regs = evaluate_locked_candidate(
            dataset_name,
            scenario,
            candidate,
            selected,
            config,
            train_pool,
            locked_test,
            int(seed),
            device,
        )
        member_probs.append(probs)
        y_true = labels
        reg_true = regs
    mean_probs = np.mean(np.stack(member_probs, axis=0), axis=0)
    preds = predict_with_threshold(mean_probs, selected["threshold_low"])
    metrics = compute_required_metrics(y_true, preds, mean_probs, y_reg_true=reg_true)
    return {
        "dataset": dataset_name,
        "scenario": scenario,
        "candidate_id": candidate.candidate_id,
        "variant": candidate.variant,
        "prediction_mode": selected["prediction_mode"],
        "threshold_low": selected["threshold_low"],
        "seed": config.seed,
        "result_type": "selected_ensemble",
        "ensemble_seeds": "|".join(str(seed) for seed in config.ensemble_seeds),
        **asdict(candidate),
        **metrics,
    }


def load_v28_baseline() -> pd.DataFrame:
    path = V28_DIR / "baseline_v28_locked_test_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def old_champions() -> pd.DataFrame:
    path = ABLATION_DIR / "deep_ablation_locked_test.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame = frame.sort_values(["macro_f1", "recall_low", "f1_low"], ascending=[False, False, False])
    return frame.groupby(["dataset", "scenario"], as_index=False).head(1)


def v28_locked() -> pd.DataFrame:
    path = V28_DIR / "deep_v28_locked_test_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    out = frame[columns].head(limit).copy()
    for col in out.select_dtypes(include=["float"]).columns:
        out[col] = out[col].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(out.columns) + " |"
    sep = "| " + " | ".join("---" for _ in out.columns) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in out.to_numpy()]
    return "\n".join([header, sep, *rows])


def build_comparison_table(locked_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    old = old_champions()
    v28 = v28_locked()
    for dataset, scenario in REQUIRED_TASKS:
        old_match = old[(old["dataset"] == dataset) & (old["scenario"] == scenario)]
        if not old_match.empty:
            row = old_match.iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "source": "old_champion",
                    "variant": row.get("variant", ""),
                    "prediction_mode": row.get("prediction_mode", ""),
                    "macro_f1": row.get("macro_f1", np.nan),
                    "recall_low": row.get("recall_low", np.nan),
                    "f1_low": row.get("f1_low", np.nan),
                }
            )
        v28_match = v28[(v28["dataset"] == dataset) & (v28["scenario"] == scenario)]
        if not v28_match.empty:
            row = v28_match.iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "source": "v28_selected",
                    "variant": row.get("variant", ""),
                    "prediction_mode": row.get("prediction_mode", ""),
                    "macro_f1": row.get("macro_f1", np.nan),
                    "recall_low": row.get("recall_low", np.nan),
                    "f1_low": row.get("f1_low", np.nan),
                }
            )
        v29_match = locked_df[(locked_df["dataset"] == dataset) & (locked_df["scenario"] == scenario)]
        if not v29_match.empty:
            row = v29_match.iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "source": "v29_selected",
                    "variant": row.get("variant", ""),
                    "prediction_mode": row.get("prediction_mode", ""),
                    "macro_f1": row.get("macro_f1", np.nan),
                    "recall_low": row.get("recall_low", np.nan),
                    "f1_low": row.get("f1_low", np.nan),
                }
            )
    return pd.DataFrame(rows)


def write_summary(
    cv_df: pd.DataFrame,
    locked_df: pd.DataFrame,
    ensemble_df: pd.DataFrame,
    vs_df: pd.DataFrame,
    config: V29RunConfig,
) -> None:
    comparison = build_comparison_table(locked_df)
    selection = cv_df[cv_df["selected_for_locked"] == True].copy()
    lines = [
        "# Final V29 Summary",
        "",
        "## Protocol",
        "",
        "- Controlled V29 ablation: old sequence branch, small kernel/loss changes, and light xAPI fusion only.",
        "- No ADASYN.",
        "- No student-combine.",
        "- Thresholds are tuned from OOF train-pool probabilities, never locked test.",
        "- Model selection uses CV/OOF only: Macro F1 mean, then Recall Low, F1 Low, then lower fold std.",
        "- Locked test is final evaluation only for the CV/OOF-selected row per dataset/scenario.",
        "- Regression head is not claimed; RMSE/R2 are mapped-class diagnostics only.",
        f"- Runtime config: cv_folds={config.cv_folds}, max_epochs={config.max_epochs}, patience={config.patience}, ensemble_seeds={list(config.ensemble_seeds)}.",
        "",
        "## Old Champion vs V28 vs V29",
        "",
        markdown_table(comparison, ["dataset", "scenario", "source", "variant", "prediction_mode", "macro_f1", "recall_low", "f1_low"]),
        "",
        "## V29 CV/OOF Selected Models",
        "",
        markdown_table(selection, ["dataset", "scenario", "candidate_id", "variant", "prediction_mode", "macro_f1_mean", "recall_low_mean", "f1_low_mean", "macro_f1_std"]),
        "",
        "## V29 Locked-Test Results",
        "",
        markdown_table(locked_df, ["dataset", "scenario", "candidate_id", "variant", "prediction_mode", "macro_f1", "recall_low", "f1_low", "rmse", "r2"]),
        "",
        "## V29 Ensemble OOF Results",
        "",
        markdown_table(ensemble_df, ["dataset", "scenario", "candidate_id", "prediction_mode", "macro_f1_mean", "recall_low_mean", "f1_low_mean", "macro_f1_std"]),
        "",
        "## Deep vs Baseline Same Scenario",
        "",
        markdown_table(vs_df, ["dataset", "scenario", "deep_variant", "deep_prediction_mode", "deep_macro_f1", "deep_recall_low", "deep_f1_low", "baseline_model", "baseline_macro_f1", "macro_f1_gap_deep_minus_baseline"]),
        "",
        "## Conclusion",
        "",
    ]
    if not locked_df.empty:
        best_by_task = comparison.sort_values(["dataset", "scenario", "macro_f1", "recall_low", "f1_low"], ascending=[True, True, False, False, False])
        best_by_task = best_by_task.groupby(["dataset", "scenario"], as_index=False).head(1)
        lines.append(
            "Use the best controlled deep row per dataset/scenario from the comparison table for thesis reporting. "
            "Do not replace a stronger old champion with a weaker V29 row; V29 is accepted only where its CV/OOF-selected locked result is competitive or better."
        )
        lines.append("")
        lines.append(markdown_table(best_by_task, ["dataset", "scenario", "source", "variant", "prediction_mode", "macro_f1", "recall_low", "f1_low"]))
    else:
        lines.append("No V29 locked-test rows were produced.")
    (V29_DIR / "final_v29_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_v29_experiments(tasks: list[tuple[str, str]], config: V29RunConfig) -> dict[str, pd.DataFrame]:
    ensure_v29_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running V29 on device=%s config=%s", device, config)
    cv_rows: list[dict] = []
    threshold_rows: list[dict] = []
    locked_rows: list[dict] = []
    selection_rows: list[dict] = []

    for dataset_name, scenario in tasks:
        if dataset_name == "student-combine":
            raise ValueError("student-combine is not allowed in V29 experiments.")
        train_pool, locked_test = _load_or_create_splits(dataset_name)
        candidates = candidate_grid(dataset_name)
        candidate_lookup = {candidate.candidate_id: candidate for candidate in candidates}
        task_rows: list[dict] = []
        for candidate in candidates:
            rows, threshold_candidate_rows = run_candidate_oof(dataset_name, scenario, candidate, config, train_pool, device)
            cv_rows.extend(rows)
            threshold_rows.extend(threshold_candidate_rows)
            task_rows.extend(rows)
        task_df = pd.DataFrame(task_rows)
        selected = select_for_task(task_df).to_dict()
        selected_candidate = candidate_lookup[selected["candidate_id"]]
        for row in task_rows:
            row["selected_for_locked"] = (
                row["dataset"] == dataset_name
                and row["scenario"] == scenario
                and row["candidate_id"] == selected["candidate_id"]
                and row["prediction_mode"] == selected["prediction_mode"]
            )
        selection_rows.append(selected)
        locked_rows.append(
            evaluate_locked_selected(
                dataset_name,
                scenario,
                selected_candidate,
                selected,
                config,
                train_pool,
                locked_test,
                device,
            )
        )

    cv_df = pd.DataFrame(cv_rows)
    if not cv_df.empty:
        selected_keys = {(row["dataset"], row["scenario"], row["candidate_id"], row["prediction_mode"]) for row in selection_rows}
        cv_df["selected_for_locked"] = cv_df.apply(
            lambda row: (row["dataset"], row["scenario"], row["candidate_id"], row["prediction_mode"]) in selected_keys,
            axis=1,
        )
    thresholds_df = pd.DataFrame(threshold_rows)
    locked_df = pd.DataFrame(locked_rows)
    ensemble_df = cv_df[cv_df["is_ensemble"] == True].copy() if not cv_df.empty else pd.DataFrame()

    baseline_df = load_v28_baseline()
    vs_rows = []
    for _, deep in locked_df.iterrows():
        baseline = baseline_df[(baseline_df["dataset"] == deep["dataset"]) & (baseline_df["scenario"] == deep["scenario"])]
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        vs_rows.append(
            {
                "dataset": deep["dataset"],
                "scenario": deep["scenario"],
                "deep_variant": deep["variant"],
                "deep_candidate_id": deep["candidate_id"],
                "deep_prediction_mode": deep["prediction_mode"],
                "deep_macro_f1": deep["macro_f1"],
                "deep_recall_low": deep["recall_low"],
                "deep_f1_low": deep["f1_low"],
                "baseline_model": base["model"],
                "baseline_macro_f1": base["macro_f1"],
                "baseline_recall_low": base["recall_low"],
                "baseline_f1_low": base["f1_low"],
                "macro_f1_gap_deep_minus_baseline": deep["macro_f1"] - base["macro_f1"],
                "recall_low_gap_deep_minus_baseline": deep["recall_low"] - base["recall_low"],
                "f1_low_gap_deep_minus_baseline": deep["f1_low"] - base["f1_low"],
            }
        )
    vs_df = pd.DataFrame(vs_rows)

    cv_df.to_csv(V29_DIR / "v29_cv_oof_results.csv", index=False)
    locked_df.to_csv(V29_DIR / "v29_locked_test_results.csv", index=False)
    thresholds_df.to_csv(V29_DIR / "v29_thresholds.csv", index=False)
    ensemble_df.to_csv(V29_DIR / "v29_ensemble_results.csv", index=False)
    vs_df.to_csv(V29_DIR / "v29_vs_baseline.csv", index=False)
    save_json(
        V29_DIR / "v29_config.json",
        {
            "config": asdict(config),
            "tasks": tasks,
            "selection_rule": "CV/OOF only: keep rows within 0.01 Macro F1 mean of the best, then maximize Recall Low mean, F1 Low mean, and minimize Macro F1 std.",
            "locked_test": "final evaluation only for one selected row per dataset/scenario",
            "regression_head": "not claimed",
        },
    )
    write_summary(cv_df, locked_df, ensemble_df, vs_df, config)
    return {
        "cv_oof": cv_df,
        "locked": locked_df,
        "thresholds": thresholds_df,
        "ensemble": ensemble_df,
        "vs_baseline": vs_df,
    }
