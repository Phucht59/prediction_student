from __future__ import annotations

from dataclasses import asdict, dataclass

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
from src.experiments.v28 import SEED_ENSEMBLE_11, SEED_ENSEMBLE_5, _load_or_create_splits, make_dataset, prepare_v28_fold, transform_v28
from src.experiments.v29 import (
    V29Candidate,
    V29RunConfig,
    SequenceCNNBiLSTMLight,
    load_v28_baseline,
    markdown_table,
    old_champions,
)
from src.losses_v27 import FocalLoss
from src.models_v27 import StudentHybridV27
from src.train_pipeline import calculate_class_weights
from src.utils import set_seed, setup_logger

logger = setup_logger("v30_experiments")

V30_DIR = REPORTS_DIR / "v30"
V29_DIR = REPORTS_DIR / "v29"
REQUIRED_TASKS = (("student-mat", "late"), ("xapi", "xapi"))


@dataclass(frozen=True)
class V30RunConfig(V29RunConfig):
    cv_folds: int = 3
    max_epochs: int = 50
    patience: int = 10


@dataclass(frozen=True)
class V30Candidate(V29Candidate):
    threshold_grid: str = "standard"
    calibrate_temperature: bool = False


class ContextMLPV2(nn.Module):
    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: list[int],
        hidden_dim: int = 96,
        num_classes: int = 3,
        dropout: float = 0.25,
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
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.class_head = nn.Linear(hidden_dim, num_classes)

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
        return self.class_head(hidden), None


def ensure_v30_dir() -> None:
    V30_DIR.mkdir(parents=True, exist_ok=True)


def student_mat_candidates() -> list[V30Candidate]:
    return [
        V30Candidate("old_seq_default_control", "old_seq_default", "sequence", 3, 32, 64, "ce", False, "none"),
        V30Candidate(
            "old_seq_default_ensemble11",
            "old_seq_default_ensemble11",
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
        V30Candidate("old_seq_temperature", "old_seq_default_temperature", "sequence", 3, 32, 64, "ce", False, "none", calibrate_temperature=True),
        V30Candidate("old_seq_fine_low_grid", "old_seq_default_fine_threshold", "sequence", 3, 32, 64, "ce", False, "none", threshold_grid="fine"),
        V30Candidate("old_seq_class_weight", "old_seq_default_class_weight", "sequence", 3, 32, 64, "ce", True, "none"),
    ]


def xapi_candidates() -> list[V30Candidate]:
    candidates: list[V30Candidate] = []
    for kernel in (1, 2, 3):
        for channels in (16, 32, 64):
            for hidden in (32, 64):
                candidates.append(
                    V30Candidate(
                        f"xapi_seq_k{kernel}_c{channels}_h{hidden}",
                        "xapi_sequence_only_v30",
                        "sequence",
                        kernel,
                        channels,
                        hidden,
                        "ce",
                        True,
                        "none",
                    )
                )
                candidates.append(
                    V30Candidate(
                        f"xapi_gated_k{kernel}_c{channels}_h{hidden}",
                        "xapi_gated_fusion_v30",
                        "fusion",
                        kernel,
                        channels,
                        hidden,
                        "ce",
                        True,
                        "none",
                    )
                )
    for kernel in (2, 3):
        for channels in (32, 64):
            candidates.append(
                V30Candidate(
                    f"xapi_gated_focal_k{kernel}_c{channels}_h64",
                    "xapi_gated_fusion_focal_light",
                    "fusion",
                    kernel,
                    channels,
                    64,
                    "focal_loss",
                    True,
                    "none",
                )
            )
    candidates.extend(
        [
            V30Candidate("xapi_context_mlp_v2_d15", "xapi_context_mlp_v2", "context", 1, 32, 64, "ce", True, "none", dropout=0.15, context_hidden_dim=96),
            V30Candidate("xapi_context_mlp_v2_d25", "xapi_context_mlp_v2", "context", 1, 32, 64, "ce", True, "none", dropout=0.25, context_hidden_dim=96),
            V30Candidate(
                "xapi_gated_ensemble5",
                "xapi_gated_fusion_v30_ensemble5",
                "fusion",
                3,
                32,
                64,
                "ce",
                True,
                "none",
                is_ensemble=True,
                base_candidate_id="xapi_gated_k3_c32_h64",
            ),
        ]
    )
    return candidates


def candidate_grid(dataset_name: str, scenario: str) -> list[V30Candidate]:
    if dataset_name == "student-mat" and scenario == "late":
        return student_mat_candidates()
    if dataset_name == "xapi":
        return xapi_candidates()
    return []


def seeds_for_candidate(candidate: V30Candidate, config: V30RunConfig) -> tuple[int, ...]:
    if not candidate.is_ensemble:
        return (config.seed,)
    if candidate.candidate_id == "xapi_gated_ensemble5":
        return SEED_ENSEMBLE_5
    return config.ensemble_seeds


def make_model(candidate: V30Candidate, dataset, prepared) -> nn.Module:
    if candidate.model_family == "context":
        cat_cardinalities = [len(prepared.preprocessor.label_encoders[column].classes_) for column in dataset.cat_cols]
        return ContextMLPV2(
            num_numerical=len(dataset.num_cols),
            cat_cardinalities=cat_cardinalities,
            hidden_dim=candidate.context_hidden_dim,
            dropout=candidate.dropout,
        )
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


def loss_for_candidate(candidate: V30Candidate, labels: np.ndarray, device: torch.device) -> nn.Module:
    weights = calculate_class_weights(labels, num_classes=3).to(device) if candidate.use_class_weight else None
    if candidate.loss_name == "focal_loss":
        return FocalLoss(weight=weights, gamma=1.5).to(device)
    return nn.CrossEntropyLoss(weight=weights)


def train_model(model, train_loader, val_loader, candidate: V30Candidate, labels: np.ndarray, config: V30RunConfig, device: torch.device):
    criterion = loss_for_candidate(candidate, labels, device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = None
    best_f1 = -1.0
    stale = 0
    for _ in range(config.max_epochs):
        model.train()
        for seq_x, num_x, cat_x, y, _, _ in train_loader:
            optimizer.zero_grad()
            logits = model(seq_x.to(device), num_x.to(device), cat_x.to(device))[0]
            loss = criterion(logits, y.to(device))
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


def predict_proba(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def validate_macro_f1(model, loader: DataLoader, device: torch.device) -> float:
    probs, labels, _ = predict_proba(model, loader, device)
    return float(f1_score(labels, np.argmax(probs, axis=1), average="macro", zero_division=0))


def predict_with_threshold(probabilities: np.ndarray, threshold: float | None) -> np.ndarray:
    if threshold is None or pd.isna(threshold):
        return np.argmax(probabilities, axis=1)
    non_low = np.argmax(probabilities[:, 1:], axis=1) + 1
    return np.where(probabilities[:, 0] >= float(threshold), 0, non_low)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-8, 1.0)
    scaled = np.power(clipped, 1.0 / float(temperature))
    return scaled / scaled.sum(axis=1, keepdims=True)


def threshold_values(candidate: V30Candidate) -> np.ndarray:
    if candidate.threshold_grid == "fine":
        return np.linspace(0.01, 0.99, 99)
    return np.linspace(0.05, 0.95, 37)


def tune_thresholds(y_true: np.ndarray, probabilities: np.ndarray, reg_true: np.ndarray, candidate: V30Candidate) -> list[dict]:
    rows: list[dict] = []
    temps = np.linspace(0.5, 3.0, 11) if candidate.calibrate_temperature else np.array([1.0])
    objectives = {
        "low_f1_tuned": lambda m: m["f1_low"] + 1e-3 * m["macro_f1"],
        "low_recall_priority": lambda m: 0.80 * m["recall_low"] + 0.20 * m["f1_low"],
        "balanced_low_macro": lambda m: 0.50 * m["macro_f1"] + 0.25 * m["recall_low"] + 0.25 * m["f1_low"],
    }
    best_argmax = None
    for temperature in temps:
        probs_t = apply_temperature(probabilities, float(temperature))
        preds = np.argmax(probs_t, axis=1)
        metrics = compute_required_metrics(y_true, preds, probs_t, y_reg_true=reg_true)
        row = {"prediction_mode": "argmax", "threshold_low": np.nan, "temperature": float(temperature), "selection_score": metrics["macro_f1"], **metrics}
        if best_argmax is None or row["selection_score"] > best_argmax["selection_score"]:
            best_argmax = row
    rows.append(best_argmax)
    for mode, objective in objectives.items():
        best = None
        for temperature in temps:
            probs_t = apply_temperature(probabilities, float(temperature))
            for threshold in threshold_values(candidate):
                preds = predict_with_threshold(probs_t, float(threshold))
                metrics = compute_required_metrics(y_true, preds, probs_t, y_reg_true=reg_true)
                row = {
                    "prediction_mode": mode,
                    "threshold_low": float(threshold),
                    "temperature": float(temperature),
                    "selection_score": float(objective(metrics)),
                    **metrics,
                }
                if best is None or row["selection_score"] > best["selection_score"]:
                    best = row
        rows.append(best)
    return rows


def fold_metric_summary(y_true, probabilities, reg_true, fold_ids, threshold_row):
    fold_rows = []
    probs_t = apply_temperature(probabilities, float(threshold_row.get("temperature", 1.0)))
    for fold in sorted(np.unique(fold_ids)):
        mask = fold_ids == fold
        preds = predict_with_threshold(probs_t[mask], threshold_row["threshold_low"])
        fold_rows.append(compute_required_metrics(y_true[mask], preds, probs_t[mask], y_reg_true=reg_true[mask]))
    summary: dict[str, float] = {}
    for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "recall_low", "f1_low", "rmse", "r2"):
        values = np.asarray([row[metric] for row in fold_rows], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
    return summary


def train_candidate_fold(train_pool, dataset_name, scenario, train_idx, val_idx, candidate: V30Candidate, config: V30RunConfig, fold_seed: int, device):
    spec = DATASETS[dataset_name]
    prepared = prepare_v28_fold(train_pool.iloc[train_idx].copy(), train_pool.iloc[val_idx].copy(), dataset_name, scenario, candidate, config)
    train_ds = make_dataset(prepared.train, prepared)
    val_ds = make_dataset(prepared.validation, prepared)
    set_seed(fold_seed)
    drop_last = len(train_ds) > config.batch_size and len(train_ds) % config.batch_size == 1
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    model = make_model(candidate, train_ds, prepared).to(device)
    labels = train_pool.iloc[train_idx][spec.target_col].astype(int).to_numpy()
    model = train_model(model, train_loader, val_loader, candidate, labels, config, device)
    return predict_proba(model, val_loader, device)


def run_candidate_oof(dataset_name: str, scenario: str, candidate: V30Candidate, config: V30RunConfig, train_pool: pd.DataFrame, device) -> list[dict]:
    spec = DATASETS[dataset_name]
    labels = train_pool[spec.target_col].astype(int).to_numpy()
    folds = list(StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed).split(train_pool, labels))
    seeds = seeds_for_candidate(candidate, config)
    oof_probs = np.zeros((len(train_pool), 3), dtype=float)
    oof_targets = np.zeros(len(train_pool), dtype=int)
    oof_reg_true = np.zeros(len(train_pool), dtype=float)
    fold_ids = np.zeros(len(train_pool), dtype=int)
    for seed_index, seed in enumerate(seeds, start=1):
        logger.info("OOF task=%s/%s candidate=%s seed=%s/%s", dataset_name, scenario, candidate.candidate_id, seed_index, len(seeds))
        for fold, (train_idx, val_idx) in enumerate(folds, start=1):
            probs, y_val, reg_true = train_candidate_fold(train_pool, dataset_name, scenario, train_idx, val_idx, candidate, config, int(seed) + 100 * fold, device)
            oof_probs[val_idx] += probs / len(seeds)
            oof_targets[val_idx] = y_val
            oof_reg_true[val_idx] = reg_true
            fold_ids[val_idx] = fold
    rows = []
    for threshold_row in tune_thresholds(oof_targets, oof_probs, oof_reg_true, candidate):
        fold_summary = fold_metric_summary(oof_targets, oof_probs, oof_reg_true, fold_ids, threshold_row)
        rows.append(
            {
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
        )
    return rows


def select_for_task(frame: pd.DataFrame) -> pd.Series:
    ranked = frame.copy()
    best_macro = ranked["macro_f1_mean"].max()
    ranked = ranked[ranked["macro_f1_mean"] >= best_macro - 0.01].copy()
    ranked = ranked.sort_values(["recall_low_mean", "f1_low_mean", "macro_f1_mean", "macro_f1_std"], ascending=[False, False, False, True])
    return ranked.iloc[0]


def evaluate_locked_candidate(dataset_name, scenario, candidate: V30Candidate, selected: dict, config: V30RunConfig, train_pool, locked_test, seed: int, device):
    spec = DATASETS[dataset_name]
    prepared = prepare_v28_fold(train_pool.copy(), locked_test.copy(), dataset_name, scenario, candidate, config)
    locked_selected = transform_v28(locked_test.copy(), prepared, dataset_name, scenario)
    train_ds = make_dataset(prepared.train, prepared)
    locked_ds = make_dataset(locked_selected, prepared)
    set_seed(seed)
    drop_last = len(train_ds) > config.batch_size and len(train_ds) % config.batch_size == 1
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=drop_last)
    train_eval_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False)
    locked_loader = DataLoader(locked_ds, batch_size=config.batch_size, shuffle=False)
    model = make_model(candidate, train_ds, prepared).to(device)
    labels = train_pool[spec.target_col].astype(int).to_numpy()
    model = train_model(model, train_loader, train_eval_loader, candidate, labels, config, device)
    probs, y_true, reg_true = predict_proba(model, locked_loader, device)
    probs_t = apply_temperature(probs, float(selected.get("temperature", 1.0)))
    preds = predict_with_threshold(probs_t, selected["threshold_low"])
    metrics = compute_required_metrics(y_true, preds, probs_t, y_reg_true=reg_true)
    return {
        "dataset": dataset_name,
        "scenario": scenario,
        "candidate_id": candidate.candidate_id,
        "variant": candidate.variant,
        "prediction_mode": selected["prediction_mode"],
        "threshold_low": selected["threshold_low"],
        "temperature": selected.get("temperature", 1.0),
        "seed": seed,
        "result_type": "selected_seed",
        **asdict(candidate),
        **metrics,
    }, probs_t, y_true, reg_true


def evaluate_locked_selected(dataset_name, scenario, candidate: V30Candidate, selected: dict, config: V30RunConfig, train_pool, locked_test, device):
    if not candidate.is_ensemble:
        row, _, _, _ = evaluate_locked_candidate(dataset_name, scenario, candidate, selected, config, train_pool, locked_test, config.seed, device)
        return row
    member_probs = []
    y_true = None
    reg_true = None
    member_seeds = seeds_for_candidate(candidate, config)
    for seed in member_seeds:
        _, probs, labels, regs = evaluate_locked_candidate(dataset_name, scenario, candidate, selected, config, train_pool, locked_test, int(seed), device)
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
        "temperature": selected.get("temperature", 1.0),
        "seed": config.seed,
        "result_type": "selected_ensemble",
        "ensemble_seeds": "|".join(str(seed) for seed in member_seeds),
        **asdict(candidate),
        **metrics,
    }


def load_v29_locked() -> pd.DataFrame:
    path = V29_DIR / "v29_locked_test_results.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def v28_locked() -> pd.DataFrame:
    path = REPORTS_DIR / "v28" / "deep_v28_locked_test_results.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_comparison(locked_df: pd.DataFrame) -> pd.DataFrame:
    old = old_champions()
    v29 = load_v29_locked()
    rows = []
    for dataset, scenario in REQUIRED_TASKS:
        if dataset == "student-mat":
            old_match = old[(old["dataset"] == dataset) & (old["scenario"] == scenario)]
            if not old_match.empty:
                row = old_match.iloc[0]
                rows.append({"dataset": dataset, "scenario": scenario, "source": "old_champion", "variant": row["variant"], "prediction_mode": row["prediction_mode"], "macro_f1": row["macro_f1"], "recall_low": row["recall_low"], "f1_low": row["f1_low"]})
        else:
            v28 = v28_locked()
            v28_match = v28[(v28["dataset"] == dataset) & (v28["scenario"] == scenario)]
            if not v28_match.empty:
                row = v28_match.iloc[0]
                rows.append({"dataset": dataset, "scenario": scenario, "source": "v28_best_deep", "variant": row["variant"], "prediction_mode": row["prediction_mode"], "macro_f1": row["macro_f1"], "recall_low": row["recall_low"], "f1_low": row["f1_low"]})
        v29_match = v29[(v29["dataset"] == dataset) & (v29["scenario"] == scenario)]
        if not v29_match.empty:
            row = v29_match.iloc[0]
            rows.append({"dataset": dataset, "scenario": scenario, "source": "v29_selected", "variant": row["variant"], "prediction_mode": row["prediction_mode"], "macro_f1": row["macro_f1"], "recall_low": row["recall_low"], "f1_low": row["f1_low"]})
        v30_match = locked_df[(locked_df["dataset"] == dataset) & (locked_df["scenario"] == scenario)]
        if not v30_match.empty:
            row = v30_match.iloc[0]
            rows.append({"dataset": dataset, "scenario": scenario, "source": "v30_selected", "variant": row["variant"], "prediction_mode": row["prediction_mode"], "macro_f1": row["macro_f1"], "recall_low": row["recall_low"], "f1_low": row["f1_low"]})
    return pd.DataFrame(rows)


def write_summary(cv_df: pd.DataFrame, locked_df: pd.DataFrame, vs_df: pd.DataFrame, config: V30RunConfig) -> None:
    comparison = build_comparison(locked_df)
    selected = cv_df[cv_df["selected_for_locked"] == True].copy()
    lines = [
        "# Final V30 Summary",
        "",
        "## Protocol",
        "",
        "- Scope: only student-mat late and xAPI.",
        "- No ADASYN.",
        "- No student-combine.",
        "- Threshold and temperature choices are tuned from OOF train-pool probabilities only.",
        "- Model selection uses CV/OOF only. Locked test is final evaluation only.",
        "- Regression head is not claimed.",
        f"- Runtime config: cv_folds={config.cv_folds}, max_epochs={config.max_epochs}, patience={config.patience}, student_mat_ensemble_seeds={list(config.ensemble_seeds)}, xapi_ensemble_seeds={list(SEED_ENSEMBLE_5)}.",
        "",
        "## Old/V29/V30 Comparison",
        "",
        markdown_table(comparison, ["dataset", "scenario", "source", "variant", "prediction_mode", "macro_f1", "recall_low", "f1_low"]),
        "",
        "## V30 CV/OOF Selected Models",
        "",
        markdown_table(selected, ["dataset", "scenario", "candidate_id", "variant", "prediction_mode", "temperature", "macro_f1_mean", "recall_low_mean", "f1_low_mean", "macro_f1_std"]),
        "",
        "## V30 Locked-Test Results",
        "",
        markdown_table(locked_df, ["dataset", "scenario", "candidate_id", "variant", "prediction_mode", "temperature", "macro_f1", "recall_low", "f1_low", "rmse", "r2"]),
        "",
        "## Deep vs Baseline Same Scenario",
        "",
        markdown_table(vs_df, ["dataset", "scenario", "deep_variant", "deep_prediction_mode", "deep_macro_f1", "deep_recall_low", "deep_f1_low", "baseline_model", "baseline_macro_f1", "macro_f1_gap_deep_minus_baseline"]),
        "",
        "## Conclusion",
        "",
    ]
    best_by_task = comparison.sort_values(["dataset", "scenario", "macro_f1", "recall_low", "f1_low"], ascending=[True, True, False, False, False])
    best_by_task = best_by_task.groupby(["dataset", "scenario"], as_index=False).head(1)
    lines.append("Final champion is changed only if the CV/OOF-selected V30 row improves the existing best locked-test deep row for the same dataset/scenario.")
    lines.append("")
    lines.append(markdown_table(best_by_task, ["dataset", "scenario", "source", "variant", "prediction_mode", "macro_f1", "recall_low", "f1_low"]))
    (V30_DIR / "final_v30_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_v30_experiments(config: V30RunConfig) -> dict[str, pd.DataFrame]:
    ensure_v30_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running V30 on device=%s config=%s", device, config)
    cv_rows = []
    locked_rows = []
    selection_rows = []
    for dataset_name, scenario in REQUIRED_TASKS:
        train_pool, locked_test = _load_or_create_splits(dataset_name)
        candidates = candidate_grid(dataset_name, scenario)
        lookup = {candidate.candidate_id: candidate for candidate in candidates}
        task_rows = []
        for candidate in candidates:
            rows = run_candidate_oof(dataset_name, scenario, candidate, config, train_pool, device)
            cv_rows.extend(rows)
            task_rows.extend(rows)
        task_df = pd.DataFrame(task_rows)
        selected = select_for_task(task_df).to_dict()
        selection_rows.append(selected)
        locked_rows.append(evaluate_locked_selected(dataset_name, scenario, lookup[selected["candidate_id"]], selected, config, train_pool, locked_test, device))
    cv_df = pd.DataFrame(cv_rows)
    selected_keys = {(row["dataset"], row["scenario"], row["candidate_id"], row["prediction_mode"], float(row.get("temperature", 1.0))) for row in selection_rows}
    cv_df["selected_for_locked"] = cv_df.apply(lambda row: (row["dataset"], row["scenario"], row["candidate_id"], row["prediction_mode"], float(row.get("temperature", 1.0))) in selected_keys, axis=1)
    locked_df = pd.DataFrame(locked_rows)
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
    cv_df.to_csv(V30_DIR / "v30_cv_oof_results.csv", index=False)
    locked_df.to_csv(V30_DIR / "v30_locked_test_results.csv", index=False)
    vs_df.to_csv(V30_DIR / "v30_vs_baseline.csv", index=False)
    save_json(
        V30_DIR / "v30_config.json",
        {
            "config": asdict(config),
            "tasks": REQUIRED_TASKS,
            "selection": "CV/OOF only: Macro F1 mean with 0.01 near-best tolerance, then Recall Low, F1 Low, lower fold std.",
            "locked_test": "final evaluation only",
            "regression_head": "not claimed",
        },
    )
    write_summary(cv_df, locked_df, vs_df, config)
    return {"cv_oof": cv_df, "locked": locked_df, "vs_baseline": vs_df}
