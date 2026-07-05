"""Validation-only model selection for the CNN-BiLSTM + Context MLP pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.config import DEFAULT_SEED, FIXED_SEEDS, TrainingConfig
from src.data_pipeline import (
    DataPreprocessor,
    FeatureSelector,
    StudentDataset,
    apply_feature_engineering,
    get_context_excluded_columns,
    get_sequence_columns,
)
from src.models import FocalLoss, create_model
from src.train_pipeline import calculate_class_weights, train_model
from src.utils import set_seed, setup_logger

logger = setup_logger("model_selection")


@dataclass(frozen=True)
class FoldModelResult:
    fold_index: int
    seed: int
    row_positions: list[int]
    probabilities: np.ndarray
    predictions: np.ndarray
    true_labels: np.ndarray
    selected_features: list[str]
    numerical_cols: list[str]
    categorical_cols: list[str]


def student_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Optuna space for student performance, evaluated only on CV folds."""
    loss_name = trial.suggest_categorical("loss", ["weighted_ce", "focal"])
    params: dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 2e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "oversample_method": trial.suggest_categorical("oversample_method", ["none", "smote", "smotenc", "adasyn"]),
        "smote_ratio": trial.suggest_float("smote_ratio", 0.35, 1.0),
        "resampling_k_neighbors": trial.suggest_int("resampling_k_neighbors", 2, 7),
        "cnn_channels": trial.suggest_categorical("cnn_channels", [16, 32, 64, 96]),
        "cnn_kernel_size": trial.suggest_categorical("cnn_kernel_size", [2, 3, 5]),
        "lstm_hidden_dim": trial.suggest_categorical("lstm_hidden_dim", [32, 64, 96]),
        "context_hidden_dim": trial.suggest_categorical("context_hidden_dim", [32, 64, 128]),
        "fusion_hidden_dim": trial.suggest_categorical("fusion_hidden_dim", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.55),
        "sequence_dropout": trial.suggest_float("sequence_dropout", 0.05, 0.55),
        "context_dropout": trial.suggest_float("context_dropout", 0.05, 0.5),
        "fusion_dropout": trial.suggest_float("fusion_dropout", 0.05, 0.55),
        "loss": loss_name,
        "max_epochs": trial.suggest_categorical("max_epochs", [60, 80, 100]),
        "patience": trial.suggest_categorical("patience", [12, 15, 20]),
        "scheduler_patience": trial.suggest_categorical("scheduler_patience", [4, 5, 7]),
    }
    if loss_name == "focal":
        params["focal_gamma"] = trial.suggest_float("focal_gamma", 1.0, 3.0)
    return params


def make_folds(train_pool: pd.DataFrame, target_col: str, n_splits: int = 5, seed: int = DEFAULT_SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = train_pool[target_col].astype(int).to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train_idx, val_idx) for train_idx, val_idx in splitter.split(train_pool, labels)]


def build_training_config(params: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        max_epochs=int(params.get("max_epochs", 100)),
        patience=int(params.get("patience", 15)),
        scheduler_patience=int(params.get("scheduler_patience", 5)),
    )


def _criterion(spec, params: dict[str, Any], class_weights: torch.Tensor):
    if spec.kind == "xapi":
        return nn.BCEWithLogitsLoss()
    if params.get("loss") == "focal" or "focal_gamma" in params:
        return FocalLoss(weight=class_weights, gamma=float(params.get("focal_gamma", 2.0)))
    return nn.CrossEntropyLoss(weight=class_weights)


def fit_fold_predict_proba(
    *,
    train_fold: pd.DataFrame,
    validation_fold: pd.DataFrame,
    spec,
    params: dict[str, Any],
    seed: int,
    fold_index: int,
    ablation_mode: str = "hybrid",
) -> FoldModelResult:
    """Fit only on a training fold and predict probabilities for its validation fold."""
    set_seed(seed)
    train_engineered = apply_feature_engineering(train_fold.copy(), spec.kind)
    validation_engineered = apply_feature_engineering(validation_fold.copy(), spec.kind)

    preprocessor = DataPreprocessor(
        target_col=spec.target_col,
        oversample_method=params.get("oversample_method", "none"),
        smote_ratio=float(params.get("smote_ratio", 1.0)),
        resampling_k_neighbors=int(params.get("resampling_k_neighbors", 5)),
    )
    train_prepared = preprocessor.fit_transform(train_engineered, apply_oversampling=True)
    validation_prepared = preprocessor.transform(validation_engineered)

    selector = FeatureSelector(
        target_col=spec.target_col,
        use_feature_selection=True,
        required_features=get_sequence_columns(spec.kind),
    )
    train_selected = selector.fit_transform(
        train_prepared,
        preprocessor.numerical_cols,
        preprocessor.categorical_cols,
    )
    validation_selected = selector.transform(validation_prepared)

    model_params = dict(params)
    model_params["ablation_mode"] = ablation_mode
    train_dataset = StudentDataset(train_selected, spec.kind, spec.target_col, preprocessor.numerical_cols, preprocessor.categorical_cols)
    validation_dataset = StudentDataset(validation_selected, spec.kind, spec.target_col, preprocessor.numerical_cols, preprocessor.categorical_cols)
    batch_size = int(params.get("batch_size", 32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=len(train_dataset) > batch_size)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cat_cardinalities = [len(preprocessor.label_encoders[column].classes_) for column in train_dataset.cat_cols]
    model = create_model(spec.kind, model_params, len(train_dataset.num_cols), cat_cardinalities).to(device)
    class_weights = calculate_class_weights(train_fold[spec.target_col].astype(int).to_numpy(), num_classes=3).to(device)
    criterion = _criterion(spec, params, class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    model, _, _ = train_model(model, train_loader, validation_loader, criterion, optimizer, build_training_config(params), device)

    probabilities = []
    with torch.no_grad():
        model.eval()
        for batch in validation_loader:
            seq_x, num_x, cat_x, _, _ = batch[:5]
            batch_probabilities = model.predict_proba(seq_x.to(device), num_x.to(device), cat_x.to(device))
            probabilities.extend(batch_probabilities.cpu().numpy())
    probability_array = np.asarray(probabilities)
    predictions = np.argmax(probability_array, axis=1)
    return FoldModelResult(
        fold_index=fold_index,
        seed=seed,
        row_positions=list(validation_fold.index.astype(int)),
        probabilities=probability_array,
        predictions=np.asarray(predictions, dtype=int),
        true_labels=validation_fold[spec.target_col].astype(int).to_numpy(),
        selected_features=list(selector.selected_features),
        numerical_cols=list(train_dataset.num_cols),
        categorical_cols=list(train_dataset.cat_cols),
    )


def objective_mean_cv_f1(
    trial: optuna.Trial,
    train_pool: pd.DataFrame,
    spec,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    params = student_search_space(trial)
    fold_scores = []
    for fold_index, (train_idx, val_idx) in enumerate(folds):
        result = fit_fold_predict_proba(
            train_fold=train_pool.iloc[train_idx].copy(),
            validation_fold=train_pool.iloc[val_idx].copy(),
            spec=spec,
            params=params,
            seed=DEFAULT_SEED + trial.number * len(folds) + fold_index,
            fold_index=fold_index,
        )
        score = f1_score(result.true_labels, result.predictions, average="macro", zero_division=0)
        fold_scores.append(float(score))
        trial.report(float(np.mean(fold_scores)), step=fold_index)
        if trial.should_prune():
            raise optuna.TrialPruned()
    trial.set_user_attr("fold_f1_macro", fold_scores)
    return float(np.mean(fold_scores))


def run_optuna_cv_search(
    train_pool: pd.DataFrame,
    spec,
    *,
    n_trials: int,
    n_splits: int = 5,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[np.ndarray, np.ndarray]]]:
    folds = make_folds(train_pool, spec.target_col, n_splits=n_splits, seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    )
    study.optimize(lambda trial: objective_mean_cv_f1(trial, train_pool, spec, folds), n_trials=n_trials)
    trial_history = [
        {
            "number": trial.number,
            "state": str(trial.state),
            "value": None if trial.value is None else float(trial.value),
            "params": dict(trial.params),
            "fold_f1_macro": trial.user_attrs.get("fold_f1_macro"),
        }
        for trial in study.trials
    ]
    return {"best_cv_f1_macro": float(study.best_value), "best_params": dict(study.best_params)}, trial_history, folds


def collect_oof_by_seed(
    train_pool: pd.DataFrame,
    spec,
    params: dict[str, Any],
    folds: list[tuple[np.ndarray, np.ndarray]],
    seeds: list[int],
    *,
    ablation_mode: str = "hybrid",
) -> dict[str, Any]:
    y_true = train_pool[spec.target_col].astype(int).to_numpy()
    seed_probabilities: dict[int, np.ndarray] = {
        int(seed): np.zeros((len(train_pool), 3), dtype=float)
        for seed in seeds
    }
    fold_ids = np.zeros(len(train_pool), dtype=int)
    fold_reports = []
    selected_features: dict[str, list[list[str]]] = {str(seed): [] for seed in seeds}
    for fold_index, (train_idx, val_idx) in enumerate(folds):
        fold_ids[val_idx] = fold_index
        for seed in seeds:
            result = fit_fold_predict_proba(
                train_fold=train_pool.iloc[train_idx].copy(),
                validation_fold=train_pool.iloc[val_idx].copy(),
                spec=spec,
                params=params,
                seed=int(seed),
                fold_index=fold_index,
                ablation_mode=ablation_mode,
            )
            seed_probabilities[int(seed)][val_idx] = result.probabilities
            selected_features[str(seed)].append(result.selected_features)
            fold_reports.append(
                {
                    "fold": fold_index,
                    "seed": int(seed),
                    "f1_macro": float(f1_score(result.true_labels, result.predictions, average="macro", zero_division=0)),
                    "accuracy": float(accuracy_score(result.true_labels, result.predictions)),
                    "selected_features": result.selected_features,
                }
            )
    return {
        "y_true": y_true,
        "fold_ids": fold_ids,
        "seed_probabilities": seed_probabilities,
        "fold_reports": fold_reports,
        "selected_features_by_seed": selected_features,
    }


def apply_threshold_policy(probabilities: np.ndarray, threshold_policy: dict[str, Any] | None) -> np.ndarray:
    if not threshold_policy or threshold_policy.get("type") == "argmax":
        return np.argmax(probabilities, axis=1).astype(int)
    if threshold_policy.get("type") != "class_thresholds":
        raise ValueError(f"Unsupported threshold policy: {threshold_policy}")
    thresholds = np.asarray(threshold_policy["thresholds"], dtype=float)
    predictions = []
    for row in probabilities:
        candidates = np.where(row >= thresholds)[0]
        if len(candidates) == 0:
            predictions.append(int(np.argmax(row)))
        else:
            ratios = row[candidates] / np.maximum(thresholds[candidates], 1e-8)
            predictions.append(int(candidates[int(np.argmax(ratios))]))
    return np.asarray(predictions, dtype=int)


def optimize_class_thresholds(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    *,
    source: str,
    grid: list[float] | None = None,
) -> dict[str, Any]:
    if source != "oof_validation":
        raise ValueError("Threshold optimization must use OOF validation probabilities only.")
    grid = grid or [round(value, 2) for value in np.arange(0.25, 0.76, 0.05)]
    best_score = -1.0
    best_thresholds = [0.0, 0.0, 0.0]
    for low in grid:
        for medium in grid:
            for high in grid:
                policy = {"type": "class_thresholds", "thresholds": [low, medium, high]}
                predictions = apply_threshold_policy(probabilities, policy)
                score = f1_score(y_true, predictions, average="macro", zero_division=0)
                if score > best_score:
                    best_score = float(score)
                    best_thresholds = [float(low), float(medium), float(high)]
    return {
        "type": "class_thresholds",
        "source": source,
        "thresholds": best_thresholds,
        "optimized_metric": "F1-Macro",
        "oof_score": best_score,
    }


def combine_seed_probabilities(
    seed_probabilities: dict[int, np.ndarray],
    *,
    method: str,
    seed_list: list[int] | None = None,
    weights: dict[int, float] | None = None,
) -> np.ndarray:
    seeds = [int(seed) for seed in (seed_list or sorted(seed_probabilities))]
    stack = np.stack([seed_probabilities[seed] for seed in seeds], axis=0)
    if method == "mean_probability":
        return np.mean(stack, axis=0)
    if method == "median_probability":
        return np.median(stack, axis=0)
    if method == "weighted_probability":
        if not weights:
            raise ValueError("weighted_probability requires weights.")
        weight_array = np.asarray([float(weights[seed]) for seed in seeds], dtype=float)
        weight_array = weight_array / weight_array.sum()
        return np.tensordot(weight_array, stack, axes=(0, 0))
    if method == "majority_vote":
        votes = np.argmax(stack, axis=2)
        probabilities = np.zeros_like(stack[0])
        for row_index in range(votes.shape[1]):
            counts = np.bincount(votes[:, row_index], minlength=stack.shape[2]).astype(float)
            probabilities[row_index] = counts / counts.sum()
        return probabilities
    raise ValueError(f"Unsupported ensemble method: {method}")


def metric_summary(y_true: np.ndarray, probabilities: np.ndarray, threshold_policy: dict[str, Any] | None, fold_ids: np.ndarray) -> dict[str, Any]:
    predictions = apply_threshold_policy(probabilities, threshold_policy)
    fold_scores = [
        float(f1_score(y_true[fold_ids == fold], predictions[fold_ids == fold], average="macro", zero_division=0))
        for fold in sorted(set(fold_ids.tolist()))
    ]
    precision, recall, f1, support = precision_recall_fscore_support(y_true, predictions, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "fold_f1_macro": fold_scores,
        "cv_f1_macro_mean": float(np.mean(fold_scores)),
        "cv_f1_macro_std": float(np.std(fold_scores)),
        "cv_f1_macro_min": float(np.min(fold_scores)),
        "cv_f1_macro_max": float(np.max(fold_scores)),
        "class_report": {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate([0, 1, 2])
        },
    }


def evaluate_ensemble_strategies(oof: dict[str, Any], seeds: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y_true = oof["y_true"]
    fold_ids = oof["fold_ids"]
    seed_probabilities = oof["seed_probabilities"]
    seed_scores = {
        seed: f1_score(y_true, np.argmax(seed_probabilities[seed], axis=1), average="macro", zero_division=0)
        for seed in seeds
    }
    ranked_seeds = [seed for seed, _ in sorted(seed_scores.items(), key=lambda item: item[1], reverse=True)]
    rows = []

    def append_strategy(name: str, method: str, seed_list: list[int], threshold_policy: dict[str, Any] | None, weights: dict[int, float] | None = None):
        probabilities = combine_seed_probabilities(seed_probabilities, method=method, seed_list=seed_list, weights=weights)
        summary = metric_summary(y_true, probabilities, threshold_policy, fold_ids)
        rows.append(
            {
                "strategy_name": name,
                "ensemble_method": method,
                "seed_list": seed_list,
                "seed_weights": weights,
                "seed_count": len(seed_list),
                "threshold_policy": threshold_policy or {"type": "argmax"},
                **summary,
            }
        )

    for seed in seeds:
        append_strategy(f"single_seed_{seed}", "mean_probability", [seed], None)
    append_strategy("all_seed_mean_probability_argmax", "mean_probability", seeds, None)
    append_strategy("all_seed_majority_vote_argmax", "majority_vote", seeds, None)
    append_strategy("all_seed_median_probability_argmax", "median_probability", seeds, None)
    for k in [3, 5, 7]:
        append_strategy(f"top_{k}_seed_mean_probability_argmax", "mean_probability", ranked_seeds[:k], None)
    weights = {seed: max(float(seed_scores[seed]), 1e-8) for seed in seeds}
    append_strategy("all_seed_weighted_probability_argmax", "weighted_probability", seeds, None, weights=weights)

    base_probabilities = combine_seed_probabilities(seed_probabilities, method="mean_probability", seed_list=seeds)
    threshold_policy = optimize_class_thresholds(base_probabilities, y_true, source="oof_validation")
    append_strategy("all_seed_mean_probability_oof_thresholds", "mean_probability", seeds, threshold_policy)

    rows = sorted(rows, key=lambda row: (-row["cv_f1_macro_mean"], row["cv_f1_macro_std"], row["seed_count"]))
    selected = rows[0]
    for candidate in rows[1:]:
        if selected["cv_f1_macro_mean"] - candidate["cv_f1_macro_mean"] <= 0.005:
            if (
                candidate["cv_f1_macro_std"] < selected["cv_f1_macro_std"]
                or (
                    abs(candidate["cv_f1_macro_std"] - selected["cv_f1_macro_std"]) <= 1e-12
                    and candidate["seed_count"] < selected["seed_count"]
                )
            ):
                selected = candidate
    return rows, selected


def write_json(path: Path, payload: Any) -> None:
    def default(value: Any):
        if isinstance(value, (datetime, date, Path)):
            return str(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=default), encoding="utf-8")
