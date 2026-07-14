"""Validation-only model selection for the CNN-BiLSTM classifier pipeline."""

from __future__ import annotations

import json
import math
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
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
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
from src.estimator_factory import (
    ResolvedConfigError,
    StudentEstimatorFactory,
    resolve_student_config,
    validate_resolved_config,
)
from src.models import FocalLoss, create_model
from src.train_pipeline import train_fixed_epochs, train_model, train_model_fixed_lr
from src.utils import set_seed, setup_logger
from src.evaluation.protocol import validate_scenario_features

logger = setup_logger("model_selection")


def loader_statistics(dataset_size: int, batch_size: int, drop_last_train: bool) -> dict[str, int | bool]:
    """Return deterministic DataLoader accounting for training diagnostics."""
    if dataset_size < 1 or batch_size < 1:
        raise ValueError("dataset_size and batch_size must be positive.")
    remainder = dataset_size % batch_size
    consumed = dataset_size - (remainder if drop_last_train and remainder else 0)
    return {
        "dataset_size": dataset_size,
        "batch_size": batch_size,
        "n_batches": dataset_size // batch_size if drop_last_train else math.ceil(dataset_size / batch_size),
        "final_batch_size": remainder if remainder else min(batch_size, dataset_size),
        "samples_consumed_per_epoch": consumed,
        "samples_dropped_per_epoch": dataset_size - consumed,
        "drop_last_train": bool(drop_last_train),
    }


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
    train_row_positions: list[int]
    early_stop_row_positions: list[int]
    refit_state_dict: dict[str, torch.Tensor] | None = None
    training_diagnostics: dict[str, Any] | None = None
    shape_diagnostics: dict[str, Any] | None = None
    resolved_config: dict[str, Any] | None = None
    refit_preprocessor: Any | None = None
    refit_selector: Any | None = None


@dataclass
class FittedEstimatorResult:
    """A full-training-partition refit produced by the shared estimator factory."""

    model: nn.Module
    preprocessor: Any
    selector: Any
    selected_features: list[str]
    numerical_cols: list[str]
    categorical_cols: list[str]
    train_row_positions: list[int]
    early_stop_row_positions: list[int]
    refit_state_dict: dict[str, torch.Tensor]
    training_diagnostics: dict[str, Any]
    shape_diagnostics: dict[str, Any]
    resolved_config: dict[str, Any]


def _row_positions(frame: pd.DataFrame) -> list[int]:
    return list(frame.index.astype(int))


def student_search_space(
    trial: optuna.Trial,
    *,
    architecture_variant: str = "cnn_bilstm",
    fair_comparison: bool = False,
    drop_last_train: bool = False,
) -> dict[str, Any]:
    """Optuna space for student performance, evaluated only on CV folds."""
    loss_name = "cross_entropy" if fair_comparison else trial.suggest_categorical("loss", ["cross_entropy", "focal"])
    params: dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 2e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "oversample_method": "none" if fair_comparison else trial.suggest_categorical("oversample_method", ["none", "smote"]),
        "class_weight_mode": "none" if fair_comparison else trial.suggest_categorical("class_weight_mode", ["none", "balanced"]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.55),
        "sequence_dropout": trial.suggest_float("sequence_dropout", 0.05, 0.55),
        "loss": loss_name,
        "max_epochs": trial.suggest_categorical("max_epochs", [40, 60]),
        "patience": trial.suggest_categorical("patience", [8, 12]),
    }
    if not fair_comparison:
        params["smote_ratio"] = trial.suggest_float("smote_ratio", 0.35, 1.0)
        params["resampling_k_neighbors"] = trial.suggest_int("resampling_k_neighbors", 2, 7)
    # Do not spend a candidate's search budget on dimensions its architecture
    # does not contain.  Shared components retain identical ranges.
    if architecture_variant != "bilstm_only":
        params["cnn_channels"] = trial.suggest_categorical("cnn_channels", [8, 16, 32])
        params["cnn_kernel_size"] = trial.suggest_categorical("cnn_kernel_size", [1])
    if architecture_variant != "cnn_only":
        params["lstm_hidden_dim"] = trial.suggest_categorical("lstm_hidden_dim", [8, 16, 32])
    if loss_name == "focal":
        params["focal_gamma"] = trial.suggest_float("focal_gamma", 1.0, 3.0)
    # The architecture is fixed per candidate before inner-CV.  It is not an
    # Optuna choice, so comparing candidates cannot select an architecture by
    # looking at an outer fold or the locked test set.
    params["architecture_variant"] = architecture_variant
    # Canonical configs retain every constructor constant, including dimensions
    # that are inactive for a parameter-matched ablation architecture.
    params.setdefault("cnn_channels", 1)
    params.setdefault("cnn_kernel_size", 1)
    params.setdefault("lstm_hidden_dim", 1)
    trial_suggestions = dict(getattr(trial, "params", {}))
    if not trial_suggestions:
        # Lightweight contract-test trial doubles do not expose Optuna's
        # ``params`` property.  Real Optuna trials always take the first path.
        trial_suggestions = {
            key: value
            for key, value in params.items()
            if key != "architecture_variant"
            and not (architecture_variant == "cnn_only" and key == "lstm_hidden_dim")
            and not (architecture_variant == "bilstm_only" and key in {"cnn_channels", "cnn_kernel_size"})
        }
    return resolve_student_config(
        params,
        architecture_variant=architecture_variant,
        suggested_parameters=trial_suggestions,
        scheduler_type="fixed_lr",
        swa_enabled=False,
        drop_last_train=drop_last_train,
        evidence_role="optuna_candidate_corrected_estimator",
    )


def make_folds(train_pool: pd.DataFrame, target_col: str, n_splits: int = 5, seed: int = DEFAULT_SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = train_pool[target_col].astype(int).to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train_idx, val_idx) for train_idx, val_idx in splitter.split(train_pool, labels)]


def build_training_config(params: dict[str, Any]) -> TrainingConfig:
    validate_resolved_config(params)
    return TrainingConfig(
        max_epochs=int(params["max_epochs"]),
        patience=int(params["patience"]),
        scheduler_patience=int(params["scheduler"].get("parameters", {}).get("patience", 1)),
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )


def _criterion(spec, params: dict[str, Any], class_weights: torch.Tensor):
    validate_resolved_config(params)
    if spec.kind == "xapi":
        return nn.BCEWithLogitsLoss()
    use_class_weights = params["class_weight_mode"] == "balanced"
    effective_weights = class_weights if use_class_weights else None
    if params["loss"] == "focal":
        return FocalLoss(weight=effective_weights, gamma=float(params["focal_gamma"]))
    return nn.CrossEntropyLoss(weight=effective_weights)


def split_model_train_and_early_stop(
    train_fold: pd.DataFrame,
    target_col: str,
    *,
    seed: int,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split only the fold-training partition for early stopping.

    The scoring fold passed to ``fit_fold_predict_proba`` must not be used by
    scheduler, early stopping, preprocessing fit, feature selection fit or class
    weight computation.
    """
    labels = train_fold[target_col].astype(int).to_numpy()
    positions = np.arange(len(train_fold))
    train_idx, early_idx = train_test_split(
        positions,
        test_size=validation_fraction,
        stratify=labels,
        random_state=seed,
    )
    return train_fold.iloc[train_idx].copy(), train_fold.iloc[early_idx].copy()


def fit_training_partition_estimator(
    *,
    train_partition: pd.DataFrame,
    spec,
    resolved_config: dict[str, Any],
    seed: int,
    fold_index: int,
) -> FittedEstimatorResult:
    """Select an epoch internally and refit the exact estimator on all rows.

    This is the single estimator factory path used by inner search, outer
    evaluation and final full-development training.  A scoring frame is never
    accepted by this function, making accidental scheduler/early-stop leakage
    structurally impossible.
    """

    validate_resolved_config(resolved_config)
    factory = StudentEstimatorFactory(spec, resolved_config)
    scenario = str(resolved_config["feature_contract"]["scenario"])
    validate_scenario_features(resolved_config["feature_contract"]["sequence_columns"], scenario)
    set_seed(seed)
    model_train_partition, early_stop_partition = split_model_train_and_early_stop(
        train_partition,
        spec.target_col,
        seed=seed,
    )
    train_engineered = apply_feature_engineering(model_train_partition.copy(), spec.kind)
    early_stop_engineered = apply_feature_engineering(early_stop_partition.copy(), spec.kind)
    preprocessor = factory.create_preprocessor()
    train_prepared = preprocessor.fit_transform(train_engineered, apply_oversampling=True)
    early_stop_prepared = preprocessor.transform(early_stop_engineered)
    selector = factory.create_selector()
    train_selected = selector.fit_transform(
        train_prepared,
        preprocessor.numerical_cols,
        preprocessor.categorical_cols,
    )
    early_stop_selected = selector.transform(early_stop_prepared)
    train_dataset = StudentDataset(
        train_selected, spec.kind, spec.target_col,
        preprocessor.numerical_cols, preprocessor.categorical_cols,
    )
    early_stop_dataset = StudentDataset(
        early_stop_selected, spec.kind, spec.target_col,
        preprocessor.numerical_cols, preprocessor.categorical_cols,
    )
    batch_size = int(resolved_config["batch_size"])
    drop_last_train = bool(resolved_config["drop_last_train"])
    train_generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        generator=train_generator,
    )
    early_stop_loader = DataLoader(early_stop_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cat_cardinalities = [len(preprocessor.label_encoders[column].classes_) for column in train_dataset.cat_cols]
    model = factory.create_model(len(train_dataset.num_cols), cat_cardinalities, device)
    criterion = factory.create_criterion(
        model_train_partition[spec.target_col].astype(int).to_numpy(),
        device,
    )
    optimizer = factory.create_optimizer(model)
    scheduler_type = str(resolved_config["scheduler"]["type"])
    if scheduler_type == "fixed_lr":
        if bool(resolved_config["swa"]["enabled"]):
            raise ResolvedConfigError("SWA must be disabled for the fixed-LR Strategy B estimator.")
        model, early_history, _ = train_model_fixed_lr(
            model,
            train_loader,
            early_stop_loader,
            criterion,
            optimizer,
            build_training_config(resolved_config),
            device,
        )
    elif scheduler_type == "legacy_reduce_on_plateau":
        model, early_history, _ = train_model(
            model,
            train_loader,
            early_stop_loader,
            criterion,
            optimizer,
            build_training_config(resolved_config),
            device,
        )
    else:  # validation already rejects this, retained as defense in depth.
        raise ResolvedConfigError(f"Unsupported training policy: {scheduler_type}")
    selected_epochs = max(1, int(np.argmax(early_history["val_f1"]) + 1))

    # Full-partition refit.  The seed offset makes this stage deterministic and
    # independent of how many epochs were executed by the selection stage.
    set_seed(int(seed) + 100_003)
    refit_engineered = apply_feature_engineering(train_partition.copy(), spec.kind)
    refit_preprocessor = factory.create_preprocessor()
    refit_prepared = refit_preprocessor.fit_transform(refit_engineered, apply_oversampling=True)
    refit_selector = factory.create_selector()
    refit_selected = refit_selector.fit_transform(
        refit_prepared,
        refit_preprocessor.numerical_cols,
        refit_preprocessor.categorical_cols,
    )
    refit_dataset = StudentDataset(
        refit_selected, spec.kind, spec.target_col,
        refit_preprocessor.numerical_cols, refit_preprocessor.categorical_cols,
    )
    refit_generator = torch.Generator().manual_seed(int(seed) + 100_003)
    refit_loader = DataLoader(
        refit_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        generator=refit_generator,
    )
    refit_cardinalities = [
        len(refit_preprocessor.label_encoders[column].classes_)
        for column in refit_dataset.cat_cols
    ]
    refit_model = factory.create_model(len(refit_dataset.num_cols), refit_cardinalities, device)
    refit_criterion = factory.create_criterion(
        train_partition[spec.target_col].astype(int).to_numpy(),
        device,
    )
    refit_optimizer = factory.create_optimizer(refit_model)
    refit_model, refit_history = train_fixed_epochs(
        refit_model,
        refit_loader,
        refit_criterion,
        refit_optimizer,
        device,
        selected_epochs,
        scheduler_policy={"type": "fixed_lr", "parameters": {}, "replayable": True},
    )

    early_scheduler_state = early_history.get("scheduler_state") or {
        "type": "legacy_reduce_on_plateau",
        "replayable": False,
        "reductions": int(early_history.get("scheduler_reductions", 0)),
        "learning_rates": [float(value) for value in early_history.get("learning_rate", [])],
    }
    early_swa_state = early_history.get("swa_state") or {
        "enabled": bool(resolved_config["swa"]["enabled"]),
        "replayable": False,
        "batch_norm_statistics_updated": False,
    }
    estimator_parity = bool(
        early_scheduler_state.get("type") == "fixed_lr"
        and early_scheduler_state.get("replayable")
        and not early_swa_state.get("enabled")
        and refit_history["scheduler_state"]["type"] == "fixed_lr"
    )
    diagnostics = {
        "fold": int(fold_index),
        "seed": int(seed),
        "epochs_ran": int(early_history["epochs_ran"]),
        "selected_epoch": int(selected_epochs),
        "refit_epochs": int(refit_history["epochs"]),
        "max_epochs": int(resolved_config["max_epochs"]),
        "patience": int(resolved_config["patience"]),
        "hit_epoch_cap": bool(int(early_history["epochs_ran"]) >= int(resolved_config["max_epochs"])),
        "best_internal_validation_macro_f1": float(max(early_history["val_f1"])),
        "best_internal_validation_loss": float(min(early_history["val_loss"])),
        "final_internal_validation_loss": float(early_history["val_loss"][-1]),
        "final_train_loss": float(refit_history["train_loss"][-1]),
        "scheduler_state_selection": early_scheduler_state,
        "scheduler_state_refit": refit_history["scheduler_state"],
        "swa_state_selection": early_swa_state,
        "swa_state_refit": refit_history["swa_state"],
        "estimator_parity": estimator_parity,
        "factory_signature_selection": factory.estimator_signature(),
        "factory_signature_refit": factory.estimator_signature(),
        "criterion_parity": factory.criterion_signature() == factory.criterion_signature(),
        "resampling_parity": factory.resampling_signature() == factory.resampling_signature(),
        "sample_utilization_selection": loader_statistics(len(train_dataset), batch_size, drop_last_train),
        "sample_utilization_refit": loader_statistics(len(refit_dataset), batch_size, drop_last_train),
        "full_refit_input_records": int(len(train_partition)),
    }
    kernel = int(resolved_config["cnn_kernel_size"])
    cnn_output_length = (
        2
        if resolved_config["architecture_variant"] == "bilstm_only"
        else 2 + 2 * (kernel // 2) - kernel + 1
    )
    shapes = {
        "input_sequence_length": 2,
        "cnn_kernel_size": kernel,
        "cnn_output_sequence_length": int(cnn_output_length),
        "bilstm_input_sequence_length": int(cnn_output_length),
    }
    state = {key: value.detach().cpu().clone() for key, value in refit_model.state_dict().items()}
    return FittedEstimatorResult(
        model=refit_model,
        preprocessor=refit_preprocessor,
        selector=refit_selector,
        selected_features=list(refit_selector.selected_features),
        numerical_cols=list(refit_dataset.num_cols),
        categorical_cols=list(refit_dataset.cat_cols),
        train_row_positions=_row_positions(train_partition),
        early_stop_row_positions=_row_positions(early_stop_partition),
        refit_state_dict=state,
        training_diagnostics=diagnostics,
        shape_diagnostics=shapes,
        resolved_config=dict(resolved_config),
    )


def predict_with_fitted_estimator(
    *,
    frame: pd.DataFrame,
    spec,
    resolved_config: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    preprocessor: Any,
    selector: Any,
) -> np.ndarray:
    """Rebuild a saved estimator and return probabilities for ``frame``."""

    validate_resolved_config(resolved_config)
    factory = StudentEstimatorFactory(spec, resolved_config)
    engineered = apply_feature_engineering(frame.copy(), spec.kind)
    prepared = preprocessor.transform(engineered)
    selected = selector.transform(prepared)
    dataset = StudentDataset(
        selected, spec.kind, spec.target_col,
        preprocessor.numerical_cols, preprocessor.categorical_cols,
    )
    loader = DataLoader(dataset, batch_size=int(resolved_config["batch_size"]), shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cardinalities = [len(preprocessor.label_encoders[column].classes_) for column in dataset.cat_cols]
    model = factory.create_model(len(dataset.num_cols), cardinalities, device)
    model.load_state_dict(state_dict)
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            seq_x, num_x, cat_x, _, _ = batch[:5]
            output = model.predict_proba(seq_x.to(device), num_x.to(device), cat_x.to(device))
            probabilities.extend(output.detach().cpu().numpy())
    return np.asarray(probabilities, dtype=float)


def fit_final_development_estimator(
    *,
    development_frame: pd.DataFrame,
    spec,
    resolved_config: dict[str, Any],
    seed: int,
) -> FittedEstimatorResult:
    """Final estimator path: epoch selection then refit on every development row."""

    result = fit_training_partition_estimator(
        train_partition=development_frame,
        spec=spec,
        resolved_config=resolved_config,
        seed=seed,
        fold_index=-1,
    )
    if result.training_diagnostics["full_refit_input_records"] != len(development_frame):
        raise RuntimeError("Final estimator did not refit on the full development frame.")
    return result


def fit_fold_predict_proba(
    *,
    train_fold: pd.DataFrame,
    validation_fold: pd.DataFrame,
    spec,
    params: dict[str, Any],
    seed: int,
    fold_index: int,
    ablation_mode: str = "sequence_only",
    scenario: str = "late_stage",
) -> FoldModelResult:
    """Fit through the shared factory and score a never-trained-on fold."""

    del ablation_mode  # The approved Phase A-B estimator is sequence-only.
    validate_resolved_config(params)
    if scenario != params["feature_contract"]["scenario"]:
        raise ResolvedConfigError("Scenario argument disagrees with the resolved feature contract.")
    fitted = fit_training_partition_estimator(
        train_partition=train_fold,
        spec=spec,
        resolved_config=params,
        seed=seed,
        fold_index=fold_index,
    )
    probability_array = predict_with_fitted_estimator(
        frame=validation_fold,
        spec=spec,
        resolved_config=params,
        state_dict=fitted.refit_state_dict,
        preprocessor=fitted.preprocessor,
        selector=fitted.selector,
    )
    predictions = np.argmax(probability_array, axis=1)
    return FoldModelResult(
        fold_index=fold_index,
        seed=seed,
        row_positions=_row_positions(validation_fold),
        probabilities=probability_array,
        predictions=np.asarray(predictions, dtype=int),
        true_labels=validation_fold[spec.target_col].astype(int).to_numpy(),
        selected_features=fitted.selected_features,
        numerical_cols=fitted.numerical_cols,
        categorical_cols=fitted.categorical_cols,
        train_row_positions=fitted.train_row_positions,
        early_stop_row_positions=fitted.early_stop_row_positions,
        refit_state_dict=fitted.refit_state_dict,
        training_diagnostics=fitted.training_diagnostics,
        shape_diagnostics=fitted.shape_diagnostics,
        resolved_config=fitted.resolved_config,
        refit_preprocessor=fitted.preprocessor,
        refit_selector=fitted.selector,
    )


def objective_mean_cv_f1(
    trial: optuna.Trial,
    train_pool: pd.DataFrame,
    spec,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    architecture_variant: str = "cnn_bilstm",
    fair_comparison: bool = False,
    drop_last_train: bool = False,
) -> float:
    params = student_search_space(
        trial,
        architecture_variant=architecture_variant,
        fair_comparison=fair_comparison,
        drop_last_train=drop_last_train,
    )
    trial.set_user_attr("resolved_config", params)
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
    architecture_variant: str = "cnn_bilstm",
    fair_comparison: bool = False,
    drop_last_train: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[np.ndarray, np.ndarray]]]:
    folds = make_folds(train_pool, spec.target_col, n_splits=n_splits, seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    )
    study.optimize(
        lambda trial: objective_mean_cv_f1(
            trial,
            train_pool,
            spec,
            folds,
            architecture_variant=architecture_variant,
            fair_comparison=fair_comparison,
            drop_last_train=drop_last_train,
        ),
        n_trials=n_trials,
    )
    trial_history = [
        {
            "number": trial.number,
            "state": str(trial.state),
            "value": None if trial.value is None else float(trial.value),
            "params": dict(trial.params),
            "resolved_config": trial.user_attrs.get("resolved_config"),
            "fold_f1_macro": trial.user_attrs.get("fold_f1_macro"),
        }
        for trial in study.trials
    ]
    best_params = study.best_trial.user_attrs.get("resolved_config")
    if best_params is None:
        best_params = resolve_student_config(
            study.best_params,
            architecture_variant=architecture_variant,
            suggested_parameters=dict(study.best_params),
            scheduler_type="fixed_lr",
            swa_enabled=False,
            drop_last_train=drop_last_train,
            evidence_role="optuna_candidate_corrected_estimator",
        )
    validate_resolved_config(best_params)
    return {
        "best_cv_f1_macro": float(study.best_value),
        "best_params": best_params,
        "resolved_config": best_params,
    }, trial_history, folds


def collect_oof_by_seed(
    train_pool: pd.DataFrame,
    spec,
    params: dict[str, Any],
    folds: list[tuple[np.ndarray, np.ndarray]],
    seeds: list[int],
    *,
    ablation_mode: str = "sequence_only",
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
    if source not in {"inner_oof", "oof_validation"}:
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


def fit_temperature_policy(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    *,
    source: str,
    grid: list[float] | None = None,
) -> dict[str, Any]:
    if source not in {"inner_oof", "oof_validation"}:
        raise ValueError("Temperature calibration must use inner OOF probabilities only.")
    grid = grid or [round(value, 2) for value in np.arange(0.7, 2.55, 0.15)]
    best_temperature = 1.0
    best_nll = math.inf
    clipped = np.clip(probabilities, 1e-8, 1.0)
    for temperature in grid:
        calibrated = apply_probability_calibration(clipped, {"type": "temperature", "temperature": temperature})
        row_probabilities = calibrated[np.arange(len(y_true)), y_true.astype(int)]
        nll = -float(np.mean(np.log(np.clip(row_probabilities, 1e-8, 1.0))))
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)
    return {
        "type": "temperature",
        "source": source,
        "temperature": best_temperature,
        "optimized_metric": "NLL",
        "nll": best_nll,
    }


def apply_probability_calibration(probabilities: np.ndarray, calibration_policy: dict[str, Any] | None) -> np.ndarray:
    if not calibration_policy or calibration_policy.get("type") in {None, "none"}:
        return probabilities
    if calibration_policy.get("type") != "temperature":
        raise ValueError(f"Unsupported calibration policy: {calibration_policy}")
    temperature = max(float(calibration_policy["temperature"]), 1e-6)
    calibrated = np.power(np.clip(probabilities, 1e-8, 1.0), 1.0 / temperature)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


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


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.eye(probabilities.shape[1])[y_true.astype(int)]
    return float(np.mean(np.sum((probabilities - labels) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    correctness = (predictions == y_true.astype(int)).astype(float)
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        mask = (confidences > lower) & (confidences <= upper)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask) * abs(np.mean(correctness[mask]) - np.mean(confidences[mask])))
    return ece


def metric_summary(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold_policy: dict[str, Any] | None,
    fold_ids: np.ndarray,
) -> dict[str, Any]:
    predictions = apply_threshold_policy(probabilities, threshold_policy)
    return metric_summary_from_predictions(y_true, probabilities, predictions, fold_ids)


def metric_summary_from_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    fold_ids: np.ndarray,
) -> dict[str, Any]:
    fold_scores = [
        float(f1_score(y_true[fold_ids == fold], predictions[fold_ids == fold], average="macro", zero_division=0))
        for fold in sorted(set(fold_ids.tolist()))
    ]
    precision, recall, f1, support = precision_recall_fscore_support(y_true, predictions, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "brier_score": multiclass_brier_score(y_true, probabilities),
        "ece": expected_calibration_error(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1, 2]).tolist(),
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


def evaluate_ensemble_strategies(
    oof: dict[str, Any],
    seeds: list[int],
    *,
    single_seed_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y_true = oof["y_true"]
    fold_ids = oof["fold_ids"]
    seed_probabilities = oof["seed_probabilities"]
    seed_scores = {
        seed: f1_score(y_true, np.argmax(seed_probabilities[seed], axis=1), average="macro", zero_division=0)
        for seed in seeds
    }
    if single_seed_only:
        if len(seeds) != 1:
            raise ValueError("single_seed_only requires exactly one predeclared seed.")
        seed = int(seeds[0])
        probabilities = seed_probabilities[seed]
        summary = metric_summary(y_true, probabilities, None, fold_ids)
        selected = {
            "strategy_name": f"single_seed_{seed}_argmax",
            "ensemble_method": "mean_probability",
            "seed_list": [seed],
            "seed_weights": None,
            "seed_count": 1,
            "calibration_policy": {"type": "none"},
            "threshold_policy": {"type": "argmax"},
            **summary,
        }
        return [selected], selected
    ranked_seeds = [seed for seed, _ in sorted(seed_scores.items(), key=lambda item: item[1], reverse=True)]
    rows = []

    def append_strategy(
        name: str,
        method: str,
        seed_list: list[int],
        threshold_policy: dict[str, Any] | None,
        weights: dict[int, float] | None = None,
        calibration_policy: dict[str, Any] | None = None,
    ):
        probabilities = combine_seed_probabilities(seed_probabilities, method=method, seed_list=seed_list, weights=weights)
        probabilities = apply_probability_calibration(probabilities, calibration_policy)
        summary = metric_summary(y_true, probabilities, threshold_policy, fold_ids)
        rows.append(
            {
                "strategy_name": name,
                "ensemble_method": method,
                "seed_list": seed_list,
                "seed_weights": weights,
                "seed_count": len(seed_list),
                "calibration_policy": calibration_policy or {"type": "none"},
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
    threshold_policy = optimize_class_thresholds(base_probabilities, y_true, source="inner_oof")
    append_strategy("all_seed_mean_probability_oof_thresholds", "mean_probability", seeds, threshold_policy)
    calibration_policy = fit_temperature_policy(base_probabilities, y_true, source="inner_oof")
    append_strategy("all_seed_mean_temperature_argmax", "mean_probability", seeds, None, calibration_policy=calibration_policy)

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
