from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import DATASETS
from src.experiments.common import (
    ExperimentConfig,
    IMBALANCE_STRATEGIES,
    PreparedFold,
    TechnicalStudentDataset,
    compute_required_metrics,
    ensure_technical_report_dirs,
    load_or_create_student_splits,
    predict_with_low_threshold,
    prepare_fold,
    save_json,
    stratified_folds,
    summarize_cv,
    transform_with_prepared,
    tune_low_threshold_from_oof,
    write_config,
)
from src.losses_v27 import ClassBalancedFocalLoss, JointHybridLoss, OrdinalLoss
from src.train_pipeline import calculate_class_weights
from src.train_v27_pipeline import create_model_v27, train_model_v27
from src.utils import set_seed, setup_logger

logger = setup_logger("imbalance_experiments")


@dataclass(frozen=True)
class DeepRunConfig:
    max_epochs: int = 15
    batch_size: int = 32
    patience: int = 5
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    class_balanced_beta: float = 0.99
    focal_gamma: float = 2.0
    ordinal_weight: float = 0.5
    regression_weight: float = 0.05
    ensemble_seeds: tuple[int, ...] = (42, 123, 155)


class _TrainConfig:
    def __init__(self, run_config: DeepRunConfig):
        self.max_epochs = run_config.max_epochs
        self.patience = run_config.patience
        self.scheduler_patience = run_config.scheduler_patience
        self.scheduler_factor = run_config.scheduler_factor


def _dataset_from_frame(frame: pd.DataFrame, prepared: PreparedFold, target_col: str) -> TechnicalStudentDataset:
    return TechnicalStudentDataset(
        frame,
        target_col=target_col,
        numerical_cols=prepared.preprocessor.numerical_cols,
        categorical_cols=prepared.preprocessor.categorical_cols,
        sequence_cols=prepared.sequence_cols,
    )


def _make_loss(strategy: str, train_labels: np.ndarray, run_config: DeepRunConfig, device: torch.device) -> JointHybridLoss:
    if strategy in {"focal_loss", "smotenc_focal_loss"}:
        counts = np.bincount(train_labels.astype(int), minlength=3)
        class_loss = ClassBalancedFocalLoss(
            class_counts=counts,
            beta=run_config.class_balanced_beta,
            gamma=run_config.focal_gamma,
        ).to(device)
    elif strategy == "class_weight":
        weights = calculate_class_weights(train_labels, num_classes=3).to(device)
        class_loss = nn.CrossEntropyLoss(weight=weights)
    else:
        class_loss = nn.CrossEntropyLoss()
    return JointHybridLoss(
        class_loss_fn=class_loss,
        ordinal_loss_fn=OrdinalLoss(),
        regression_loss_fn=nn.MSELoss(),
        w_class=1.0,
        w_ord=run_config.ordinal_weight,
        w_reg=run_config.regression_weight,
    )


def _train_single_model(
    prepared: PreparedFold,
    target_col: str,
    strategy: str,
    run_config: DeepRunConfig,
    seed: int,
    device: torch.device,
    loss_labels: np.ndarray | None = None,
):
    set_seed(seed)
    train_ds = _dataset_from_frame(prepared.train, prepared, target_col)
    val_ds = _dataset_from_frame(prepared.validation, prepared, target_col)
    train_loader = DataLoader(train_ds, batch_size=run_config.batch_size, shuffle=True, drop_last=len(train_ds) > run_config.batch_size)
    val_loader = DataLoader(val_ds, batch_size=run_config.batch_size, shuffle=False)
    cat_cardinalities = [
        len(prepared.preprocessor.label_encoders[column].classes_)
        for column in train_ds.cat_cols
    ]
    model = create_model_v27(
        "student",
        {
            "learning_rate": run_config.learning_rate,
            "weight_decay": run_config.weight_decay,
            "batch_size": run_config.batch_size,
            "cnn_channels": 32,
            "cnn_kernel_size": 3,
            "lstm_hidden_dim": 64,
            "context_hidden_dim": 64,
            "fusion_hidden_dim": 64,
            "dropout": 0.3,
        },
        num_numerical=len(train_ds.num_cols),
        cat_cardinalities=cat_cardinalities,
    ).to(device)
    original_labels = loss_labels if loss_labels is not None else prepared.train[target_col].astype(int).to_numpy()
    criterion = _make_loss(strategy, original_labels, run_config, device)
    optimizer = optim.Adam(model.parameters(), lr=run_config.learning_rate, weight_decay=run_config.weight_decay)
    trained, history, best_score = train_model_v27(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        _TrainConfig(run_config),
        device,
    )
    return trained, val_loader, history, best_score


def _predict_model(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    reg_true: list[np.ndarray] = []
    reg_pred: list[np.ndarray] = []
    with torch.no_grad():
        for seq_x, num_x, cat_x, y, _, y_reg in loader:
            outputs = model(seq_x.to(device), num_x.to(device), cat_x.to(device))
            batch_probs = torch.softmax(outputs[0], dim=1).cpu().numpy()
            probs.append(batch_probs)
            labels.append(y.numpy())
            reg_true.append(y_reg.numpy())
            reg_pred.append(outputs[2].cpu().numpy())
    return np.vstack(probs), np.concatenate(labels), np.concatenate(reg_true), np.concatenate(reg_pred)


def run_imbalance_suite(
    dataset_name: str,
    scenario: str,
    strategies: list[str],
    config: ExperimentConfig,
    run_config: DeepRunConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_config = run_config or DeepRunConfig()
    invalid = sorted(set(strategies) - set(IMBALANCE_STRATEGIES))
    if invalid:
        raise ValueError(f"Unsupported imbalance strategies: {invalid}")
    spec = DATASETS[dataset_name]
    train_pool, locked_test = load_or_create_student_splits(dataset_name, config.target_mode)
    labels = train_pool[spec.target_col].astype(int).to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cv_rows: list[dict] = []
    locked_rows: list[dict] = []
    threshold_rows: list[dict] = []

    for strategy in strategies:
        logger.info("Running CNN-BiLSTM V27: dataset=%s scenario=%s strategy=%s", dataset_name, scenario, strategy)
        oof_probs = np.zeros((len(train_pool), 3), dtype=float)
        oof_targets = np.zeros(len(train_pool), dtype=int)
        for fold, (train_idx, val_idx) in enumerate(stratified_folds(labels, config).split(train_pool, labels), start=1):
            prepared = prepare_fold(
                train_pool.iloc[train_idx].copy(),
                train_pool.iloc[val_idx].copy(),
                spec.target_col,
                scenario,
                strategy,
                config,
            )
            seed = run_config.ensemble_seeds[0] + fold
            loss_labels = train_pool.iloc[train_idx][spec.target_col].astype(int).to_numpy()
            model, val_loader, _, best_score = _train_single_model(
                prepared,
                spec.target_col,
                strategy,
                run_config,
                seed,
                device,
                loss_labels=loss_labels,
            )
            probs, y_val, reg_true, reg_pred = _predict_model(model, val_loader, device)
            preds = np.argmax(probs, axis=1)
            metrics = compute_required_metrics(y_val, preds, probs, y_reg_true=reg_true, y_reg_pred=reg_pred)
            row = {
                "dataset": dataset_name,
                "scenario": scenario,
                "strategy": strategy,
                "model": "cnn_bilstm_v27",
                "fold": fold,
                "best_val_score": float(best_score),
                **metrics,
            }
            cv_rows.append(row)
            oof_probs[val_idx] = probs
            oof_targets[val_idx] = y_val

        threshold = tune_low_threshold_from_oof(oof_targets, oof_probs)
        threshold_rows.append(
            {
                "dataset": dataset_name,
                "scenario": scenario,
                "strategy": strategy,
                **threshold,
            }
        )

        full_prepared = prepare_fold(train_pool.copy(), locked_test.copy(), spec.target_col, scenario, strategy, config)
        locked_selected = transform_with_prepared(locked_test.copy(), full_prepared, spec.target_col, scenario)
        locked_ds = _dataset_from_frame(locked_selected, full_prepared, spec.target_col)
        locked_loader = DataLoader(locked_ds, batch_size=run_config.batch_size, shuffle=False)
        ensemble_probs = []
        ensemble_reg = []
        y_locked = None
        reg_true_locked = None
        for member_index, seed in enumerate(run_config.ensemble_seeds, start=1):
            logger.info("Training locked-test ensemble member %s seed=%s", member_index, seed)
            model, _, _, _ = _train_single_model(
                full_prepared,
                spec.target_col,
                strategy,
                run_config,
                seed,
                device,
                loss_labels=train_pool[spec.target_col].astype(int).to_numpy(),
            )
            probs, y_eval, reg_true, reg_pred = _predict_model(model, locked_loader, device)
            ensemble_probs.append(probs)
            ensemble_reg.append(reg_pred)
            y_locked = y_eval
            reg_true_locked = reg_true
        mean_probs = np.mean(np.stack(ensemble_probs, axis=0), axis=0)
        mean_reg = np.mean(np.stack(ensemble_reg, axis=0), axis=0)
        raw_preds = np.argmax(mean_probs, axis=1)
        tuned_preds = predict_with_low_threshold(mean_probs, float(threshold["threshold_low"]))
        for prediction_mode, preds in (("argmax", raw_preds), ("low_threshold_tuned", tuned_preds)):
            metrics = compute_required_metrics(y_locked, preds, mean_probs, y_reg_true=reg_true_locked, y_reg_pred=mean_reg)
            locked_rows.append(
                {
                    "dataset": dataset_name,
                    "scenario": scenario,
                    "strategy": strategy,
                    "model": "cnn_bilstm_v27",
                    "prediction_mode": prediction_mode,
                    "fold": "locked_test",
                    "threshold_low": threshold["threshold_low"] if prediction_mode == "low_threshold_tuned" else 0.0,
                    "ensemble_seeds": "|".join(str(seed) for seed in run_config.ensemble_seeds),
                    **metrics,
                }
            )

    cv_df = pd.DataFrame(cv_rows)
    locked_df = pd.DataFrame(locked_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    dirs = ensure_technical_report_dirs()
    prefix = f"{dataset_name}_{scenario}"
    cv_df.to_csv(dirs["imbalance"] / f"{prefix}_cnn_bilstm_cv.csv", index=False)
    locked_df.to_csv(dirs["imbalance"] / f"{prefix}_cnn_bilstm_locked_test.csv", index=False)
    threshold_df.to_csv(dirs["imbalance"] / f"{prefix}_thresholds.csv", index=False)
    cv_df.to_csv(dirs["ablation"] / f"{prefix}_cnn_bilstm_strategy_ablation_cv.csv", index=False)
    locked_df.to_csv(dirs["ablation"] / f"{prefix}_cnn_bilstm_strategy_ablation_locked_test.csv", index=False)

    summary_rows = []
    for keys, group in cv_df.groupby(["dataset", "scenario", "strategy", "model"]):
        summary_rows.append({**dict(zip(["dataset", "scenario", "strategy", "model"], keys)), **summarize_cv(group.to_dict("records"))})
    save_json(
        dirs["imbalance"] / f"{prefix}_summary.json",
        {
            "cv_summary": summary_rows,
            "locked_test": locked_rows,
            "thresholds_from_oof": threshold_rows,
            "notes": "Thresholds are tuned on out-of-fold train-pool predictions, never on locked test.",
        },
    )
    save_json(
        dirs["ablation"] / f"{prefix}_cnn_bilstm_strategy_ablation_summary.json",
        {
            "cv_summary": summary_rows,
            "locked_test": locked_rows,
            "thresholds_from_oof": threshold_rows,
            "ablation_axis": "imbalance strategy, class-balanced focal loss, SMOTENC, random oversampling, and Low-threshold tuning",
        },
    )
    write_config(
        dirs["imbalance"] / f"{prefix}_config.json",
        config,
        {"deep": run_config.__dict__, "strategies": strategies},
    )
    return cv_df, locked_df
