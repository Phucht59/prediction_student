from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.config import DATASETS
from src.experiments.common import (
    ExperimentConfig,
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
from src.utils import setup_logger

logger = setup_logger("baseline_experiments")


@dataclass(frozen=True)
class BaselineRunConfig:
    max_iter: int = 300
    n_estimators: int = 200


def _split_xy(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    drop_cols = [target_col]
    if "G3_raw" in df.columns:
        drop_cols.append("G3_raw")
    x = df.drop(columns=drop_cols, errors="ignore")
    return x, df[target_col].astype(int).to_numpy()


def _gradient_boosting_available(seed: int, run_config: BaselineRunConfig):
    try:
        from xgboost import XGBClassifier

        return (
            "xgboost",
            XGBClassifier(
                n_estimators=run_config.n_estimators,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=seed,
                n_jobs=1,
            ),
        )
    except Exception:
        try:
            from catboost import CatBoostClassifier

            return (
                "catboost",
                CatBoostClassifier(
                    iterations=run_config.n_estimators,
                    depth=4,
                    learning_rate=0.05,
                    loss_function="MultiClass",
                    random_seed=seed,
                    verbose=False,
                ),
            )
        except Exception:
            return (
                "hist_gradient_boosting",
                HistGradientBoostingClassifier(
                    max_iter=run_config.max_iter,
                    learning_rate=0.05,
                    random_state=seed,
                ),
            )


def build_baseline_models(seed: int, strategy: str, run_config: BaselineRunConfig) -> dict[str, Any]:
    class_weight = "balanced" if strategy == "class_weight" else None
    gb_name, gb_model = _gradient_boosting_available(seed, run_config)
    return {
        "logistic_regression": LogisticRegression(
            max_iter=run_config.max_iter,
            class_weight=class_weight,
            solver="lbfgs",
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=run_config.n_estimators,
            class_weight="balanced_subsample" if strategy == "class_weight" else None,
            random_state=seed,
            n_jobs=1,
        ),
        gb_name: gb_model,
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=run_config.max_iter,
            early_stopping=True,
            random_state=seed,
        ),
    }


def _fit_model(model: Any, x_train: pd.DataFrame, y_train: np.ndarray, strategy: str) -> Any:
    if strategy == "class_weight" and model.__class__.__name__ in {
        "XGBClassifier",
        "CatBoostClassifier",
        "HistGradientBoostingClassifier",
        "MLPClassifier",
    }:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        try:
            model.fit(x_train, y_train, sample_weight=sample_weight)
            return model
        except TypeError:
            logger.info("%s does not accept sample_weight in this environment; fitting unweighted.", model.__class__.__name__)
    model.fit(x_train, y_train)
    return model


def _predict_proba_or_none(model: Any, x: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(x), dtype=float)
        except Exception:
            return None
    return None


def run_baseline_suite(
    dataset_name: str,
    scenario: str,
    strategies: list[str],
    config: ExperimentConfig,
    run_config: BaselineRunConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_config = run_config or BaselineRunConfig()
    spec = DATASETS[dataset_name]
    train_pool, locked_test = load_or_create_student_splits(dataset_name, config.target_mode)
    labels = train_pool[spec.target_col].astype(int).to_numpy()
    cv_rows: list[dict] = []
    locked_rows: list[dict] = []

    for strategy in strategies:
        logger.info("Running baselines: dataset=%s scenario=%s strategy=%s", dataset_name, scenario, strategy)
        oof_by_model: dict[str, list[dict]] = {}
        for fold, (train_idx, val_idx) in enumerate(stratified_folds(labels, config).split(train_pool, labels), start=1):
            prepared = prepare_fold(
                train_pool.iloc[train_idx].copy(),
                train_pool.iloc[val_idx].copy(),
                spec.target_col,
                scenario,
                strategy,
                config,
            )
            x_train, y_train = _split_xy(prepared.train, spec.target_col)
            x_val, y_val = _split_xy(prepared.validation, spec.target_col)
            models = build_baseline_models(config.seed + fold, strategy, run_config)
            for model_name, model in models.items():
                fitted = _fit_model(model, x_train, y_train, strategy)
                preds = fitted.predict(x_val)
                probs = _predict_proba_or_none(fitted, x_val)
                metrics = compute_required_metrics(y_val, preds, probs)
                row = {
                    "dataset": dataset_name,
                    "scenario": scenario,
                    "strategy": strategy,
                    "model": model_name,
                    "fold": fold,
                    **metrics,
                }
                cv_rows.append(row)
                oof_by_model.setdefault(model_name, []).append(row)

        full_prepared = prepare_fold(train_pool.copy(), locked_test.copy(), spec.target_col, scenario, strategy, config)
        x_full, y_full = _split_xy(full_prepared.train, spec.target_col)
        locked_selected = transform_with_prepared(locked_test.copy(), full_prepared, spec.target_col, scenario)
        x_locked, y_locked = _split_xy(locked_selected, spec.target_col)
        locked_reg_true = locked_selected["G3_raw"].to_numpy(dtype=float) if "G3_raw" in locked_selected else None
        for model_name, model in build_baseline_models(config.seed, strategy, run_config).items():
            fitted = _fit_model(model, x_full, y_full, strategy)
            preds = fitted.predict(x_locked)
            probs = _predict_proba_or_none(fitted, x_locked)
            metrics = compute_required_metrics(y_locked, preds, probs, y_reg_true=locked_reg_true)
            locked_rows.append(
                {
                    "dataset": dataset_name,
                    "scenario": scenario,
                    "strategy": strategy,
                    "model": model_name,
                    "fold": "locked_test",
                    **metrics,
                }
            )

    cv_df = pd.DataFrame(cv_rows)
    locked_df = pd.DataFrame(locked_rows)
    dirs = ensure_technical_report_dirs()
    prefix = f"{dataset_name}_{scenario}"
    cv_path = dirs["baselines"] / f"{prefix}_cv.csv"
    locked_path = dirs["baselines"] / f"{prefix}_locked_test.csv"
    summary_path = dirs["baselines"] / f"{prefix}_summary.json"
    cv_df.to_csv(cv_path, index=False)
    locked_df.to_csv(locked_path, index=False)

    summary_rows = []
    for keys, group in cv_df.groupby(["dataset", "scenario", "strategy", "model"]):
        summary_rows.append({**dict(zip(["dataset", "scenario", "strategy", "model"], keys)), **summarize_cv(group.to_dict("records"))})
    summary = {
        "cv_summary": summary_rows,
        "locked_test": locked_rows,
        "notes": "Locked test rows are final evaluations only; model/strategy selection must use CV summaries.",
    }
    save_json(summary_path, summary)
    write_config(dirs["baselines"] / f"{prefix}_config.json", config, {"baseline": run_config.__dict__, "strategies": strategies})
    return cv_df, locked_df
