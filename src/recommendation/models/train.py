"""Deterministic per-action CV, EBM search, baselines, and OOF diagnostics."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ..evaluation.metrics import mae, rmse, spearman
from .baselines import fit_random_forest, fit_ridge, stage_column, stage_prior_predict
from .datasets import feature_matrix
from .ebm import fit_ebm, predict_raw
from .features import ACTION_TO_KEY, APPROVED_FEATURES


def _simpler(left: dict, right: dict) -> bool:
    left_key = (int(left["interactions"]), int(left["max_bins"]), -int(left["min_samples_leaf"]))
    right_key = (int(right["interactions"]), int(right["max_bins"]), -int(right["min_samples_leaf"]))
    return left_key < right_key


def make_folds(n_rows: int, *, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train.astype(int), val.astype(int)) for train, val in splitter.split(np.arange(n_rows))]


def choose_n_splits(n_rows: int, config: dict, action_id: str) -> tuple[int, str]:
    default = int(config["cv"]["n_splits"])
    if action_id == "assessment_recovery" and n_rows < int(config["cv"]["a1_small_n_threshold"]):
        return int(config["cv"]["a1_small_n_splits"]), "A1 n is below the 5-fold stability threshold"
    if n_rows < default * 8:
        return min(default, 3), "sample size is too small for stable 5-fold partitions"
    return default, "deterministic shuffled K-fold; Panel A identities are unique"


def config_grid(phase8: dict) -> list[dict]:
    search = phase8["ebm"]["search"]
    fixed = dict(phase8["ebm"]["fixed"])
    grid = []
    for max_bins, interactions, min_samples_leaf in product(search["max_bins"], search["interactions"], search["min_samples_leaf"]):
        item = dict(fixed)
        item.update({"max_bins": int(max_bins), "interactions": int(interactions), "min_samples_leaf": int(min_samples_leaf)})
        grid.append(item)
    return grid


def _metrics(y_true, y_pred, weights) -> dict:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "weighted_mae": mae(y_true, y_pred, weights),
        "weighted_rmse": rmse(y_true, y_pred, weights),
        "spearman": spearman(y_true, y_pred),
        "pred_min": float(np.min(y_pred)),
        "pred_max": float(np.max(y_pred)),
        "residual_mean": float(np.mean(y_true - y_pred)),
        "residual_std": float(np.std(y_true - y_pred)),
    }


def oof_fit_predict(frame: pd.DataFrame, folds, *, fit_fn, predict_fn) -> tuple[np.ndarray, list[int]]:
    X, y, w = feature_matrix(frame)
    preds = np.full(len(frame), np.nan, dtype=float)
    fold_ids = np.full(len(frame), -1, dtype=int)
    for fold_id, (train_idx, val_idx) in enumerate(folds):
        model = fit_fn(X[train_idx], y[train_idx], w[train_idx])
        preds[val_idx] = predict_fn(model, X[val_idx])
        fold_ids[val_idx] = fold_id
    if np.isnan(preds).any():
        raise ValueError("OOF predictions are incomplete")
    return preds, fold_ids.tolist()


def search_ebm(frame: pd.DataFrame, phase8: dict, action_id: str) -> dict:
    X, y, w = feature_matrix(frame)
    n_splits, cv_reason = choose_n_splits(len(frame), phase8, action_id)
    folds = make_folds(len(frame), n_splits=n_splits, seed=int(phase8["cv"]["seed"]))
    scored = []
    for config in config_grid(phase8):
        preds, fold_ids = oof_fit_predict(
            frame,
            folds,
            fit_fn=lambda Xt, yt, wt, cfg=config: fit_ebm(Xt, yt, sample_weight=wt, config=cfg),
            predict_fn=lambda model, Xv: predict_raw(model, Xv),
        )
        metrics = _metrics(y, preds, w)
        scored.append({"config": config, "metrics": metrics, "oof": preds, "fold_ids": fold_ids})
    scored.sort(key=lambda item: (item["metrics"]["mae"], item["metrics"]["rmse"], item["config"]["interactions"], item["config"]["max_bins"], -item["config"]["min_samples_leaf"]))
    selected = scored[0]
    if action_id == "assessment_recovery" and phase8["ebm"].get("a1_prefer_no_interactions"):
        no_int = [item for item in scored if item["config"]["interactions"] == 0]
        if no_int:
            gain = (no_int[0]["metrics"]["mae"] - selected["metrics"]["mae"]) / max(no_int[0]["metrics"]["mae"], 1e-9)
            if selected["config"]["interactions"] > 0 and gain < float(phase8["ebm"]["interaction_keep_relative_mae_gain"]):
                selected = no_int[0]
    unweighted_preds, _ = oof_fit_predict(
        frame,
        folds,
        fit_fn=lambda Xt, yt, wt, cfg=selected["config"]: fit_ebm(Xt, yt, sample_weight=None, config=cfg),
        predict_fn=lambda model, Xv: predict_raw(model, Xv),
    )
    prior_preds, _ = oof_fit_predict(
        frame,
        folds,
        fit_fn=lambda Xt, yt, wt: ("prior", Xt, yt, wt),
        predict_fn=lambda model, Xv: stage_prior_predict(stage_column(model[1]), model[2], model[3], stage_column(Xv)),
    )
    ridge_preds, _ = oof_fit_predict(
        frame,
        folds,
        fit_fn=lambda Xt, yt, wt: fit_ridge(Xt, yt, wt, alpha=float(phase8["ridge"]["alpha"])),
        predict_fn=lambda model, Xv: model.predict(Xv),
    )
    rf_preds, _ = oof_fit_predict(
        frame,
        folds,
        fit_fn=lambda Xt, yt, wt: fit_random_forest(Xt, yt, wt, config=phase8["random_forest"]),
        predict_fn=lambda model, Xv: model.predict(Xv),
    )
    oof = pd.DataFrame({
        "case_id": frame["case_id"].astype(str),
        "action_id": action_id,
        "y_expected": y,
        "y_pred_oof": selected["oof"],
        "absolute_error": np.abs(y - selected["oof"]),
        "sample_weight": w,
        "fold_id": selected["fold_ids"],
        "y_pred_unweighted": unweighted_preds,
        "y_pred_stage_prior": prior_preds,
        "y_pred_ridge": ridge_preds,
        "y_pred_random_forest": rf_preds,
    })
    return {
        "action_id": action_id,
        "action_key": ACTION_TO_KEY[action_id],
        "n_rows": int(len(frame)),
        "n_splits": n_splits,
        "cv_reason": cv_reason,
        "selected_config": selected["config"],
        "cv_metrics": selected["metrics"],
        "unweighted_metrics": _metrics(y, unweighted_preds, w),
        "baseline_metrics": {
            "ACTION_STAGE_PRIOR": _metrics(y, prior_preds, w),
            "RIDGE": _metrics(y, ridge_preds, w),
            "RANDOM_FOREST": _metrics(y, rf_preds, w),
        },
        "oof": oof,
        "features": list(APPROVED_FEATURES),
    }
