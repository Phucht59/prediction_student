"""Frozen protocol, search, metrics, and selection helpers for Phase C."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from src.config import DATASETS
from src.estimator_factory import resolve_phase_c_neural_config, resolved_config_hash, validate_resolved_config
from src.model_selection import expected_calibration_error, multiclass_brier_score
from src.models import count_trainable_parameters, create_phase_c_model


PHASE_C_PROTOCOL_VERSION = "strategy_b_phase_c_v1"
PHASE_C_SEEDS = [42, 123, 155]
MAIN_NEURAL = ["N0", "N1", "N2", "N3"]
ABLATIONS = ["A1", "A2"]
ML_CANDIDATES = ["M1", "M2"]
RANKING_CANDIDATES = ["R0", "M1", "M2", "N0", "N1", "N2", "N3"]
ALL_CANDIDATES = RANKING_CANDIDATES + ABLATIONS
PRACTICAL_MARGIN = 0.01
PARAMETER_GUARDRAIL = 5000


def candidate_registry() -> dict[str, Any]:
    rows = [
        ("R0", "G2 deterministic threshold rule", "rule", True, False),
        ("M1", "Random Forest", "machine_learning", True, False),
        ("M2", "SVM RBF", "machine_learning", True, False),
        ("N0", "Corrected compact nominal CNN-BiLSTM", "cnn_bilstm", True, True),
        ("N1", "Corrected compact ordinal CNN-BiLSTM", "cnn_bilstm", True, True),
        ("N2", "Tiny nominal MLP", "neural_mlp", True, False),
        ("N3", "Tiny ordered MLP", "neural_mlp", True, False),
        ("A1", "Parameter-matched CNN-only", "ablation", False, False),
        ("A2", "Parameter-matched BiLSTM-only", "ablation", False, False),
    ]
    return {
        "registry_version": "strategy_b_phase_c_candidates_v1",
        "candidates": [
            {
                "id": candidate_id,
                "name": name,
                "family": family,
                "eligible_overall": eligible_overall,
                "eligible_thesis_hybrid": eligible_hybrid,
            }
            for candidate_id, name, family, eligible_overall, eligible_hybrid in rows
        ],
        "optional_sanity_models": {
            "S0": "not_activated_to_avoid_unnecessary_multiplicity",
            "S1": "not_activated_to_avoid_unnecessary_multiplicity",
        },
        "prohibited_phase_c": ["cnn_lstm", "gru", "transformer", "context_fusion", "hybrid_ensemble"],
    }


def search_spaces() -> dict[str, Any]:
    common = {
        "normalization": ["none", "layer_norm"],
        "dropout": [0.0, 0.30],
        "batch_size": [16, 32],
        "max_epochs": [40, 60, 80],
        "learning_rate": {"low": 1e-4, "high": 1e-2, "log": True},
        "weight_decay": {"low": 1e-6, "high": 1e-3, "log": True},
        "fixed": {
            "features": ["G1", "G2"], "class_weight_mode": "none", "oversample_method": "none",
            "loss": "unweighted_cross_entropy_or_ordered_bce", "drop_last": False,
            "scheduler": "fixed_lr", "swa": False, "batch_norm": False,
        },
    }
    return {
        "frozen_before_full_outer_aggregation": True,
        "trial_budget_per_outer_fold_per_searched_family": 30,
        "inner_folds": 3,
        "N0_N1": {
            **deepcopy(common),
            "cnn_channels": [4, 8, 16], "cnn_kernel_size": [1, 2],
            "lstm_hidden_dim": [4, 8, 16], "sequence_dropout": [0.0, 0.20],
            "parameter_guardrail": PARAMETER_GUARDRAIL,
        },
        "N2_N3": {
            **deepcopy(common),
            "hidden_dim": [4, 8, 16, 32], "num_layers": [1, 2],
            "parameter_guardrail": PARAMETER_GUARDRAIL,
        },
        "M1": {
            "n_estimators": [100, 200, 300], "max_depth": [None, 3, 5, 8],
            "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2", None],
            "class_weight": [None], "random_state": [42],
        },
        "M2": {
            "C": {"low": 0.05, "high": 50.0, "log": True},
            "gamma": {"low": 1e-3, "high": 2.0, "log": True},
            "kernel": ["rbf"], "class_weight": [None], "probability": [True], "random_state": [42],
        },
        "A1_A2": "fixed per outer fold from the selected N0 dimensions; no independent search",
    }


def selection_rule() -> dict[str, Any]:
    return {
        "frozen_before_full_outer_aggregation": True,
        "primary_metric": "mean declared-seed outer OOF macro_f1 (single OOF for deterministic/ML)",
        "weighted_f1_primary": False,
        "composite_score": None,
        "practical_margin_absolute_macro_f1": PRACTICAL_MARGIN,
        "uncertainty": "paired record bootstrap is descriptive only",
        "no_clear_superiority_if": ["absolute_delta_below_0.01", "paired_bootstrap_interval_includes_zero"],
        "tie_break_order": [
            "fewer_class_collapses", "lower_seed_sd", "higher_worst_seed", "lower_two_step_error",
            "lower_ece", "fewer_parameters", "simpler_training",
        ],
        "eligible_overall": RANKING_CANDIDATES,
        "eligible_thesis_hybrid": ["N0", "N1"],
        "seed_selection_prohibited": True,
    }


def sample_neural_config(trial: Any, candidate_id: str) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "oversample_method": "none",
        "class_weight_mode": "none",
        "loss": "cross_entropy",
        "smote_ratio": 1.0,
        "resampling_k_neighbors": 5,
        "normalization": trial.suggest_categorical("normalization", ["none", "layer_norm"]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.30),
        "sequence_dropout": 0.0,
        "max_epochs": trial.suggest_categorical("max_epochs", [40, 60, 80]),
        "patience": 10,
    }
    if candidate_id in {"N0", "N1"}:
        parameters.update({
            "cnn_channels": trial.suggest_categorical("cnn_channels", [4, 8, 16]),
            "cnn_kernel_size": trial.suggest_categorical("cnn_kernel_size", [1, 2]),
            "lstm_hidden_dim": trial.suggest_categorical("lstm_hidden_dim", [4, 8, 16]),
            "sequence_dropout": trial.suggest_float("sequence_dropout", 0.0, 0.20),
        })
    else:
        parameters.update({
            "hidden_dim": trial.suggest_categorical("hidden_dim", [4, 8, 16, 32]),
            "num_layers": trial.suggest_categorical("num_layers", [1, 2]),
            "cnn_channels": 1,
            "cnn_kernel_size": 1,
            "lstm_hidden_dim": 1,
        })
    config = resolve_phase_c_neural_config(
        candidate_id,
        parameters,
        suggested_parameters=dict(getattr(trial, "params", {})),
    )
    count = count_trainable_parameters(create_phase_c_model(config))
    config["parameter_count"] = count
    if count > PARAMETER_GUARDRAIL:
        raise ValueError(f"parameter_guardrail_exceeded:{count}>{PARAMETER_GUARDRAIL}")
    validate_resolved_config(config)
    return config


def matched_ablation_config(candidate_id: str, n0_config: dict[str, Any]) -> dict[str, Any]:
    if candidate_id not in ABLATIONS:
        raise ValueError("Only A1/A2 are matched ablations.")
    parameters = {
        key: deepcopy(value)
        for key, value in n0_config.items()
        if key in {
            "learning_rate", "weight_decay", "batch_size", "oversample_method", "class_weight_mode",
            "loss", "smote_ratio", "resampling_k_neighbors", "cnn_channels", "cnn_kernel_size",
            "lstm_hidden_dim", "dropout", "sequence_dropout", "max_epochs", "patience",
            "normalization",
        }
    }
    config = resolve_phase_c_neural_config(
        candidate_id,
        parameters,
        suggested_parameters={},
        evidence_role="phase_c_fixed_parameter_matched_ablation",
    )
    config["parameter_count"] = count_trainable_parameters(create_phase_c_model(config))
    config["matched_to_config_hash"] = resolved_config_hash(n0_config)
    return config


def ml_resolved_config(candidate_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
    required = {
        "M1": {"n_estimators", "max_depth", "min_samples_leaf", "max_features"},
        "M2": {"C", "gamma"},
    }[candidate_id]
    missing = sorted(required - set(parameters))
    if missing:
        raise ValueError(f"ML resolved config missing keys: {missing}")
    return {
        "schema_version": "strategy_b_phase_c_ml_resolved_v1",
        "candidate_id": candidate_id,
        "parameters": deepcopy(parameters),
        "fixed_constants": {
            "features": ["G1", "G2"], "preprocessing": "fold_train_minmax",
            "class_weight": None, "random_state": 42, "outer_refit": "full_outer_train",
        },
    }


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def probability_contract(probabilities: np.ndarray, tolerance: float = 1e-6) -> None:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Probabilities must have shape [records, 3].")
    if not np.isfinite(values).all():
        raise ValueError("Probabilities contain non-finite values.")
    if values.min() < -tolerance or values.max() > 1.0 + tolerance:
        raise ValueError("Probabilities violate the [0,1] range contract.")
    if not np.allclose(values.sum(axis=1), 1.0, atol=tolerance, rtol=0):
        raise ValueError("Class probabilities do not sum to one.")


def detailed_metrics(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return model, fold-seed, per-class, confusion, and ordinal/calibration rows."""

    fold_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    extra_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for (candidate, seed, fold), frame in oof.groupby(["candidate_id", "seed", "outer_fold"], sort=True):
        y = frame["true_label"].to_numpy(int)
        pred = frame["predicted_label"].to_numpy(int)
        probs = frame[["prob_0", "prob_1", "prob_2"]].to_numpy(float)
        probability_contract(probs)
        fold_rows.append({
            "candidate_id": candidate, "seed": int(seed), "outer_fold": int(fold), "records": len(frame),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "class_collapse": len(set(pred)) < 3,
        })
        precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
        for label in range(3):
            class_rows.append({
                "candidate_id": candidate, "seed": int(seed), "outer_fold": int(fold), "class_label": label,
                "precision": precision[label], "recall": recall[label], "f1": f1[label], "support": int(support[label]),
            })
        matrix = confusion_matrix(y, pred, labels=[0, 1, 2])
        for actual in range(3):
            for predicted in range(3):
                confusion_rows.append({
                    "candidate_id": candidate, "seed": int(seed), "outer_fold": int(fold),
                    "actual": actual, "predicted": predicted, "count": int(matrix[actual, predicted]),
                })
        absolute = np.abs(y - pred)
        extra_rows.append({
            "candidate_id": candidate, "seed": int(seed), "outer_fold": int(fold),
            "qwk": cohen_kappa_score(y, pred, weights="quadratic"),
            "ordinal_mae": float(absolute.mean()), "one_step_error_rate": float((absolute == 1).mean()),
            "two_step_error_rate": float((absolute == 2).mean()),
            "brier": multiclass_brier_score(y, probs), "nll": log_loss(y, probs, labels=[0, 1, 2]),
            "ece": expected_calibration_error(y, probs),
        })
    for (candidate, seed), frame in oof.groupby(["candidate_id", "seed"], sort=True):
        seed_rows.append({
            "candidate_id": candidate, "seed": int(seed),
            "seed_oof_macro_f1": f1_score(frame["true_label"], frame["predicted_label"], average="macro", zero_division=0),
        })
    return tuple(pd.DataFrame(rows) for rows in (fold_rows, class_rows, confusion_rows, extra_rows, seed_rows))


def boundary_error_analysis(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, seed, raw_grade), frame in oof[oof["raw_g3"].isin([9, 10, 14, 15])].groupby(
        ["candidate_id", "seed", "raw_g3"], sort=True
    ):
        rows.append({
            "candidate_id": candidate, "seed": int(seed), "raw_g3": int(raw_grade), "records": len(frame),
            "error_rate": float((frame["true_label"] != frame["predicted_label"]).mean()),
        })
    return pd.DataFrame(rows)


def model_summary(
    oof: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    extra_metrics: pd.DataFrame,
    per_class: pd.DataFrame,
    parameter_counts: pd.DataFrame,
    runtimes: pd.DataFrame,
) -> pd.DataFrame:
    registry = {row["id"]: row for row in candidate_registry()["candidates"]}
    rows: list[dict[str, Any]] = []
    for candidate in ALL_CANDIDATES:
        candidate_oof = oof[oof["candidate_id"] == candidate]
        candidate_folds = fold_metrics[fold_metrics["candidate_id"] == candidate]
        candidate_extra = extra_metrics[extra_metrics["candidate_id"] == candidate]
        outer_fold_means = candidate_folds.groupby("outer_fold")["macro_f1"].mean()
        seed_scores = []
        for _, frame in candidate_oof.groupby("seed"):
            seed_scores.append(f1_score(frame["true_label"], frame["predicted_label"], average="macro", zero_division=0))
        high = per_class[(per_class["candidate_id"] == candidate) & (per_class["class_label"] == 2)]
        counts = parameter_counts[parameter_counts["candidate_id"] == candidate]["parameter_count"]
        runtime = runtimes[runtimes["candidate_id"] == candidate]["runtime_seconds"]
        rows.append({
            "candidate_id": candidate, "model": registry[candidate]["name"], "family": registry[candidate]["family"],
            "parameter_count": int(round(counts.mean())) if len(counts) else 0,
            "oof_macro_f1": float(np.mean(seed_scores)),
            "outer_mean_macro_f1": float(candidate_folds["macro_f1"].mean()),
            "outer_sd": float(outer_fold_means.std(ddof=1)) if len(outer_fold_means) > 1 else 0.0,
            "seed_mean": float(np.mean(seed_scores)), "seed_sd": float(np.std(seed_scores, ddof=1)) if len(seed_scores) > 1 else 0.0,
            "seed_median": float(np.median(seed_scores)), "worst_seed": float(np.min(seed_scores)), "best_seed": float(np.max(seed_scores)),
            "accuracy": float(candidate_folds["accuracy"].mean()),
            "balanced_accuracy": float(candidate_folds["balanced_accuracy"].mean()),
            "high_class_f1": float(high["f1"].mean()), "qwk": float(candidate_extra["qwk"].mean()),
            "ordinal_mae": float(candidate_extra["ordinal_mae"].mean()),
            "two_step_error": float(candidate_extra["two_step_error_rate"].mean()),
            "brier": float(candidate_extra["brier"].mean()), "nll": float(candidate_extra["nll"].mean()),
            "ece": float(candidate_extra["ece"].mean()), "runtime_seconds": float(runtime.sum()),
            "class_collapse_count": int(candidate_folds["class_collapse"].sum()),
            "eligible_overall": registry[candidate]["eligible_overall"],
            "eligible_thesis_hybrid": registry[candidate]["eligible_thesis_hybrid"],
        })
    return pd.DataFrame(rows).sort_values("oof_macro_f1", ascending=False).reset_index(drop=True)


def paired_deltas(oof: pd.DataFrame, comparisons: Iterable[tuple[str, str]], bootstrap_samples: int = 1000) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260714)
    for left, right in comparisons:
        left_frame = oof[oof["candidate_id"] == left]
        right_frame = oof[oof["candidate_id"] == right]
        common_seeds = sorted(set(left_frame["seed"]) & set(right_frame["seed"]))
        if not common_seeds:
            # Repeat deterministic/ML predictions against each neural declared seed.
            neural_seeds = sorted(set(left_frame["seed"]) | set(right_frame["seed"]))
            common_seeds = [seed for seed in neural_seeds if seed in PHASE_C_SEEDS] or [42]
        seed_deltas: list[float] = []
        fold_deltas: list[float] = []
        fold_deltas_by_outer: dict[int, list[float]] = {}
        bootstrap_by_seed: list[np.ndarray] = []
        for seed in common_seeds:
            lf = left_frame[left_frame["seed"] == seed]
            rf = right_frame[right_frame["seed"] == seed]
            if lf.empty:
                lf = left_frame.copy()
            if rf.empty:
                rf = right_frame.copy()
            merged = lf.merge(rf, on="source_row_number", suffixes=("_l", "_r"))
            if len(merged) == 0:
                raise ValueError(f"No aligned OOF rows for {left} vs {right}.")
            y = merged["true_label_l"].to_numpy(int)
            lp = merged["predicted_label_l"].to_numpy(int)
            rp = merged["predicted_label_r"].to_numpy(int)
            seed_deltas.append(f1_score(y, lp, average="macro", zero_division=0) - f1_score(y, rp, average="macro", zero_division=0))
            for fold, fold_frame in merged.groupby("outer_fold_l"):
                fy = fold_frame["true_label_l"].to_numpy(int)
                fold_delta = (
                    f1_score(fy, fold_frame["predicted_label_l"], average="macro", zero_division=0)
                    - f1_score(fy, fold_frame["predicted_label_r"], average="macro", zero_division=0)
                )
                fold_deltas.append(fold_delta)
                fold_deltas_by_outer.setdefault(int(fold), []).append(float(fold_delta))
            seed_bootstrap: list[float] = []
            for _ in range(bootstrap_samples):
                idx = rng.integers(0, len(merged), len(merged))
                by = y[idx]
                seed_bootstrap.append(
                    f1_score(by, lp[idx], average="macro", zero_division=0)
                    - f1_score(by, rp[idx], average="macro", zero_division=0)
                )
            bootstrap_by_seed.append(np.asarray(seed_bootstrap, dtype=float))
        bootstrap = np.mean(np.stack(bootstrap_by_seed, axis=0), axis=0)
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        delta = float(np.mean(seed_deltas))
        outer_fold_deltas = [float(np.mean(values)) for _, values in sorted(fold_deltas_by_outer.items())]
        rows.append({
            "left": left, "right": right, "macro_f1_delta_left_minus_right": delta,
            "paired_outer_fold_delta_mean": float(np.mean(outer_fold_deltas)),
            "paired_outer_fold_delta_sd": float(np.std(outer_fold_deltas, ddof=1)) if len(outer_fold_deltas) > 1 else 0.0,
            "paired_outer_fold_deltas": json.dumps(outer_fold_deltas),
            "record_bootstrap_ci_low": float(low), "record_bootstrap_ci_high": float(high),
            "practical_tie": abs(delta) < PRACTICAL_MARGIN or (low <= 0 <= high),
            "bootstrap_interpretation": "descriptive_not_absolute_significance",
        })
    return pd.DataFrame(rows)
