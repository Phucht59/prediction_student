"""Frozen helpers for Strategy B Phase E-Prediction.

This module deliberately treats the Phase C result as immutable input.  It
contains no architecture search routine: stability consumes the Phase C outer
resolved configurations and varies only the declared new random seeds.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    r2_score,
    root_mean_squared_error,
)

from src.model_selection import expected_calibration_error, multiclass_brier_score
from src.strategy_b_phase_c import PRACTICAL_MARGIN, probability_contract


PHASE_E_PROTOCOL_VERSION = "strategy_b_phase_e_prediction_v1"
PHASE_E_SEEDS = [202601, 202602, 202603, 202604, 202605]
PHASE_C_SEEDS = {42, 123, 155}
FINALISTS = ["R0", "M1", "M2", "N0", "N1"]
OVERALL_FINALISTS = ["R0", "M1", "M2"]
HYBRID_FINALISTS = ["N0", "N1"]
DETERMINISTIC_SEED = -1
TEMPERATURE_BRIER_TOLERANCE = 0.005
TEMPERATURE_ECE_TOLERANCE = 0.005


def phase_e_registry() -> dict[str, Any]:
    return {
        "registry_version": "strategy_b_phase_e_finalists_v1",
        "overall_finalists": OVERALL_FINALISTS,
        "thesis_hybrid_finalists": HYBRID_FINALISTS,
        "excluded_phase_c_candidates": ["N2", "N3", "A1", "A2"],
        "prohibited_branches": ["C1_huber", "C2_residual", "imbalance", "context", "recommendation_phase_d"],
        "frozen_phase_c_interpretation": {
            "phase_c_provisional_overall": "M1",
            "phase_c_protocol_selected_hybrid": "N1",
            "phase_c_n0_higher_point_macro_f1_and_better_calibration": True,
            "phase_c_n0_n1_no_clear_superiority": True,
            "phase_c_cnn_incremental_value_not_established": True,
            "phase_c_ordinal_incremental_value_not_established": True,
        },
    }


def seed_registry() -> dict[str, Any]:
    return {
        "phase_c_seeds_excluded": sorted(PHASE_C_SEEDS),
        "new_stability_seeds": PHASE_E_SEEDS,
        "R0": {"deterministic_rule": True, "seed_not_applicable": True, "stored_seed": DETERMINISTIC_SEED},
        "M1": {"stochastic": True, "seeds": PHASE_E_SEEDS, "best_seed_selection": False},
        "M2": {"deterministic_replay_verified": True, "seed_not_applicable": True, "stored_seed": DETERMINISTIC_SEED},
        "N0": {"stochastic": True, "seeds": PHASE_E_SEEDS, "best_seed_selection": False},
        "N1": {"stochastic": True, "seeds": PHASE_E_SEEDS, "best_seed_selection": False},
    }


def selection_rule() -> dict[str, Any]:
    return {
        "primary_metric": "mean new-seed OOF macro_f1",
        "practical_margin_absolute_macro_f1": PRACTICAL_MARGIN,
        "uncertainty": "paired_record_bootstrap_descriptive_only",
        "no_clear_superiority_if": ["absolute_delta_below_0.01", "paired_bootstrap_interval_includes_zero"],
        "tie_break_order": [
            "fewer_class_collapses", "lower_genuine_seed_sd", "higher_worst_seed",
            "lower_two_step_error", "lower_ece", "fewer_parameters", "simpler_training",
        ],
        "deterministic_seed_handling": "seed_sd_not_applicable; deterministic rows are never duplicated",
        "no_best_seed_selection": True,
        "secondary_metrics_cannot_replace_macro_f1": ["accuracy", "weighted_f1", "rmse", "r2", "pr_auc"],
    }


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Scalar temperature on clipped log probabilities, with stable softmax."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    probability_contract(probabilities)
    logits = np.log(np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    probability_contract(calibrated)
    return calibrated


def fit_temperature(inner_probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Fit only on explicitly supplied inner-OOF predictions and labels."""
    probability_contract(inner_probabilities)
    labels = np.asarray(labels, dtype=int)
    if len(labels) != len(inner_probabilities) or not set(labels).issubset({0, 1, 2}):
        raise ValueError("Temperature fitting needs aligned three-class inner labels.")

    def objective(log_temperature: float) -> float:
        return float(log_loss(labels, apply_temperature(inner_probabilities, float(np.exp(log_temperature))), labels=[0, 1, 2]))

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded", options={"xatol": 1e-8})
    temperature = float(np.exp(result.x))
    return {
        "method": "scalar_temperature_on_clipped_log_probabilities",
        "temperature": temperature,
        "optimizer_success": bool(result.success),
        "inner_oof_nll_before": objective(0.0),
        "inner_oof_nll_after": objective(float(result.x)),
    }


def calibration_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probability_contract(probabilities)
    return {
        "nll": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "brier": float(multiclass_brier_score(y, probabilities)),
        "ece": float(expected_calibration_error(y, probabilities)),
    }


def choose_temperature(calibration: pd.DataFrame) -> dict[str, Any]:
    """Pre-registered candidate-level temperature retention decision."""
    required = {"candidate_id", "outer_fold", "variant", "nll", "brier", "ece"}
    if not required <= set(calibration.columns):
        raise ValueError("Calibration metric columns are incomplete.")
    decisions: dict[str, Any] = {}
    for candidate, frame in calibration.groupby("candidate_id", sort=True):
        if set(frame["variant"]) == {"uncalibrated"}:
            decisions[candidate] = {"selected_variant": "uncalibrated", "reason": "no_probabilistic_temperature_basis"}
            continue
        pivot = frame.groupby(["outer_fold", "variant"])[["nll", "brier", "ece"]].mean().unstack("variant")
        before = pivot.xs("uncalibrated", level=1, axis=1)
        after = pivot.xs("temperature", level=1, axis=1)
        nll_majority = int((after["nll"] < before["nll"]).sum()) > len(before) / 2
        brier_ok = bool((after["brier"] <= before["brier"] + TEMPERATURE_BRIER_TOLERANCE).all())
        ece_ok = bool((after["ece"] <= before["ece"] + TEMPERATURE_ECE_TOLERANCE).all())
        decisions[candidate] = {
            "selected_variant": "temperature" if nll_majority and brier_ok and ece_ok else "uncalibrated",
            "nll_improved_majority_outer_folds": nll_majority,
            "brier_tolerance": TEMPERATURE_BRIER_TOLERANCE,
            "brier_not_worse_within_tolerance": brier_ok,
            "ece_tolerance": TEMPERATURE_ECE_TOLERANCE,
            "ece_not_worse_within_tolerance": ece_ok,
            "decision_predictions_unchanged_by_positive_temperature": True,
        }
    return decisions


def classification_rows(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, seed, fold), frame in oof.groupby(["candidate_id", "seed", "outer_fold"], sort=True):
        y = frame["true_label"].to_numpy(int)
        pred = frame["predicted_label"].to_numpy(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
        row = {
            "candidate_id": candidate, "outer_fold": int(fold), "seed": int(seed),
            "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_precision": float(precision_recall_fscore_support(y, pred, average="macro", zero_division=0)[0]),
            "micro_precision": float(precision_recall_fscore_support(y, pred, average="micro", zero_division=0)[0]),
            "weighted_precision": float(precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)[0]),
            "macro_recall": float(precision_recall_fscore_support(y, pred, average="macro", zero_division=0)[1]),
            "micro_recall": float(precision_recall_fscore_support(y, pred, average="micro", zero_division=0)[1]),
            "weighted_recall": float(precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)[1]),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "class_collapse": bool(len(set(pred)) < 3),
        }
        for index, name in enumerate(["low", "medium", "high"]):
            row[f"{name}_precision"] = float(precision[index])
            row[f"{name}_recall"] = float(recall[index])
            row[f"{name}_f1"] = float(f1[index])
        rows.append(row)
    return pd.DataFrame(rows)


def precision_recall_rows(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    for (candidate, seed, fold), frame in oof.groupby(["candidate_id", "seed", "outer_fold"], sort=True):
        y = frame["true_label"].to_numpy(int)
        probabilities = frame[["prob_0", "prob_1", "prob_2"]].to_numpy(float)
        probability_contract(probabilities)
        one_hot = np.eye(3, dtype=int)[y]
        aps: list[float] = []
        supports = one_hot.sum(axis=0).astype(float)
        for label, name in enumerate(["Low", "Medium", "High"]):
            binary = one_hot[:, label]
            ap = float(average_precision_score(binary, probabilities[:, label]))
            aps.append(ap)
            precision, recall, thresholds = precision_recall_curve(binary, probabilities[:, label])
            output_thresholds = np.concatenate([thresholds, [1.0]])
            for threshold, p, r in zip(output_thresholds, precision, recall):
                points.append({
                    "candidate_id": candidate, "outer_fold": int(fold), "seed": int(seed),
                    "class_name": name, "threshold": float(threshold), "precision": float(p), "recall": float(r),
                })
        metrics.append({
            "candidate_id": candidate, "outer_fold": int(fold), "seed": int(seed),
            "low_average_precision": aps[0], "medium_average_precision": aps[1], "high_average_precision": aps[2],
            "macro_pr_auc": float(np.mean(aps)),
            "micro_pr_auc": float(average_precision_score(one_hot.ravel(), probabilities.ravel())),
            "weighted_pr_auc": float(np.average(aps, weights=supports)),
        })
    return pd.DataFrame(metrics), pd.DataFrame(points)


def regression_rows(continuous: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, seed, fold, method), frame in continuous.groupby(
        ["candidate_id", "seed", "outer_fold", "continuous_prediction_method"], sort=True
    ):
        y = frame["true_g3"].to_numpy(float)
        predicted = frame["predicted_g3"].to_numpy(float)
        rows.append({
            "candidate_id": candidate, "outer_fold": int(fold), "seed": int(seed),
            "continuous_prediction_method": method, "mae": float(np.abs(y - predicted).mean()),
            "rmse": float(root_mean_squared_error(y, predicted)), "r2": float(r2_score(y, predicted)),
        })
    return pd.DataFrame(rows)


def seed_stability(oof: pd.DataFrame, classification: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, frame in oof.groupby("candidate_id", sort=True):
        # Stability is defined on each seed's complete 316-record OOF vector,
        # not as an unweighted average of fold-level Macro-F1 values.
        seed_values = frame.groupby("seed").apply(
            lambda value: f1_score(value["true_label"], value["predicted_label"], average="macro", zero_division=0),
            include_groups=False,
        )
        stochastic = len(seed_values) > 1
        prediction_table = frame.pivot(index="source_row_number", columns="seed", values="predicted_label")
        prob_columns = ["prob_0", "prob_1", "prob_2"]
        probability_variance = np.nan
        disagreement = np.nan
        if stochastic:
            disagreements = []
            for left in prediction_table.columns:
                for right in prediction_table.columns:
                    if int(left) < int(right):
                        disagreements.append(float((prediction_table[left] != prediction_table[right]).mean()))
            disagreement = float(np.mean(disagreements))
            values = frame.pivot(index="source_row_number", columns="seed", values=prob_columns)
            probability_variance = float(values.var(axis=1, ddof=0).to_numpy().mean())
        rows.append({
            "candidate_id": candidate, "seed_count": int(len(seed_values)),
            "oof_macro_f1_mean": float(seed_values.mean()),
            "seed_sd": float(seed_values.std(ddof=1)) if stochastic else np.nan,
            "seed_sd_not_applicable": not stochastic,
            "seed_median": float(seed_values.median()), "worst_seed": float(seed_values.min()), "best_seed": float(seed_values.max()),
            "prediction_disagreement_rate": disagreement, "probability_variance": probability_variance,
        })
    return pd.DataFrame(rows)


def _aligned(left: pd.DataFrame, right: pd.DataFrame, seed: int | None) -> pd.DataFrame:
    lframe = left if seed is None or seed not in set(left["seed"]) else left[left["seed"] == seed]
    rframe = right if seed is None or seed not in set(right["seed"]) else right[right["seed"] == seed]
    return lframe.merge(rframe, on="source_row_number", suffixes=("_l", "_r"), validate="one_to_one")


def paired_metric_deltas(
    oof: pd.DataFrame,
    continuous: pd.DataFrame,
    comparisons: Iterable[tuple[str, str]],
    *, bootstrap_samples: int = 1000,
) -> pd.DataFrame:
    """Descriptive paired deltas; deterministic predictions are aligned, not duplicated."""
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260714)
    for left_id, right_id in comparisons:
        left = oof[oof["candidate_id"] == left_id]
        right = oof[oof["candidate_id"] == right_id]
        raw_seeds = set(left["seed"]) | set(right["seed"])
        # When only one side is stochastic, align each genuine stochastic seed
        # against the one deterministic prediction.  Do not add an artificial
        # aggregate "-1" seed comparison.
        seeds = sorted(seed for seed in raw_seeds if seed != DETERMINISTIC_SEED)
        if not seeds:
            seeds = [DETERMINISTIC_SEED]
        seed_metrics: list[dict[str, float]] = []
        bootstrap: list[float] = []
        fold_values: dict[int, list[float]] = {}
        for seed in seeds:
            merged = _aligned(left, right, seed)
            y = merged["true_label_l"].to_numpy(int)
            lp, rp = merged["predicted_label_l"].to_numpy(int), merged["predicted_label_r"].to_numpy(int)
            lprob = merged[["prob_0_l", "prob_1_l", "prob_2_l"]].to_numpy(float)
            rprob = merged[["prob_0_r", "prob_1_r", "prob_2_r"]].to_numpy(float)
            lpr = precision_recall_fscore_support(y, lp, labels=[0, 1, 2], average="macro", zero_division=0)
            rpr = precision_recall_fscore_support(y, rp, labels=[0, 1, 2], average="macro", zero_division=0)
            high_l = precision_recall_fscore_support(y, lp, labels=[0, 1, 2], zero_division=0)
            high_r = precision_recall_fscore_support(y, rp, labels=[0, 1, 2], zero_division=0)
            one_hot = np.eye(3, dtype=int)[y]
            left_continuous = continuous[continuous["candidate_id"] == left_id]
            right_continuous = continuous[continuous["candidate_id"] == right_id]
            continuous_left = left_continuous[left_continuous["seed"] == seed] if seed in set(left_continuous["seed"]) else left_continuous
            continuous_right = right_continuous[right_continuous["seed"] == seed] if seed in set(right_continuous["seed"]) else right_continuous
            cm = continuous_left.merge(continuous_right, on="source_record_id", suffixes=("_l", "_r"), validate="one_to_one")
            values = {
                "accuracy": float(accuracy_score(y, lp) - accuracy_score(y, rp)),
                "macro_precision": float(lpr[0] - rpr[0]), "macro_recall": float(lpr[1] - rpr[1]),
                "macro_f1": float(lpr[2] - rpr[2]),
                "high_precision": float(high_l[0][2] - high_r[0][2]),
                "high_recall": float(high_l[1][2] - high_r[1][2]), "high_f1": float(high_l[2][2] - high_r[2][2]),
                "macro_pr_auc": float(np.mean([average_precision_score(one_hot[:, c], lprob[:, c]) - average_precision_score(one_hot[:, c], rprob[:, c]) for c in range(3)])),
                "rmse": float(root_mean_squared_error(cm["true_g3_l"], cm["predicted_g3_l"]) - root_mean_squared_error(cm["true_g3_r"], cm["predicted_g3_r"])),
                "r2": float(r2_score(cm["true_g3_l"], cm["predicted_g3_l"]) - r2_score(cm["true_g3_r"], cm["predicted_g3_r"])),
            }
            values["seed"] = int(seed)
            seed_metrics.append(values)
            for fold, group in merged.groupby("outer_fold_l"):
                fold_values.setdefault(int(fold), []).append(float(f1_score(group["true_label_l"], group["predicted_label_l"], average="macro", zero_division=0) - f1_score(group["true_label_l"], group["predicted_label_r"], average="macro", zero_division=0)))
            for _ in range(bootstrap_samples):
                positions = rng.integers(0, len(merged), len(merged))
                bootstrap.append(float(f1_score(y[positions], lp[positions], average="macro", zero_division=0) - f1_score(y[positions], rp[positions], average="macro", zero_division=0)))
        aggregate = pd.DataFrame(seed_metrics)
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        row: dict[str, Any] = {"left": left_id, "right": right_id, "comparison": f"{left_id}_minus_{right_id}"}
        for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "high_precision", "high_recall", "high_f1", "macro_pr_auc", "rmse", "r2"]:
            row[f"{metric}_delta_left_minus_right"] = float(aggregate[metric].mean())
        row.update({
            "macro_f1_record_bootstrap_ci_low": float(low), "macro_f1_record_bootstrap_ci_high": float(high),
            "macro_f1_practical_tie": bool(abs(row["macro_f1_delta_left_minus_right"]) < PRACTICAL_MARGIN or low <= 0 <= high),
            "seed_deltas_json": json.dumps(seed_metrics),
            "outer_fold_macro_f1_deltas_json": json.dumps([float(np.mean(v)) for _, v in sorted(fold_values.items())]),
            "bootstrap_interpretation": "descriptive_not_absolute_significance",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def choose_final(summary: pd.DataFrame, eligible: list[str], paired: pd.DataFrame) -> tuple[str, str]:
    """Apply Phase C rule without turning deterministic seed absence into zero SD."""
    candidates = summary[summary["candidate_id"].isin(eligible)].sort_values("oof_macro_f1", ascending=False).copy()
    leader = str(candidates.iloc[0]["candidate_id"])
    leader_score = float(candidates.iloc[0]["oof_macro_f1"])
    tied = [leader]
    for _, candidate in candidates.iloc[1:].iterrows():
        candidate_id = str(candidate["candidate_id"])
        pair = paired[((paired["left"] == leader) & (paired["right"] == candidate_id)) | ((paired["left"] == candidate_id) & (paired["right"] == leader))]
        interval_zero = bool(len(pair) and float(pair.iloc[0]["macro_f1_record_bootstrap_ci_low"]) <= 0 <= float(pair.iloc[0]["macro_f1_record_bootstrap_ci_high"]))
        if leader_score - float(candidate["oof_macro_f1"]) < PRACTICAL_MARGIN or interval_zero:
            tied.append(candidate_id)
    tied_rows = candidates[candidates["candidate_id"].isin(tied)].copy()
    if len(tied_rows) == 1:
        return leader, "clear_by_practical_margin"
    # The genuine-seed criterion only resolves a tie when both candidates have it.
    minimum_collapse = tied_rows["class_collapse_count"].min()
    tied_rows = tied_rows[tied_rows["class_collapse_count"] == minimum_collapse]
    stochastic = tied_rows[tied_rows["seed_sd_not_applicable"] == False]  # noqa: E712
    if len(stochastic) == len(tied_rows) and len(stochastic) > 1:
        best_sd = stochastic["seed_sd"].min()
        tied_rows = stochastic[stochastic["seed_sd"] == best_sd]
    for column, ascending in [("worst_seed", False), ("two_step_error", True), ("ece", True), ("parameter_count", True), ("simplicity_rank", True)]:
        best = tied_rows[column].min() if ascending else tied_rows[column].max()
        tied_rows = tied_rows[tied_rows[column] == best]
        if len(tied_rows) == 1:
            break
    return str(tied_rows.sort_values("candidate_id").iloc[0]["candidate_id"]), "practical_tie_resolved_by_preregistered_tiebreak"
