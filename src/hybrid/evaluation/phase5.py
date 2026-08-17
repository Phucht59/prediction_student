"""Frozen Phase 5 metrics, enumeration, and paired-bootstrap helpers."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASELINE_FAMILIES = (
    "logistic_regression",
    "svm",
    "random_forest",
    "xgboost",
    "catboost",
    "mlp",
)
SEEDS = (42, 1201, 2026, 3407, 7319)
STAGES = {"uci": ("S0", "S1", "S2"), "oulad": ("20pct", "35pct", "50pct", "75pct")}
OUTER_FOLDS = {"uci": 5, "oulad": 3}


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def expected_hybrid_jobs(*, include_final: bool = False) -> list[tuple[str, int, int, str]]:
    jobs = [
        (domain, fold, seed, "operational")
        for domain in ("uci", "oulad")
        for fold in range(OUTER_FOLDS[domain])
        for seed in SEEDS
    ]
    if include_final:
        jobs.extend(("oulad", fold, seed, "final") for fold in range(3) for seed in SEEDS)
    return jobs


def expected_baseline_jobs(*, include_final: bool = False) -> list[tuple[str, str, int, str, int]]:
    jobs = [
        (domain, stage, fold, family, seed)
        for domain in ("uci", "oulad")
        for stage in STAGES[domain]
        for fold in range(OUTER_FOLDS[domain])
        for family in BASELINE_FAMILIES
        for seed in SEEDS
    ]
    if include_final:
        jobs.extend(("oulad", "FINAL", fold, family, seed) for fold in range(3) for family in BASELINE_FAMILIES for seed in SEEDS)
    return jobs


def probability_calibration(target: Iterable[int], probability: Iterable[float], bins: int = 15) -> dict[str, float]:
    y = np.asarray(target, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    if len(y) != len(p) or not len(y) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Calibration requires finite aligned probabilities in [0,1]")
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.minimum(np.searchsorted(edges, p, side="right") - 1, bins - 1)
    ece = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return {"brier_score": float(brier_score_loss(y, p)), "ece_15_equal_width": float(ece)}


def classification_metrics(target: Iterable[int], score: Iterable[float], *, threshold: float, probability: bool) -> dict[str, float | None]:
    y = np.asarray(target, dtype=np.int64)
    s = np.asarray(score, dtype=np.float64)
    if len(y) != len(s) or len(np.unique(y)) != 2 or not np.isfinite(s).all():
        raise ValueError("Final metrics require aligned finite scores and both classes")
    prediction = (s >= threshold).astype(np.int64)
    result: dict[str, float | None] = {
        "pooled_outer_oof_pr_auc": float(average_precision_score(y, s)),
        "pooled_outer_oof_roc_auc": float(roc_auc_score(y, s)),
        "risk_precision": float(precision_score(y, prediction, zero_division=0)),
        "risk_recall": float(recall_score(y, prediction, zero_division=0)),
        "risk_f1": float(f1_score(y, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "brier_score": None,
        "ece_15_equal_width": None,
    }
    if probability:
        result.update(probability_calibration(y, s, 15))
    return result


def ensemble_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["record_id", "group_id", "domain", "stage", "outer_fold", "model_family", "target"]
    if frame.duplicated(keys + ["seed"]).any():
        raise ValueError("Duplicate seed-level outer prediction")
    counts = frame.groupby(keys, dropna=False).seed.nunique()
    if not (counts == len(SEEDS)).all():
        raise ValueError("Every ensemble row must contain exactly five seeds")
    return frame.groupby(keys, as_index=False, dropna=False).agg(
        ranking_score=("ranking_score", "mean"),
        probability=("probability", "mean"),
    )


def paired_group_bootstrap(
    paired: pd.DataFrame,
    stages: tuple[str, ...],
    *,
    replicates: int = 5000,
    seed: int = 20260815,
) -> list[dict[str, float | int | str]]:
    """Paired group bootstrap for each stage and the equal-stage macro delta."""
    required = {"group_id", "stage", "target", "hybrid_score", "baseline_score"}
    if not required.issubset(paired.columns):
        raise ValueError(f"Missing bootstrap columns: {sorted(required-set(paired.columns))}")
    rng = np.random.default_rng(seed)
    group_values = paired.group_id.astype(str)
    groups = np.asarray(sorted(group_values.unique()))
    group_lookup = {group: index for index, group in enumerate(groups)}
    group_code = group_values.map(group_lookup).to_numpy(dtype=np.int64)
    prepared = {}
    observed = {}
    for stage in stages:
        mask = paired.stage == stage
        part = paired[mask]
        observed[stage] = float(average_precision_score(part.target, part.hybrid_score) - average_precision_score(part.target, part.baseline_score))
        y = part.target.to_numpy(dtype=np.int64)
        hybrid = part.hybrid_score.to_numpy(dtype=np.float64)
        baseline = part.baseline_score.to_numpy(dtype=np.float64)
        hybrid_order = np.argsort(-hybrid, kind="mergesort")
        baseline_order = np.argsort(-baseline, kind="mergesort")
        prepared[stage] = {
            "target": y,
            "group_code": group_code[mask.to_numpy()],
            "hybrid_order": hybrid_order,
            "hybrid_threshold_end": np.r_[np.where(np.diff(hybrid[hybrid_order]))[0], len(hybrid_order) - 1],
            "baseline_order": baseline_order,
            "baseline_threshold_end": np.r_[np.where(np.diff(baseline[baseline_order]))[0], len(baseline_order) - 1],
        }
    values = {stage: [] for stage in stages}
    macro_values: list[float] = []
    skipped = {stage: 0 for stage in stages}
    macro_skipped = 0
    for _ in range(replicates):
        weights_by_group = rng.multinomial(len(groups), np.full(len(groups), 1.0 / len(groups)))
        draw_values = []
        for stage in stages:
            item = prepared[stage]
            weights = weights_by_group[item["group_code"]].astype(np.float64)
            positive = float(np.sum(weights * item["target"]))
            negative = float(np.sum(weights * (1 - item["target"])))
            if positive <= 0 or negative <= 0:
                skipped[stage] += 1
                continue
            delta = _weighted_average_precision(item["target"], item["hybrid_order"], item["hybrid_threshold_end"], weights) - _weighted_average_precision(item["target"], item["baseline_order"], item["baseline_threshold_end"], weights)
            values[stage].append(delta)
            draw_values.append(delta)
        if len(draw_values) == len(stages):
            macro_values.append(float(np.mean(draw_values)))
        else:
            macro_skipped += 1
    rows = []
    for stage in stages:
        array = np.asarray(values[stage], dtype=np.float64)
        rows.append(_bootstrap_row(stage, observed[stage], array, replicates, skipped[stage]))
    observed_macro = float(np.mean(list(observed.values())))
    rows.append(_bootstrap_row("MACRO", observed_macro, np.asarray(macro_values), replicates, macro_skipped))
    return rows


def _weighted_average_precision(target: np.ndarray, order: np.ndarray, threshold_end: np.ndarray, weights: np.ndarray) -> float:
    y = target[order]
    weight = weights[order]
    true_positive = np.cumsum(weight * y)
    false_positive = np.cumsum(weight * (1 - y))
    true_positive = true_positive[threshold_end]
    false_positive = false_positive[threshold_end]
    total_positive = true_positive[-1]
    recall = true_positive / total_positive
    precision = true_positive / np.maximum(true_positive + false_positive, np.finfo(float).eps)
    previous = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - previous) * precision))


def _bootstrap_row(scope: str, observed: float, values: np.ndarray, requested: int, skipped: int) -> dict[str, float | int | str]:
    if values.size == 0:
        raise ValueError(f"All bootstrap replicates invalid for {scope}")
    return {
        "scope": scope,
        "observed_delta_pr_auc": observed,
        "bootstrap_mean_delta_pr_auc": float(values.mean()),
        "ci_lower_95": float(np.percentile(values, 2.5)),
        "ci_upper_95": float(np.percentile(values, 97.5)),
        "replicates_requested": requested,
        "replicates_valid": int(values.size),
        "replicates_skipped_single_class": skipped,
    }


def classify_domain(macro_delta: float, ci_lower: float, ci_upper: float, stage_deltas: Iterable[float], severe_tradeoff: bool) -> str:
    deltas = np.asarray(list(stage_deltas), dtype=float)
    if macro_delta > 0 and ci_lower > 0 and not severe_tradeoff:
        return "SUPERIOR"
    if macro_delta >= -0.005 and ci_lower <= 0 <= ci_upper and not severe_tradeoff:
        return "COMPETITIVE"
    if (macro_delta < -0.005 and ci_upper < 0) or severe_tradeoff:
        return "UNDERPERFORMS"
    if np.any(deltas > 0) and np.any(deltas < 0):
        return "MIXED"
    return "COMPETITIVE" if macro_delta >= -0.005 else "MIXED"
