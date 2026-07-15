from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def choose_thresholds(y_true: np.ndarray, probabilities: np.ndarray, precision_constraint: float = 0.75) -> dict[str, float | bool | None]:
    thresholds = np.linspace(0.05, 0.95, 901)
    macro_rows: list[tuple[float, float]] = []
    feasible: list[tuple[float, float, float]] = []
    best_achievable: tuple[float, float, float] = (0.5, 0.0, 0.0)
    for threshold in thresholds:
        prediction = (probabilities >= threshold).astype(int)
        macro = f1_score(y_true, prediction, average="macro", zero_division=0)
        precision = precision_score(y_true, prediction, zero_division=0)
        recall = recall_score(y_true, prediction, zero_division=0)
        macro_rows.append((float(threshold), float(macro)))
        if (precision, recall) > (best_achievable[1], best_achievable[2]):
            best_achievable = (float(threshold), float(precision), float(recall))
        if prediction.sum() > 0 and precision >= precision_constraint:
            feasible.append((float(threshold), float(precision), float(recall)))
    best_macro = max(row[1] for row in macro_rows)
    macro_threshold = min((row for row in macro_rows if abs(row[1] - best_macro) < 1e-12), key=lambda row: abs(row[0] - 0.5))[0]
    if feasible:
        operational = max(feasible, key=lambda row: (row[2], row[1], -abs(row[0] - macro_threshold)))
        operational_threshold, operational_precision, operational_recall = operational
        operational_feasible = True
    else:
        operational_threshold, operational_precision, operational_recall = best_achievable
        operational_feasible = False
    return {
        "macro_threshold": macro_threshold,
        "inner_macro_f1": best_macro,
        "operational_threshold": operational_threshold,
        "operational_feasible": operational_feasible,
        "inner_operational_precision": operational_precision,
        "inner_operational_recall": operational_recall,
    }


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (probabilities >= lower) & (probabilities < upper if upper < 1 else probabilities <= upper)
        if selected.any():
            result += selected.mean() * abs(float(probabilities[selected].mean()) - float(y_true[selected].mean()))
    return float(result)


def binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    macro_threshold: float,
    operational_threshold: float | None,
    operational_feasible: bool,
) -> dict[str, float | bool | None]:
    prediction = (probabilities >= macro_threshold).astype(int)
    tn = int(((y_true == 0) & (prediction == 0)).sum())
    fp = int(((y_true == 0) & (prediction == 1)).sum())
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    result: dict[str, float | bool | None] = {
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "at_risk_precision": float(precision_score(y_true, prediction, zero_division=0)),
        "at_risk_recall": float(recall_score(y_true, prediction, zero_division=0)),
        "at_risk_f1": float(f1_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "nll": float(log_loss(y_true, np.column_stack([1 - clipped, clipped]), labels=[0, 1])),
        "ece": expected_calibration_error(y_true, probabilities),
        "class_collapse": bool(len(np.unique(prediction)) < 2),
        "predicted_positive_rate": float(prediction.mean()),
        "operational_feasible": bool(operational_feasible),
        "operational_precision": None,
        "operational_recall": None,
    }
    if operational_threshold is not None:
        operational_prediction = (probabilities >= operational_threshold).astype(int)
        result["operational_precision"] = float(precision_score(y_true, operational_prediction, zero_division=0))
        result["operational_recall"] = float(recall_score(y_true, operational_prediction, zero_division=0))
    return result


def prediction_frame_metrics(frame: pd.DataFrame) -> dict[str, float | bool | None]:
    y_true = frame["target_at_risk"].to_numpy(dtype=int)
    probabilities = frame["probability"].to_numpy(dtype=float)
    prediction = frame["predicted_label"].to_numpy(dtype=int)
    tn = int(((y_true == 0) & (prediction == 0)).sum())
    fp = int(((y_true == 0) & (prediction == 1)).sum())
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    operational_prediction = frame["operational_prediction"].to_numpy(dtype=int) if "operational_prediction" in frame else None
    feasible = bool(frame["operational_feasible"].all()) if "operational_feasible" in frame else False
    return {
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "at_risk_precision": float(precision_score(y_true, prediction, zero_division=0)),
        "at_risk_recall": float(recall_score(y_true, prediction, zero_division=0)),
        "at_risk_f1": float(f1_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "nll": float(log_loss(y_true, np.column_stack([1 - clipped, clipped]), labels=[0, 1])),
        "ece": expected_calibration_error(y_true, probabilities),
        "class_collapse": bool(len(np.unique(prediction)) < 2),
        "predicted_positive_rate": float(prediction.mean()),
        "operational_feasible": feasible,
        "operational_precision": float(precision_score(y_true, operational_prediction, zero_division=0)) if operational_prediction is not None else None,
        "operational_recall": float(recall_score(y_true, operational_prediction, zero_division=0)) if operational_prediction is not None else None,
    }


def module_metrics(predictions: pd.DataFrame, minimum_records: int = 60, minimum_positive: int = 10, minimum_negative: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, frame in predictions.groupby(["candidate_id", "seed", "code_module"], dropna=False):
        positive = int(frame["target_at_risk"].sum())
        negative = int(len(frame) - positive)
        eligible = len(frame) >= minimum_records and positive >= minimum_positive and negative >= minimum_negative
        metrics = prediction_frame_metrics(frame)
        rows.append(
            {
                "candidate_id": keys[0],
                "seed": keys[1],
                "code_module": keys[2],
                "records": len(frame),
                "positive": positive,
                "negative": negative,
                "eligible": eligible,
                "macro_f1": metrics["macro_f1"],
                "at_risk_recall": metrics["at_risk_recall"],
            }
        )
    return pd.DataFrame(rows)


def grouped_bootstrap_delta(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    metric: str = "macro_f1",
    resamples: int = 2000,
    seed: int = 3407,
) -> dict[str, float | int]:
    join_keys = ["record_id", "seed"]
    merged = left.merge(right, on=join_keys, suffixes=("_left", "_right"), validate="one_to_one")
    groups = np.asarray(sorted(merged["id_student_left"].unique()))
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(resamples):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        pieces = [merged.loc[merged["id_student_left"] == group] for group in sampled]
        frame = pd.concat(pieces, ignore_index=True)
        left_metrics = binary_metrics(
            frame["target_at_risk_left"].to_numpy(dtype=int),
            frame["probability_left"].to_numpy(dtype=float),
            0.5,
            None,
            False,
        )
        right_metrics = binary_metrics(
            frame["target_at_risk_right"].to_numpy(dtype=int),
            frame["probability_right"].to_numpy(dtype=float),
            0.5,
            None,
            False,
        )
        deltas.append(float(left_metrics[metric]) - float(right_metrics[metric]))
    return {
        "resamples": resamples,
        "mean_delta": float(np.mean(deltas)),
        "lower_95": float(np.quantile(deltas, 0.025)),
        "upper_95": float(np.quantile(deltas, 0.975)),
        "probability_delta_gt_zero": float(np.mean(np.asarray(deltas) > 0)),
    }
