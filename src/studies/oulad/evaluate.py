from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def validate_binary_probabilities(probabilities: np.ndarray) -> None:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or values.min(initial=0) < 0 or values.max(initial=1) > 1:
        raise ValueError("Invalid binary probabilities")


def tune_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    validate_binary_probabilities(probabilities)
    choices = []
    for threshold in np.linspace(0.05, 0.95, 181):
        score = f1_score(y_true, probabilities >= threshold, average="macro", zero_division=0)
        choices.append((score, -abs(threshold - 0.5), threshold))
    score, _, threshold = max(choices)
    return float(threshold), float(score)


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & (probabilities < high if high < 1 else probabilities <= high)
        if mask.any():
            value += mask.mean() * abs(float(y_true[mask].mean()) - float(probabilities[mask].mean()))
    return float(value)


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float | str, predictions: np.ndarray | None = None) -> dict[str, object]:
    validate_binary_probabilities(probabilities)
    if predictions is None:
        if not isinstance(threshold, (int, float)):
            raise ValueError("Numeric threshold required when predictions are absent")
        predictions = (probabilities >= float(threshold)).astype(int)
    p, r, f, support = precision_recall_fscore_support(y_true, predictions, labels=[0, 1], zero_division=0)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "records": len(y_true), "prevalence": float(np.mean(y_true)), "threshold": threshold,
        "macro_f1": f1_score(y_true, predictions, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, predictions), "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "at_risk_precision": p[1], "at_risk_recall": r[1], "at_risk_f1": f[1],
        "specificity": tn / (tn + fp) if tn + fp else 0.0, "npv": tn / (tn + fn) if tn + fn else 0.0,
        "pr_auc": average_precision_score(y_true, probabilities), "roc_auc": roc_auc_score(y_true, probabilities),
        "brier": float(np.mean((probabilities - y_true) ** 2)),
        "nll": log_loss(y_true, np.column_stack([1 - probabilities, probabilities]), labels=[0, 1]),
        "ece": expected_calibration_error(y_true, probabilities), "class_collapse": len(np.unique(predictions)) < 2,
        "confusion_matrix": matrix.tolist(),
    }
