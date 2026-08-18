from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize


def _probability_matrix(probability: np.ndarray, classes: int) -> np.ndarray:
    values = np.asarray(probability, dtype=float)
    if values.ndim != 2 or values.shape[1] != classes:
        raise ValueError(f"Expected probability shape [n,{classes}], got {values.shape}")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid probability value")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Multiclass probabilities do not sum to one")
    # Normalize tiny floating-point drift before sklearn scoring. The contract
    # above still rejects materially invalid rows.
    return values / values.sum(axis=1, keepdims=True)


def multiclass_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    regression_target: np.ndarray | None = None,
    regression_prediction: np.ndarray | None = None,
) -> dict[str, object]:
    y = np.asarray(target, dtype=int)
    proba = _probability_matrix(probability, 3)
    predicted = proba.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=[0, 1, 2], zero_division=0
    )
    try:
        macro_pr_auc = float(
            average_precision_score(label_binarize(y, classes=[0, 1, 2]), proba, average="macro")
        )
    except ValueError:
        macro_pr_auc = float("nan")
    result: dict[str, object] = {
        "records": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, predicted, average="weighted", zero_division=0)),
        "macro_pr_auc": macro_pr_auc,
        "nll": float(log_loss(y, proba, labels=[0, 1, 2])),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(["low", "medium", "high"])
        },
    }
    if regression_target is not None and regression_prediction is not None:
        raw = np.asarray(regression_target, dtype=float)
        estimate = np.asarray(regression_prediction, dtype=float)
        result["rmse"] = float(mean_squared_error(raw, estimate) ** 0.5)
        result["r2"] = float(r2_score(raw, estimate))
    return result


def expected_calibration_error(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(target, dtype=int)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (p >= lower) & (p < upper if upper < 1 else p <= upper)
        if selected.any():
            total += selected.mean() * abs(float(y[selected].mean()) - float(p[selected].mean()))
    return float(total)


def _binary_metrics_from_prediction(
    target: np.ndarray,
    probability: np.ndarray,
    predicted: np.ndarray,
    *,
    threshold: float | None,
    threshold_scope: str,
) -> dict[str, object]:
    y = np.asarray(target, dtype=int)
    p = np.asarray(probability, dtype=float)
    if p.ndim != 1 or len(p) != len(y) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid binary probability contract")
    predicted = np.asarray(predicted, dtype=int)
    if predicted.shape != y.shape or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Invalid binary prediction contract")
    return {
        "records": int(len(y)),
        "threshold": None if threshold is None else float(threshold),
        "threshold_scope": threshold_scope,
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "at_risk_precision": float(precision_score(y, predicted, zero_division=0)),
        "at_risk_recall": float(recall_score(y, predicted, zero_division=0)),
        "at_risk_f1": float(f1_score(y, predicted, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "nll": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def binary_metrics(target: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    p = np.asarray(probability, dtype=float)
    return _binary_metrics_from_prediction(
        target,
        p,
        (p >= threshold).astype(int),
        threshold=threshold,
        threshold_scope="global",
    )


def binary_metrics_per_record_threshold(
    target: np.ndarray,
    probability: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, object]:
    """Score pooled OOF rows with the threshold selected inside each outer fold."""

    p = np.asarray(probability, dtype=float)
    cutoffs = np.asarray(thresholds, dtype=float)
    if cutoffs.shape != p.shape or not np.isfinite(cutoffs).all() or (cutoffs < 0).any() or (cutoffs > 1).any():
        raise ValueError("Invalid per-record threshold contract")
    return _binary_metrics_from_prediction(
        target,
        p,
        (p >= cutoffs).astype(int),
        threshold=None,
        threshold_scope="per_outer_fold",
    )


__all__ = [
    "binary_metrics",
    "binary_metrics_per_record_threshold",
    "expected_calibration_error",
    "multiclass_metrics",
]
