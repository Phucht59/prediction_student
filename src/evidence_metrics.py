"""Metrics and evidence helpers for the final thesis experiments.

The functions in this module are intentionally model-agnostic.  Every metric
is derived from saved labels, predictions and probabilities so an evidence
bundle can be independently checked without loading a model checkpoint.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


CLASS_LABELS = (0, 1, 2)
CLASS_NAMES = ("Low", "Medium", "High")


def _as_probabilities(probabilities: np.ndarray, n_rows: int) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (n_rows, len(CLASS_LABELS)):
        raise ValueError(
            "Probabilities must have shape "
            f"({n_rows}, {len(CLASS_LABELS)}); received {values.shape}."
        )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Probabilities must be finite and non-negative.")
    row_sums = values.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("Every probability row must have positive mass.")
    return values / row_sums


def expected_calibration_error(
    y_true: Iterable[int], probabilities: np.ndarray, *, n_bins: int = 10
) -> float:
    """Return confidence-based multiclass ECE using equal-width bins."""
    labels = np.asarray(list(y_true), dtype=int)
    probs = _as_probabilities(probabilities, len(labels))
    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == n_bins - 1 else confidence < upper
        )
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def reliability_rows(
    y_true: Iterable[int], probabilities: np.ndarray, *, n_bins: int = 10
) -> list[dict[str, float | int]]:
    labels = np.asarray(list(y_true), dtype=int)
    probs = _as_probabilities(probabilities, len(labels))
    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == n_bins - 1 else confidence < upper
        )
        rows.append(
            {
                "bin": index,
                "lower": float(lower),
                "upper": float(upper),
                "count": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()) if mask.any() else 0.0,
                "accuracy": float(correct[mask].mean()) if mask.any() else 0.0,
            }
        )
    return rows


def classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    probabilities: np.ndarray | None = None,
) -> dict[str, object]:
    """Compute nominal, ordinal and probability metrics for three classes."""
    labels = np.asarray(list(y_true), dtype=int)
    predictions = np.asarray(list(y_pred), dtype=int)
    if labels.shape != predictions.shape:
        raise ValueError("Labels and predictions must have the same shape.")
    if labels.ndim != 1:
        raise ValueError("Labels and predictions must be one-dimensional.")

    precision, recall, per_f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(CLASS_LABELS),
        zero_division=0,
    )
    distance = np.abs(labels - predictions)
    result: dict[str, object] = {
        "n_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(labels, predictions, labels=list(CLASS_LABELS), weights="quadratic")
        ),
        "ordinal_mae": float(distance.mean()),
        "one_step_errors": int((distance == 1).sum()),
        "two_step_errors": int((distance == 2).sum()),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(CLASS_LABELS)).tolist(),
        "per_class": {
            name: {
                "label": int(label),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_f1[index]),
                "support": int(support[index]),
            }
            for index, (label, name) in enumerate(zip(CLASS_LABELS, CLASS_NAMES))
        },
    }

    if probabilities is not None:
        probs = _as_probabilities(probabilities, len(labels))
        one_hot = np.eye(len(CLASS_LABELS), dtype=float)[labels]
        pr_auc = {
            name: float(average_precision_score(one_hot[:, index], probs[:, index]))
            for index, name in enumerate(CLASS_NAMES)
        }
        result.update(
            {
                "pr_auc_ovr": pr_auc,
                "pr_auc_macro": float(np.mean(list(pr_auc.values()))),
                "multiclass_brier_score": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
                "ece": expected_calibration_error(labels, probs),
            }
        )
    return result


def bootstrap_confidence_intervals(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Stratified-by-observation bootstrap CIs for accuracy and Macro-F1."""
    labels = np.asarray(list(y_true), dtype=int)
    predictions = np.asarray(list(y_pred), dtype=int)
    if len(labels) == 0:
        raise ValueError("Cannot bootstrap an empty result set.")
    rng = np.random.default_rng(seed)
    accuracy_values = np.empty(n_resamples, dtype=float)
    macro_f1_values = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample = rng.integers(0, len(labels), size=len(labels))
        accuracy_values[index] = accuracy_score(labels[sample], predictions[sample])
        macro_f1_values[index] = f1_score(
            labels[sample], predictions[sample], average="macro", zero_division=0
        )

    def interval(values: np.ndarray, estimate: float) -> dict[str, float]:
        lower, upper = np.quantile(values, [0.025, 0.975])
        return {"estimate": float(estimate), "lower_95": float(lower), "upper_95": float(upper)}

    return {
        "accuracy": interval(accuracy_values, accuracy_score(labels, predictions)),
        "macro_f1": interval(
            macro_f1_values,
            f1_score(labels, predictions, average="macro", zero_division=0),
        ),
    }
