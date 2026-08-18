"""Canonical probability-based metrics for frozen classification evidence.

Release and replay tools use this module so that classical and neural models
are scored by the same definitions without retraining either family.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


BOOTSTRAP_POLICY = "Reuse registered frozen bootstrap evidence; never resample during release packaging."


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    """Compute expected calibration error for frozen multiclass probabilities."""

    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            value += float(mask.mean()) * abs(
                float((predicted[mask] == y_true[mask]).mean())
                - float(confidence[mask].mean())
            )
    return value


def metrics_from_confusion(matrix: list[list[int]]) -> dict[str, float]:
    """Summarise a multiclass confusion matrix."""

    cm = np.asarray(matrix, dtype=float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    diagonal = np.diag(cm)
    precision = np.divide(diagonal, predicted, out=np.zeros_like(diagonal), where=predicted != 0)
    recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(diagonal),
        where=(precision + recall) != 0,
    )
    total = float(cm.sum())
    return {
        "accuracy": float(diagonal.sum() / total),
        "balanced_accuracy": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
    }


def top_k_metrics(
    labels: np.ndarray, probabilities: np.ndarray, record_ids: np.ndarray, fraction: float
) -> dict[str, float]:
    """Compute deterministic intervention-budget metrics."""

    import math

    budget = max(1, math.ceil(len(labels) * fraction))
    order = np.lexsort((record_ids.astype(str), -probabilities))[:budget]
    selected = labels[order].astype(float)
    precision = float(selected.mean())
    recall = float(selected.sum() / labels.sum()) if labels.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    discounts = 1.0 / np.log2(np.arange(2, budget + 2))
    dcg = float((selected * discounts).sum())
    ideal = float(discounts[: min(int(labels.sum()), budget)].sum())
    return {"precision": precision, "recall": recall, "f1": f1, "ndcg": dcg / ideal if ideal else 0.0}
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


def expected_calibration_error(
    target: np.ndarray, probability: np.ndarray, bins: int = 15
) -> float:
    """Return top-label ECE using the frozen project's 15-bin convention."""
    target = np.asarray(target)
    probability = np.asarray(probability)
    if probability.ndim == 1:
        confidence = np.maximum(probability, 1.0 - probability)
        predicted = (probability >= 0.5).astype(int)
    else:
        confidence = probability.max(axis=1)
        predicted = probability.argmax(axis=1)
    correct = predicted == target
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            value += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return float(value)


def evaluate_multiclass(
    target: np.ndarray,
    probability: np.ndarray,
    class_names: Sequence[str] = ("Low", "Medium", "High"),
) -> dict[str, Any]:
    """Evaluate a three-class probability matrix with the canonical metric set."""
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    probability = np.clip(probability, 1e-7, 1.0)
    probability /= probability.sum(axis=1, keepdims=True)
    labels = np.arange(len(class_names))
    predicted = probability.argmax(axis=1)
    macro = precision_recall_fscore_support(
        target, predicted, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        target, predicted, labels=labels, average="weighted", zero_division=0
    )
    per_class = precision_recall_fscore_support(target, predicted, labels=labels, zero_division=0)
    one_hot = np.eye(len(class_names), dtype=float)[target]
    return {
        "accuracy": float(accuracy_score(target, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "pr_auc": float(average_precision_score(one_hot, probability, average="macro")),
        "roc_auc": float(
            roc_auc_score(target, probability, labels=labels, multi_class="ovr", average="macro")
        ),
        "nll": float(log_loss(target, probability, labels=labels)),
        "brier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
        "ece": expected_calibration_error(target, probability),
        "confusion_matrix": confusion_matrix(target, predicted, labels=labels).tolist(),
        "per_class": [
            {
                "class_index": int(index),
                "class_name": name,
                "precision": float(per_class[0][index]),
                "recall": float(per_class[1][index]),
                "f1": float(per_class[2][index]),
                "support": int(per_class[3][index]),
            }
            for index, name in enumerate(class_names)
        ],
    }


def evaluate_binary_risk(
    target: np.ndarray, probability: np.ndarray, threshold: float | np.ndarray
) -> dict[str, Any]:
    """Evaluate binary risk probabilities at an already frozen threshold."""
    target = np.asarray(target, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    threshold_values = np.asarray(threshold, dtype=float)
    predicted = (probability >= threshold_values).astype(int)
    macro = precision_recall_fscore_support(
        target, predicted, labels=[0, 1], average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        target, predicted, labels=[0, 1], average="weighted", zero_division=0
    )
    classes = precision_recall_fscore_support(target, predicted, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(target, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "pr_auc": float(average_precision_score(target, probability)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "nll": float(log_loss(target, probability, labels=[0, 1])),
        "brier": float(np.mean((probability - target) ** 2)),
        "ece": expected_calibration_error(target, np.column_stack([1.0 - probability, probability])),
        "threshold": float(threshold_values) if threshold_values.ndim == 0 else None,
        "risk_precision": float(classes[0][1]),
        "risk_recall": float(classes[1][1]),
        "risk_f1": float(classes[2][1]),
        "not_risk_precision": float(classes[0][0]),
        "not_risk_recall": float(classes[1][0]),
        "not_risk_f1": float(classes[2][0]),
        "specificity": float(tn / max(tn + fp, 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def paired_bootstrap_macro_f1(
    target: np.ndarray,
    left_probability: np.ndarray,
    right_probability: np.ndarray,
    *,
    left_threshold: float | np.ndarray | None,
    right_threshold: float | np.ndarray | None,
    multiclass: bool,
    replicates: int = 5000,
    seed: int = 7319,
) -> dict[str, float]:
    """Calculate a paired bootstrap interval from aligned frozen predictions."""
    target = np.asarray(target, dtype=int)
    left_probability = np.asarray(left_probability)
    right_probability = np.asarray(right_probability)
    if multiclass:
        left_prediction = left_probability.argmax(axis=1)
        right_prediction = right_probability.argmax(axis=1)
    else:
        assert left_threshold is not None and right_threshold is not None
        left_prediction = left_probability >= left_threshold
        right_prediction = right_probability >= right_threshold
    point = float(
        f1_score(target, left_prediction, average="macro", zero_division=0)
        - f1_score(target, right_prediction, average="macro", zero_division=0)
    )
    generator = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = generator.integers(0, len(target), size=len(target))
        values[index] = f1_score(
            target[sample], left_prediction[sample], average="macro", zero_division=0
        ) - f1_score(target[sample], right_prediction[sample], average="macro", zero_division=0)
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "delta_macro_f1": point,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "replicates": int(replicates),
    }
