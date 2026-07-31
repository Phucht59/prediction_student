"""Metric replay utilities shared by all canonical V3 model families."""

from __future__ import annotations

from typing import Any

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


def ece(target: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
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


def multiclass_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    probability = np.clip(probability, 1e-7, 1.0)
    probability /= probability.sum(axis=1, keepdims=True)
    predicted = probability.argmax(axis=1)
    macro = precision_recall_fscore_support(
        target, predicted, labels=np.arange(3), average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        target, predicted, labels=np.arange(3), average="weighted", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        target, predicted, labels=np.arange(3), zero_division=0
    )
    one_hot = np.eye(3, dtype=float)[target]
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
            roc_auc_score(target, probability, labels=np.arange(3), multi_class="ovr", average="macro")
        ),
        "nll": float(log_loss(target, probability, labels=np.arange(3))),
        "brier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
        "ece": ece(target, probability),
        "confusion_matrix": confusion_matrix(target, predicted, labels=np.arange(3)).tolist(),
        "per_class": [
            {
                "class_index": int(index),
                "class_name": ("Low", "Medium", "High")[index],
                "precision": float(per_class[0][index]),
                "recall": float(per_class[1][index]),
                "f1": float(per_class[2][index]),
                "support": int(per_class[3][index]),
            }
            for index in range(3)
        ],
    }


def binary_metrics(
    target: np.ndarray, probability: np.ndarray, threshold: float | np.ndarray
) -> dict[str, Any]:
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
    classes = precision_recall_fscore_support(
        target, predicted, labels=[0, 1], zero_division=0
    )
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
        "ece": ece(target, np.column_stack([1.0 - probability, probability])),
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
    hybrid_probability: np.ndarray,
    comparator_probability: np.ndarray,
    *,
    hybrid_threshold: float | np.ndarray | None,
    comparator_threshold: float | np.ndarray | None,
    multiclass: bool,
    replicates: int = 5000,
    seed: int = 7319,
) -> dict[str, float]:
    target = np.asarray(target, dtype=int)
    hybrid_probability = np.asarray(hybrid_probability)
    comparator_probability = np.asarray(comparator_probability)
    if multiclass:
        hybrid_prediction = hybrid_probability.argmax(axis=1)
        comparator_prediction = comparator_probability.argmax(axis=1)
    else:
        assert hybrid_threshold is not None and comparator_threshold is not None
        hybrid_prediction = hybrid_probability >= hybrid_threshold
        comparator_prediction = comparator_probability >= comparator_threshold
    point = float(
        f1_score(target, hybrid_prediction, average="macro", zero_division=0)
        - f1_score(target, comparator_prediction, average="macro", zero_division=0)
    )
    generator = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = generator.integers(0, len(target), size=len(target))
        values[index] = f1_score(
            target[sample], hybrid_prediction[sample], average="macro", zero_division=0
        ) - f1_score(
            target[sample], comparator_prediction[sample], average="macro", zero_division=0
        )
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "delta_macro_f1": point,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "replicates": int(replicates),
    }
