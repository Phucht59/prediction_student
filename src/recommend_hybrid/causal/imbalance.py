"""Imbalance sensitivity study on frozen Hybrid CNN-BiLSTM embeddings.

SMOTE and ADASYN are applied only to training embeddings. Validation and test
rows are never resampled and are used only for threshold selection and final
measurement respectively. This produces reporting evidence without replacing
the canonical frozen prediction checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

IMBALANCE_MODES = ("none", "class_weight", "smote", "adasyn")


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite two-dimensional matrix")
    return result


def _binary(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.int8).reshape(-1)
    if not len(result) or not np.isin(result, [0, 1]).all():
        raise ValueError(f"{name} must be a non-empty binary vector")
    return result


def _validate_split(features: np.ndarray, target: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    x = _matrix(features, f"{name}_features")
    y = _binary(target, f"{name}_target")
    if len(x) != len(y):
        raise ValueError(f"{name} features and target must align")
    return x, y


def select_validation_threshold(target: np.ndarray, probability: np.ndarray) -> float:
    """Select the F1-optimal threshold on validation only, deterministically."""

    y = _binary(target, "validation_target")
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    if len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("validation probabilities must be finite and aligned")
    candidates = np.unique(np.concatenate(([0.5], p)))
    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        prediction = (p >= threshold).astype(np.int8)
        f1 = f1_score(y, prediction, zero_division=0)
        balanced = balanced_accuracy_score(y, prediction)
        proximity = -abs(float(threshold) - 0.5)
        key = (float(f1), float(balanced), proximity)
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _metric_payload(target: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, object]:
    y = _binary(target, "test_target")
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    prediction = (p >= float(threshold)).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    roc_auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    average_precision = (
        float(average_precision_score(y, p)) if np.any(y == 1) else float("nan")
    )
    return {
        "roc_auc": roc_auc,
        "pr_auc": average_precision,
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "specificity": specificity,
        "brier_score": float(brier_score_loss(y, p)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


@dataclass(frozen=True)
class ImbalanceStudyResult:
    mode: str
    threshold: float
    original_train_count: int
    fitted_train_count: int
    original_class_counts: tuple[int, int]
    fitted_class_counts: tuple[int, int]
    metrics: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "threshold": self.threshold,
            "original_train_count": self.original_train_count,
            "fitted_train_count": self.fitted_train_count,
            "synthetic_train_count": self.fitted_train_count - self.original_train_count,
            "original_class_counts": list(self.original_class_counts),
            "fitted_class_counts": list(self.fitted_class_counts),
            "metrics": dict(self.metrics),
        }


def _resample_train(
    mode: str,
    features: np.ndarray,
    target: np.ndarray,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    if mode in {"none", "class_weight"}:
        return features, target, None
    counts = np.bincount(target, minlength=2)
    minority = int(np.min(counts))
    if minority < 2:
        return features, target, "INSUFFICIENT_MINORITY_FOR_SYNTHETIC_SAMPLING"
    neighbours = min(5, minority - 1)
    sampler = (
        SMOTE(random_state=random_state, k_neighbors=neighbours)
        if mode == "smote"
        else ADASYN(random_state=random_state, n_neighbors=neighbours)
    )
    try:
        sampled_features, sampled_target = sampler.fit_resample(features, target)
    except ValueError:
        return features, target, "SYNTHETIC_SAMPLING_FAILED"
    return (
        np.asarray(sampled_features, dtype=np.float64),
        np.asarray(sampled_target, dtype=np.int8),
        None,
    )


def run_frozen_embedding_imbalance_study(
    *,
    train_features: np.ndarray,
    train_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
    test_features: np.ndarray,
    test_target: np.ndarray,
    random_state: int = 20260806,
) -> dict[str, object]:
    """Run all preregistered imbalance modes with one identical linear head."""

    x_train, y_train = _validate_split(train_features, train_target, "train")
    x_validation, y_validation = _validate_split(
        validation_features, validation_target, "validation"
    )
    x_test, y_test = _validate_split(test_features, test_target, "test")
    if x_train.shape[1] != x_validation.shape[1] or x_train.shape[1] != x_test.shape[1]:
        raise ValueError("all embedding splits must have the same feature dimension")
    if len(np.unique(y_train)) < 2:
        raise ValueError("train target must contain both classes")

    original_counts = tuple(int(value) for value in np.bincount(y_train, minlength=2))
    results: list[dict[str, object]] = []
    for mode in IMBALANCE_MODES:
        sampled_x, sampled_y, warning = _resample_train(
            mode,
            x_train,
            y_train,
            random_state=random_state,
        )
        class_weight = "balanced" if mode == "class_weight" else None
        head = LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            class_weight=class_weight,
            random_state=random_state,
        )
        head.fit(sampled_x, sampled_y)
        validation_probability = head.predict_proba(x_validation)[:, 1]
        threshold = select_validation_threshold(y_validation, validation_probability)
        test_probability = head.predict_proba(x_test)[:, 1]
        fitted_counts = tuple(
            int(value) for value in np.bincount(sampled_y, minlength=2)
        )
        result = ImbalanceStudyResult(
            mode=mode,
            threshold=threshold,
            original_train_count=len(y_train),
            fitted_train_count=len(sampled_y),
            original_class_counts=original_counts,
            fitted_class_counts=fitted_counts,
            metrics=_metric_payload(y_test, test_probability, threshold),
        ).to_dict()
        result["warning"] = warning
        result["resampling_scope"] = "TRAIN_EMBEDDINGS_ONLY"
        result["canonical_checkpoint_replaced"] = False
        results.append(result)

    return {
        "study_id": "frozen_hybrid_embedding_imbalance_sensitivity",
        "modes": list(IMBALANCE_MODES),
        "random_state": int(random_state),
        "embedding_dimension": int(x_train.shape[1]),
        "validation_used_for": "THRESHOLD_SELECTION_ONLY",
        "test_used_for": "FINAL_METRICS_ONLY",
        "results": results,
    }


__all__ = [
    "IMBALANCE_MODES",
    "ImbalanceStudyResult",
    "run_frozen_embedding_imbalance_study",
    "select_validation_threshold",
]
