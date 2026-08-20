"""Full binary evaluation suite. Thresholds must come from STOP, never outer."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def expected_calibration_error(target: np.ndarray, score: np.ndarray, n_bins: int = 15) -> float:
    y = np.asarray(target, dtype=float)
    p = np.asarray(score, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        right = bins[i + 1] if i < n_bins - 1 else bins[i + 1] + 1e-12
        mask = (p >= bins[i]) & (p < right)
        if mask.any():
            ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    return float(ece)


def reliability_table(target: np.ndarray, score: np.ndarray, n_bins: int = 10) -> list[dict]:
    y = np.asarray(target, dtype=float)
    p = np.asarray(score, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        right = bins[i + 1] if i < n_bins - 1 else bins[i + 1] + 1e-12
        mask = (p >= bins[i]) & (p < right)
        if not mask.any():
            continue
        rows.append(
            {
                "bin": i,
                "left": float(bins[i]),
                "right": float(bins[i + 1]),
                "n": int(mask.sum()),
                "mean_p": float(p[mask].mean()),
                "mean_y": float(y[mask].mean()),
            }
        )
    return rows


def full_metrics(target: np.ndarray, score: np.ndarray, *, threshold: float) -> dict[str, float]:
    y = np.asarray(target, dtype=int)
    p = np.asarray(score, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    h2 = binary_entropy(p)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p),
        "h2_mean": float(h2.mean()),
        "h2_p50": float(np.median(h2)),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n": int(len(y)),
        "prevalence": float(y.mean()),
    }
