from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


LABELS = [0, 1, 2]
CLASS_NAMES = ["Low", "Medium", "High"]


def validate_probabilities(probabilities: np.ndarray) -> None:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Study B probabilities must have shape [n,3]")
    if not np.isfinite(values).all() or values.min() < -1e-8 or values.max() > 1 + 1e-8:
        raise ValueError("Invalid probability value")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probabilities do not sum to one")


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            value += mask.mean() * abs(float((predicted[mask] == y_true[mask]).mean()) - float(confidence[mask].mean()))
    return float(value)


def summary_metrics(frame: pd.DataFrame) -> tuple[dict[str, object], list[dict[str, object]]]:
    y_true = frame["true_label"].astype(int).to_numpy()
    y_pred = frame["predicted_label"].astype(int).to_numpy()
    probabilities_available = frame[["prob_low", "prob_medium", "prob_high"]].notna().all(axis=None)
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    row: dict[str, object] = {
        "records": len(frame),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "class_collapse": len(set(y_pred)) < len(set(y_true)),
        "probability_available": bool(probabilities_available),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }
    if probabilities_available:
        probabilities = frame[["prob_low", "prob_medium", "prob_high"]].to_numpy(float)
        validate_probabilities(probabilities)
        one_hot = np.eye(3)[y_true]
        row.update({
            "nll": log_loss(y_true, probabilities, labels=LABELS),
            "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
            "ece": expected_calibration_error(y_true, probabilities),
            "macro_pr_auc": average_precision_score(one_hot, probabilities, average="macro"),
        })
    else:
        row.update({"nll": np.nan, "brier": np.nan, "ece": np.nan, "macro_pr_auc": np.nan})
    class_rows = [
        {"class_name": name, "precision": p[index], "recall": r[index], "f1": f[index], "support": int(support[index])}
        for index, name in enumerate(CLASS_NAMES)
    ]
    return row, class_rows
