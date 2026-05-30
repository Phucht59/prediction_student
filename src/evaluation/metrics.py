from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)


CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["weak", "average", "good"]


def classification_metrics(y_true, y_pred, labels=None) -> dict[str, float]:
    metric_labels = labels if labels is not None else CLASS_LABELS
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=metric_labels, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=metric_labels, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=metric_labels, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(
                y_true,
                y_pred,
                labels=metric_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_true,
                y_pred,
                labels=metric_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                labels=metric_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "precision_weak": float(
            precision_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)
        ),
        "recall_weak": float(
            recall_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)
        ),
        "f1_weak": float(f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
    }


def classification_probability_metrics(y_true, y_prob, labels=None) -> dict[str, float]:
    """Precision-Recall metrics for classifiers that expose class probabilities."""
    metric_labels = labels if labels is not None else CLASS_LABELS
    y_true_array = np.asarray(y_true)
    y_prob_array = np.asarray(y_prob)
    if y_prob_array.ndim != 2 or y_prob_array.shape[1] != len(metric_labels):
        raise ValueError(
            "y_prob must have shape (n_samples, n_classes); "
            f"got {y_prob_array.shape}, expected n_classes={len(metric_labels)}."
        )

    per_class_scores: list[float] = []
    supports: list[int] = []
    for index, class_id in enumerate(metric_labels):
        binary_true = (y_true_array == class_id).astype(int)
        support = int(binary_true.sum())
        supports.append(support)
        if support == 0:
            per_class_scores.append(np.nan)
        else:
            per_class_scores.append(float(average_precision_score(binary_true, y_prob_array[:, index])))

    valid_scores = np.asarray([score for score in per_class_scores if not np.isnan(score)], dtype=float)
    support_array = np.asarray(supports, dtype=float)
    score_array = np.asarray([0.0 if np.isnan(score) else score for score in per_class_scores], dtype=float)
    total_support = float(support_array.sum())

    return {
        "pr_auc_macro": float(valid_scores.mean()) if valid_scores.size else 0.0,
        "pr_auc_weighted": float(np.average(score_array, weights=support_array)) if total_support > 0 else 0.0,
        "pr_auc_weak": float(score_array[0]) if len(score_array) else 0.0,
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_confusion_matrix_plot(y_true, y_pred, output_path, title) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    threshold = matrix.max() / 2 if matrix.size and matrix.max() else 0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            color = "white" if matrix[row_index, col_index] > threshold else "black"
            ax.text(
                col_index,
                row_index,
                str(matrix[row_index, col_index]),
                ha="center",
                va="center",
                color=color,
            )

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def classification_report_text(y_true, y_pred) -> str:
    return classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
