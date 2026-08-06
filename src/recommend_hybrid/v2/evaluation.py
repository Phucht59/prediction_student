"""Complete Recommendation V2 eligibility, ranking and simulation metrics."""
from __future__ import annotations

from typing import Iterable

import numpy as np
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


def _binary(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.int8).reshape(-1)
    if not len(result) or not np.isin(result, [0, 1]).all():
        raise ValueError(f"{name} must be a non-empty binary vector")
    return result


def _probability(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite and non-empty")
    if np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def expected_calibration_error(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    y = _binary(target, "target")
    p = _probability(probability, "probability")
    if len(y) != len(p):
        raise ValueError("target and probability must align")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (p >= lower) & (p < upper if index < bins - 1 else p <= upper)
        if not selected.any():
            continue
        total += float(selected.mean()) * abs(float(y[selected].mean()) - float(p[selected].mean()))
    return total


def binary_probability_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, object]:
    y = _binary(target, "target")
    p = _probability(probability, "probability")
    pred = _binary(prediction, "prediction")
    if not (len(y) == len(p) == len(pred)):
        raise ValueError("binary metric inputs must align")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    return {
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, p)) if np.any(y == 1) else None,
        "brier_score": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "specificity": specificity,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def risk_coverage_curve(
    target: np.ndarray,
    probability: np.ndarray,
    confidence: np.ndarray,
    *,
    points: Iterable[float] = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00),
) -> list[dict[str, float]]:
    y = _binary(target, "target")
    p = _probability(probability, "probability")
    c = _probability(confidence, "confidence")
    if not (len(y) == len(p) == len(c)):
        raise ValueError("risk-coverage inputs must align")
    prediction = (p >= 0.5).astype(np.int8)
    order = np.argsort(-c, kind="stable")
    rows: list[dict[str, float]] = []
    for requested in points:
        count = max(1, min(len(y), int(np.ceil(float(requested) * len(y)))))
        selected = order[:count]
        error = float(np.mean(prediction[selected] != y[selected]))
        rows.append(
            {
                "coverage": float(count / len(y)),
                "selective_risk": error,
                "selective_accuracy": 1.0 - error,
            }
        )
    return rows


def eligibility_metrics(
    *,
    target: np.ndarray,
    risk_probability: np.ndarray,
    decisions: np.ndarray,
    behaviour_value: str = "BEHAVIOURAL_ACTION",
    defer_value: str = "DEFER_TO_HUMAN",
) -> dict[str, object]:
    y = _binary(target, "target")
    p = _probability(risk_probability, "risk_probability")
    decision = np.asarray(decisions, dtype=str).reshape(-1)
    if not (len(y) == len(p) == len(decision)):
        raise ValueError("eligibility inputs must align")
    issued = decision == behaviour_value
    deferred = decision == defer_value
    pred = issued.astype(np.int8)
    base = binary_probability_metrics(y, p, pred)
    positives = int(y.sum())
    issued_count = int(issued.sum())
    false_issue = int(np.sum(issued & (y == 0)))
    missed = int(np.sum(~issued & ~deferred & (y == 1)))
    resolved = ~deferred
    selective_accuracy = (
        float(np.mean(pred[resolved] == y[resolved])) if resolved.any() else 0.0
    )
    confidence = np.abs(p - 0.5) * 2.0
    return {
        **base,
        "population": int(len(y)),
        "intervention_rate": float(issued.mean()),
        "defer_rate": float(deferred.mean()),
        "false_issue_rate": float(false_issue / max(issued_count, 1)),
        "missed_support_rate": float(missed / max(positives, 1)),
        "selective_accuracy": selective_accuracy,
        "risk_coverage_curve": risk_coverage_curve(y, p, confidence),
    }


def _dcg(labels: np.ndarray, k: int) -> float:
    selected = np.asarray(labels, dtype=np.float64)[:k]
    if not len(selected):
        return 0.0
    return float(np.sum((2.0**selected - 1.0) / np.log2(np.arange(2, len(selected) + 2))))


def ranking_metrics(
    scores: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    top_k: int = 3,
) -> dict[str, object]:
    score = np.asarray(scores, dtype=np.float64)
    label = np.asarray(target, dtype=np.int8)
    valid = np.asarray(mask, dtype=bool)
    if score.ndim != 2 or score.shape != label.shape or score.shape != valid.shape:
        raise ValueError("ranking arrays must be aligned [groups, actions]")
    positive_groups = (label * valid).sum(axis=1) > 0
    indices = np.flatnonzero(positive_groups)
    if not len(indices):
        return {
            "positive_groups": 0,
            "precision_at_1": 0.0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "ndcg_at_3": 0.0,
            "mrr": 0.0,
            "pairwise_accuracy": 0.0,
            "action_diversity": 0,
            "top_action_concentration": 1.0,
        }
    p1: list[float] = []
    r1: list[float] = []
    r3: list[float] = []
    ndcg: list[float] = []
    reciprocal: list[float] = []
    pairwise_correct = 0
    pairwise_total = 0
    top_actions: list[int] = []
    for row in indices:
        available = np.flatnonzero(valid[row])
        order = available[np.argsort(-score[row, available], kind="stable")]
        positives = set(np.flatnonzero((label[row] > 0) & valid[row]).tolist())
        top_actions.append(int(order[0]))
        p1.append(float(int(order[0]) in positives))
        r1.append(float(len(set(order[:1]).intersection(positives)) / len(positives)))
        r3.append(float(len(set(order[:top_k]).intersection(positives)) / len(positives)))
        ranked_labels = label[row, order]
        ideal = np.sort(label[row, available])[::-1]
        ideal_dcg = _dcg(ideal, top_k)
        ndcg.append(_dcg(ranked_labels, top_k) / ideal_dcg if ideal_dcg else 0.0)
        first = next((rank for rank, action in enumerate(order, 1) if int(action) in positives), None)
        reciprocal.append(1.0 / first if first is not None else 0.0)
        negatives = [int(action) for action in available if int(action) not in positives]
        for positive in positives:
            for negative in negatives:
                pairwise_correct += int(score[row, positive] > score[row, negative])
                pairwise_total += 1
    counts = np.bincount(np.asarray(top_actions), minlength=score.shape[1])
    return {
        "positive_groups": int(len(indices)),
        "precision_at_1": float(np.mean(p1)),
        "recall_at_1": float(np.mean(r1)),
        "recall_at_3": float(np.mean(r3)),
        "ndcg_at_3": float(np.mean(ndcg)),
        "mrr": float(np.mean(reciprocal)),
        "pairwise_accuracy": float(pairwise_correct / pairwise_total) if pairwise_total else 0.0,
        "action_diversity": int(np.sum(counts > 0)),
        "top_action_concentration": float(counts.max() / counts.sum()),
        "top_action_counts": counts.astype(int).tolist(),
    }


def simulation_metrics(
    baseline_risk: np.ndarray,
    simulated_risk: np.ndarray,
    *,
    threshold: float,
) -> dict[str, object]:
    baseline = _probability(baseline_risk, "baseline_risk")
    simulated = np.asarray(simulated_risk, dtype=np.float64)
    if simulated.ndim != 2 or simulated.shape[0] != len(baseline):
        raise ValueError("simulated risk must be [rows, strengths]")
    if not np.isfinite(simulated).all() or np.any((simulated < 0.0) | (simulated > 1.0)):
        raise ValueError("simulated risk must be finite probabilities")
    delta = baseline[:, None] - simulated
    monotonic = np.all(np.diff(simulated, axis=1) <= 1.0e-8, axis=1)
    final = simulated[:, -1]
    return {
        "rows": int(len(baseline)),
        "mean_risk_delta_by_strength": delta.mean(axis=0).tolist(),
        "median_risk_delta_by_strength": np.median(delta, axis=0).tolist(),
        "positive_reduction_fraction_by_strength": (delta > 0.0).mean(axis=0).tolist(),
        "threshold_crossing_fraction": float(
            np.mean((baseline >= threshold) & (final < threshold))
        ),
        "monotonic_strength_fraction": float(monotonic.mean()),
    }


__all__ = [
    "binary_probability_metrics",
    "eligibility_metrics",
    "expected_calibration_error",
    "ranking_metrics",
    "risk_coverage_curve",
    "simulation_metrics",
]
