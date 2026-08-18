"""Ranking and regression metrics for Phase 8/9 development and holdout."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    error = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    if sample_weight is None:
        return float(error.mean())
    weights = np.asarray(sample_weight, dtype=float)
    return float(np.average(error, weights=weights))


def rmse(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    squared = (np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2
    if sample_weight is None:
        return float(math.sqrt(squared.mean()))
    weights = np.asarray(sample_weight, dtype=float)
    return float(math.sqrt(np.average(squared, weights=weights)))


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 3:
        return None
    left = pd.Series(y_true).rank()
    right = pd.Series(y_pred).rank()
    if left.nunique() < 2 or right.nunique() < 2:
        return None
    value = float(left.corr(right))
    return None if not math.isfinite(value) else value


def clip_score(raw: np.ndarray | float) -> np.ndarray | float:
    return np.clip(raw, 0.0, 3.0)


def dcg_at_k(relevances: list[float], k: int = 3) -> float:
    values = np.asarray(list(relevances)[:k], dtype=float)
    if values.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, values.size + 2))
    return float(np.sum((np.power(2.0, values) - 1.0) / discounts))


def ndcg_at_k(reference_by_action: dict[str, float], ranked_actions: list[str], k: int = 3) -> float | None:
    if not reference_by_action:
        return None
    graded = [reference_by_action[action] for action in ranked_actions if action in reference_by_action][:k]
    ideal = sorted(reference_by_action.values(), reverse=True)[:k]
    idcg = dcg_at_k(ideal, k)
    if idcg <= 0:
        return None
    return dcg_at_k(graded, k) / idcg


def precision_at_1(reference_by_action: dict[str, float], ranked_actions: list[str], *, relevant_threshold: float = 2.0) -> float | None:
    if not ranked_actions or not reference_by_action:
        return None
    top = ranked_actions[0]
    if top not in reference_by_action:
        return None
    return float(reference_by_action[top] >= relevant_threshold)


def recall_at_k(reference_by_action: dict[str, float], ranked_actions: list[str], k: int = 3, *, relevant_threshold: float = 2.0) -> float | None:
    relevant = {action for action, value in reference_by_action.items() if value >= relevant_threshold}
    if not relevant:
        return None
    hit = relevant.intersection(ranked_actions[:k])
    return float(len(hit) / len(relevant))


def mrr(reference_by_action: dict[str, float], ranked_actions: list[str], *, relevant_threshold: float = 2.0) -> float | None:
    relevant = {action for action, value in reference_by_action.items() if value >= relevant_threshold}
    if not relevant:
        return None
    for index, action in enumerate(ranked_actions, start=1):
        if action in relevant:
            return 1.0 / index
    return 0.0


def pairwise_accuracy(reference_by_action: dict[str, float], score_by_action: dict[str, float]) -> float | None:
    actions = [action for action in reference_by_action if action in score_by_action]
    pairs = 0
    correct = 0
    for i, left in enumerate(actions):
        for right in actions[i + 1:]:
            delta_y = reference_by_action[left] - reference_by_action[right]
            if abs(delta_y) < 1e-12:
                continue
            pairs += 1
            delta_s = score_by_action[left] - score_by_action[right]
            if (delta_y > 0 and delta_s > 0) or (delta_y < 0 and delta_s < 0):
                correct += 1
    if pairs == 0:
        return None
    return correct / pairs


def mean_optional(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not clean else float(np.mean(clean))
