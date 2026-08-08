"""Grouped query-level ranking metrics for explainable recommendation V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


@dataclass(frozen=True)
class RankingMetrics:
    query_count: int
    positive_query_count: int
    ndcg_at_3: float
    precision_at_1: float
    mrr: float
    recall_at_3: float
    pairwise_accuracy: float
    invalid_action_rate: float
    unique_top1_actions: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _pairwise_accuracy(relevance: np.ndarray, scores: np.ndarray) -> tuple[int, int]:
    correct = 0
    total = 0
    for left in range(len(relevance)):
        for right in range(left + 1, len(relevance)):
            if relevance[left] == relevance[right] or scores[left] == scores[right]:
                continue
            total += 1
            expected = relevance[left] > relevance[right]
            predicted = scores[left] > scores[right]
            correct += int(expected == predicted)
    return correct, total


def evaluate_grouped_ranking(
    frame: pd.DataFrame,
    *,
    query_column: str = "query_id",
    action_column: str = "action_id",
    relevance_column: str = "relevance",
    score_column: str = "score",
    eligible_column: str | None = "eligible",
    k: int = 3,
    positive_threshold: float = 1.0,
) -> RankingMetrics:
    """Evaluate one row per query-action candidate without row-level leakage.

    NDCG is averaged over positive queries because an all-zero query has no
    ranking target. End-to-end recommendation coverage must be reported
    separately by the pipeline evaluator.
    """

    required = {query_column, action_column, relevance_column, score_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ranking frame is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("ranking frame is empty")
    if k <= 0:
        raise ValueError("k must be positive")

    ndcg_values: list[float] = []
    precision_values: list[float] = []
    reciprocal_ranks: list[float] = []
    recall_values: list[float] = []
    pairwise_correct = 0
    pairwise_total = 0
    invalid = 0
    issued = 0
    top1_actions: set[str] = set()
    positive_queries = 0

    for _, query in frame.groupby(query_column, sort=False):
        query = query.sort_values([score_column, action_column], ascending=[False, True])
        relevance = query[relevance_column].to_numpy(dtype=float)
        scores = query[score_column].to_numpy(dtype=float)
        positives = relevance >= positive_threshold
        if positives.any():
            positive_queries += 1
            ndcg_values.append(
                float(ndcg_score(relevance.reshape(1, -1), scores.reshape(1, -1), k=k))
            )
            precision_values.append(float(positives[0]))
            positive_ranks = np.flatnonzero(positives)
            reciprocal_ranks.append(1.0 / float(positive_ranks[0] + 1))
            top_k_positive = int(positives[:k].sum())
            recall_values.append(top_k_positive / float(positives.sum()))

        correct, total = _pairwise_accuracy(relevance, scores)
        pairwise_correct += correct
        pairwise_total += total

        top1 = query.iloc[0]
        top1_actions.add(str(top1[action_column]))
        issued += 1
        if eligible_column is not None and eligible_column in query.columns:
            invalid += int(not bool(top1[eligible_column]))

    query_count = int(frame[query_column].nunique())
    return RankingMetrics(
        query_count=query_count,
        positive_query_count=positive_queries,
        ndcg_at_3=float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        precision_at_1=float(np.mean(precision_values)) if precision_values else 0.0,
        mrr=float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        recall_at_3=float(np.mean(recall_values)) if recall_values else 0.0,
        pairwise_accuracy=(pairwise_correct / pairwise_total if pairwise_total else 0.0),
        invalid_action_rate=(invalid / issued if issued else 0.0),
        unique_top1_actions=len(top1_actions),
    )


def _bootstrap_sample(
    frame: pd.DataFrame,
    sampled_query_ids: np.ndarray,
    *,
    query_column: str,
) -> pd.DataFrame:
    """Copy sampled queries and assign unique bootstrap query identities.

    Query IDs can occur more than once in a with-replacement bootstrap sample.
    Renaming each draw prevents duplicated draws from being collapsed by the
    grouped metric evaluator.
    """

    parts: list[pd.DataFrame] = []
    for draw_index, query_id in enumerate(sampled_query_ids):
        part = frame.loc[frame[query_column].eq(query_id)].copy()
        part[query_column] = f"bootstrap-{draw_index}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def grouped_bootstrap_difference(
    full: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    metric: str = "ndcg_at_3",
    query_column: str = "query_id",
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Bootstrap a full-minus-baseline metric difference at query level."""

    full_queries = set(full[query_column].unique())
    baseline_queries = set(baseline[query_column].unique())
    if full_queries != baseline_queries:
        raise ValueError("full and baseline query sets must match exactly")
    query_ids = np.array(sorted(full_queries, key=str), dtype=object)
    if len(query_ids) < 2:
        raise ValueError("at least two queries are required for bootstrap")
    if iterations < 100:
        raise ValueError("bootstrap requires at least 100 iterations")

    valid_metrics = set(RankingMetrics.__dataclass_fields__)
    if metric not in valid_metrics:
        raise ValueError(f"unknown ranking metric: {metric}")

    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(query_ids, size=len(query_ids), replace=True)
        sampled_full = _bootstrap_sample(full, sampled, query_column=query_column)
        sampled_baseline = _bootstrap_sample(baseline, sampled, query_column=query_column)
        full_metric = getattr(
            evaluate_grouped_ranking(sampled_full, query_column=query_column), metric
        )
        baseline_metric = getattr(
            evaluate_grouped_ranking(sampled_baseline, query_column=query_column), metric
        )
        differences.append(float(full_metric - baseline_metric))

    values = np.asarray(differences, dtype=float)
    return {
        "iterations": iterations,
        "mean_difference": float(values.mean()),
        "ci_low_95": float(np.quantile(values, 0.025)),
        "ci_high_95": float(np.quantile(values, 0.975)),
        "probability_difference_positive": float(np.mean(values > 0.0)),
    }


__all__ = [
    "RankingMetrics",
    "evaluate_grouped_ranking",
    "grouped_bootstrap_difference",
]
