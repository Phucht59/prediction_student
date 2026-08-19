"""V3 ranking metrics. Official evaluation matches runtime candidate filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from src.recommend_hybrid.final.metrics import (
    RankingMetrics,
    evaluate_grouped_ranking,
    grouped_bootstrap_difference,
)

__all__ = [
    "RankingMetrics",
    "evaluate_grouped_ranking",
    "evaluate_runtime_equivalent_ranking",
    "grouped_bootstrap_difference",
    "grouped_bootstrap_difference_runtime_equivalent",
    "per_query_ndcg_at_k",
]


def ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int) -> float:
    """NDCG@k. A single-candidate query is a trivial perfect ranking."""

    if relevance.size == 0:
        return 0.0
    if relevance.size == 1:
        return 1.0
    return float(ndcg_score(relevance.reshape(1, -1), scores.reshape(1, -1), k=k))


def per_query_ndcg_at_k(
    frame: pd.DataFrame,
    *,
    query_column: str = "query_id",
    action_column: str = "action_id",
    relevance_column: str = "relevance",
    score_column: str = "score",
    eligible_column: str | None = "eligible",
    k: int = 3,
    positive_threshold: float = 1.0,
    runtime_equivalent: bool = True,
) -> pd.Series:
    """Per-query NDCG@k. Non-positive or unrankable queries are NaN."""

    working = frame
    if runtime_equivalent and eligible_column is not None and eligible_column in frame.columns:
        working = frame.loc[frame[eligible_column].astype(bool)].copy()
    values: dict[str, float] = {}
    for query_id, query in working.groupby(query_column, sort=False):
        query = query.sort_values([score_column, action_column], ascending=[False, True])
        relevance = query[relevance_column].to_numpy(dtype=float)
        scores = query[score_column].to_numpy(dtype=float)
        if not np.any(relevance >= positive_threshold):
            values[str(query_id)] = float("nan")
            continue
        values[str(query_id)] = ndcg_at_k(relevance, scores, k)
    return pd.Series(values, name="ndcg_at_k")


def evaluate_runtime_equivalent_ranking(
    frame: pd.DataFrame,
    *,
    query_column: str = "query_id",
    action_column: str = "action_id",
    relevance_column: str = "relevance",
    score_column: str = "score",
    eligible_column: str = "eligible",
    k: int = 3,
    positive_threshold: float = 1.0,
) -> RankingMetrics:
    """Rank only feasible candidates, matching RecommendationV3Pipeline.

    Queries with zero eligible actions are not issued recommendations and do
    not enter ranking metrics. invalid_action_rate is therefore 0 when the
    eligible flag is the same predicate the runtime uses.
    """

    required = {query_column, action_column, relevance_column, score_column, eligible_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ranking frame is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("ranking frame is empty")
    if k <= 0:
        raise ValueError("k must be positive")

    eligible = frame.loc[frame[eligible_column].astype(bool)].copy()
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

    for _, query in eligible.groupby(query_column, sort=False):
        query = query.sort_values([score_column, action_column], ascending=[False, True])
        relevance = query[relevance_column].to_numpy(dtype=float)
        scores = query[score_column].to_numpy(dtype=float)
        positives = relevance >= positive_threshold
        if positives.any():
            positive_queries += 1
            ndcg_values.append(ndcg_at_k(relevance, scores, k))
            precision_values.append(float(positives[0]))
            positive_ranks = np.flatnonzero(positives)
            reciprocal_ranks.append(1.0 / float(positive_ranks[0] + 1))
            top_k_positive = int(positives[:k].sum())
            recall_values.append(top_k_positive / float(positives.sum()))

        for left in range(len(relevance)):
            for right in range(left + 1, len(relevance)):
                if relevance[left] == relevance[right] or scores[left] == scores[right]:
                    continue
                pairwise_total += 1
                expected = relevance[left] > relevance[right]
                predicted = scores[left] > scores[right]
                pairwise_correct += int(expected == predicted)

        top1 = query.iloc[0]
        top1_actions.add(str(top1[action_column]))
        issued += 1
        invalid += int(not bool(top1[eligible_column]))

    query_count = int(eligible[query_column].nunique()) if not eligible.empty else 0
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


def grouped_bootstrap_difference_runtime_equivalent(
    full: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    metric: str = "ndcg_at_3",
    query_column: str = "query_id",
    iterations: int = 2000,
    seed: int = 2026,
    k: int = 3,
    positive_threshold: float = 1.0,
) -> dict[str, float | int]:
    """Paired query-level bootstrap of full-minus-baseline NDCG@k.

    For NDCG@3 this equals grouped case-level bootstrap of the mean of
    per-query NDCG because official NDCG is that mean over positive queries.
    """

    if metric != "ndcg_at_3":
        raise ValueError("runtime-equivalent bootstrap is defined for ndcg_at_3 only")
    if iterations < 100:
        raise ValueError("bootstrap requires at least 100 iterations")

    full_ndcg = per_query_ndcg_at_k(
        full,
        query_column=query_column,
        k=k,
        positive_threshold=positive_threshold,
        runtime_equivalent=True,
    )
    baseline_ndcg = per_query_ndcg_at_k(
        baseline,
        query_column=query_column,
        k=k,
        positive_threshold=positive_threshold,
        runtime_equivalent=True,
    )
    aligned = pd.concat([full_ndcg.rename("full"), baseline_ndcg.rename("baseline")], axis=1)
    aligned = aligned.dropna()
    if len(aligned) < 2:
        raise ValueError("at least two positive queries are required for bootstrap")

    differences = (aligned["full"] - aligned["baseline"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    n = len(differences)
    for _ in range(iterations):
        draw = rng.choice(differences, size=n, replace=True)
        samples.append(float(draw.mean()))
    values = np.asarray(samples, dtype=float)
    return {
        "iterations": int(iterations),
        "seed": int(seed),
        "unit": "query",
        "n_positive_queries": int(n),
        "metric": metric,
        "mean_difference": float(values.mean()),
        "ci_low_95": float(np.quantile(values, 0.025)),
        "ci_high_95": float(np.quantile(values, 0.975)),
        "probability_difference_positive": float(np.mean(values > 0.0)),
    }
