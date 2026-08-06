from __future__ import annotations

import pandas as pd

from src.recommend_hybrid.explainable_v2.metrics import (
    evaluate_grouped_ranking,
    grouped_bootstrap_difference,
)


def ranking_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"query_id": "q1", "action_id": "A", "relevance": 3, "score": 0.9, "eligible": True},
            {"query_id": "q1", "action_id": "B", "relevance": 2, "score": 0.7, "eligible": True},
            {"query_id": "q1", "action_id": "C", "relevance": 0, "score": 0.1, "eligible": True},
            {"query_id": "q2", "action_id": "A", "relevance": 0, "score": 0.2, "eligible": True},
            {"query_id": "q2", "action_id": "B", "relevance": 3, "score": 0.8, "eligible": True},
            {"query_id": "q2", "action_id": "C", "relevance": 1, "score": 0.4, "eligible": True},
        ]
    )


def test_perfect_grouped_ranking() -> None:
    metrics = evaluate_grouped_ranking(ranking_frame())
    assert metrics.query_count == 2
    assert metrics.positive_query_count == 2
    assert metrics.ndcg_at_3 == 1.0
    assert metrics.precision_at_1 == 1.0
    assert metrics.mrr == 1.0
    assert metrics.invalid_action_rate == 0.0


def test_bad_baseline_is_worse_than_full() -> None:
    full = ranking_frame()
    baseline = full.copy()
    baseline["score"] = [0.1, 0.2, 0.9, 0.9, 0.2, 0.1]
    full_metrics = evaluate_grouped_ranking(full)
    baseline_metrics = evaluate_grouped_ranking(baseline)
    assert full_metrics.ndcg_at_3 > baseline_metrics.ndcg_at_3


def test_grouped_bootstrap_preserves_duplicate_draws() -> None:
    full = ranking_frame()
    baseline = full.copy()
    baseline["score"] = [0.1, 0.2, 0.9, 0.9, 0.2, 0.1]
    result = grouped_bootstrap_difference(
        full,
        baseline,
        iterations=100,
        seed=42,
    )
    assert result["mean_difference"] > 0.0
    assert result["probability_difference_positive"] > 0.5
