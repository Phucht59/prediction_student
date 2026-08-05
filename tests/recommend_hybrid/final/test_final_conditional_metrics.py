import numpy as np

from src.recommend_hybrid.final.metrics import ranking_metrics


def test_ranking_metrics_on_positive_groups():
    values = ranking_metrics(
        np.asarray([[3.0, 1.0, 0.0], [0.0, 4.0, 1.0]]),
        np.asarray([[1, 0, 0], [0, 1, 0]]),
        np.ones((2, 3), dtype=bool),
        np.asarray([1, 1]),
    )
    assert values["conditional_precision_at_1_all_positive"] == 1.0
    assert values["mrr"] == 1.0
    assert values["ndcg_at_3"] == 1.0
