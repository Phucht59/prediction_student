from __future__ import annotations

import pandas as pd
import pytest

from src.recommend_hybrid.explainable_v2.audits import (
    assert_pre_cutoff_lineage,
    assert_ranker_schema,
    assert_student_disjoint_splits,
    permute_context_by_query,
)
from src.recommend_hybrid.explainable_v2.ranker import FEATURE_COLUMNS


def test_ranker_schema_rejects_label_conflict() -> None:
    with pytest.raises(ValueError, match="forbidden ranker features"):
        assert_ranker_schema([*FEATURE_COLUMNS, "label_conflict"])


def test_student_splits_must_be_disjoint() -> None:
    with pytest.raises(ValueError, match="student overlap"):
        assert_student_disjoint_splits(
            {"train": ["1", "2"], "validation": ["3"], "test": ["2", "4"]}
        )


def test_lineage_rejects_post_cutoff_behavior() -> None:
    lineage = pd.DataFrame(
        [
            {
                "feature_name": "active_day_rate",
                "observation_end_day": 40,
                "cutoff_day": 40,
            }
        ]
    )
    with pytest.raises(ValueError, match="post-cutoff"):
        assert_pre_cutoff_lineage(lineage)


def test_context_permutation_preserves_query_action_rows() -> None:
    rows = []
    for query_id, risk in (("q1", 0.8), ("q2", 0.6)):
        for action in ("A", "B"):
            row = {
                "query_id": query_id,
                "action_id": action,
                "stage": "EARLY_35",
                "risk_probability": risk,
                "hybrid_uncertainty": 0.1,
                "seed_disagreement": 0.02,
                "course_progress": 0.35,
                "assessment_progress": 0.4,
                "assessments_due": 1,
                "time_to_deadline_days": 7,
                "inactivity_streak": 3,
                "active_day_rate": 0.4,
                "recent_activity_trend": -0.2,
                "regularity_score": 0.5,
                "content_coverage": 0.6,
                "quiz_activity": 0.3,
                "relevance": 2,
            }
            rows.append(row)
    frame = pd.DataFrame(rows)
    permuted = permute_context_by_query(frame, seed=3)
    assert list(permuted[["query_id", "action_id"]].itertuples(index=False, name=None)) == list(
        frame[["query_id", "action_id"]].itertuples(index=False, name=None)
    )
    assert permuted.groupby("query_id")["risk_probability"].nunique().eq(1).all()
