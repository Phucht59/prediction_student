from __future__ import annotations

import numpy as np
import pytest

from src.recommend_hybrid.explainable_v2.weak_labels import (
    ABSTAIN,
    WeakLabelSource,
    source_correlation_audit,
    validate_vote_matrix,
)


def sources() -> tuple[WeakLabelSource, ...]:
    return (
        WeakLabelSource("literature_regular", "literature"),
        WeakLabelSource("behavior_regular", "behavior"),
        WeakLabelSource("llm_panel_a", "llm"),
    )


def test_valid_ordinal_votes() -> None:
    votes = np.array(
        [
            [3, 2, ABSTAIN],
            [0, 1, 0],
            [ABSTAIN, 2, 2],
        ]
    )
    result = validate_vote_matrix(votes, sources())
    assert result.shape == (3, 3)


def test_invalid_vote_is_rejected() -> None:
    votes = np.array([[4, 0, 1]])
    with pytest.raises(ValueError, match="ordinal values"):
        validate_vote_matrix(votes, sources())


def test_source_correlation_audit_is_pairwise() -> None:
    votes = np.array(
        [
            [3, 3, ABSTAIN],
            [0, 0, 1],
            [2, 1, 2],
        ]
    )
    audit = source_correlation_audit(votes, sources())
    assert len(audit) == 3
    assert set(audit.columns) == {
        "left_source",
        "left_family",
        "right_source",
        "right_family",
        "joint_vote_count",
        "exact_agreement",
    }
