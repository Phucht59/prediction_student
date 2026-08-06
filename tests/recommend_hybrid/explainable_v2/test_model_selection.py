from __future__ import annotations

import pytest

from src.recommend_hybrid.explainable_v2.model_selection import (
    CandidateEvidence,
    select_final_candidate,
)


def test_selects_simplest_statistically_indistinguishable_candidate() -> None:
    candidates = (
        CandidateEvidence("five_ebm", 2, 0.81, True, -0.015, 0.006),
        CandidateEvidence("lambdamart", 3, 0.82, True, 0.0, 0.0),
        CandidateEvidence("linear", 1, 0.77, True, -0.060, -0.020),
    )
    selected = select_final_candidate(candidates)
    assert selected.name == "five_ebm"


def test_failed_release_gate_excludes_candidate() -> None:
    candidates = (
        CandidateEvidence("five_ebm", 2, 0.81, True, 0.0, 0.0),
        CandidateEvidence("shortcut_model", 1, 0.90, False, -0.01, 0.01),
    )
    assert select_final_candidate(candidates).name == "five_ebm"


def test_no_passing_candidate_blocks_release() -> None:
    candidates = (
        CandidateEvidence("five_ebm", 2, 0.81, False, 0.0, 0.0),
    )
    with pytest.raises(RuntimeError, match="no candidate passed"):
        select_final_candidate(candidates)
