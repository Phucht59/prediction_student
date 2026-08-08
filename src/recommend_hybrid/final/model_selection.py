"""Protocol-locked final model selection from completed scientific evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class CandidateEvidence:
    name: str
    complexity_rank: int
    ndcg_at_3: float
    all_release_gates_pass: bool
    difference_from_best_ci_low: float
    difference_from_best_ci_high: float

    def __post_init__(self) -> None:
        if not self.name or self.complexity_rank < 0:
            raise ValueError("candidate name and non-negative complexity rank are required")
        values = (
            self.ndcg_at_3,
            self.difference_from_best_ci_low,
            self.difference_from_best_ci_high,
        )
        if any(not isfinite(value) for value in values):
            raise ValueError("candidate evidence values must be finite")
        if not 0.0 <= self.ndcg_at_3 <= 1.0:
            raise ValueError("NDCG@3 must be in [0, 1]")
        if self.difference_from_best_ci_low > self.difference_from_best_ci_high:
            raise ValueError("invalid paired confidence interval")

    @property
    def statistically_indistinguishable_from_best(self) -> bool:
        return self.difference_from_best_ci_low <= 0.0 <= self.difference_from_best_ci_high


def select_final_candidate(
    candidates: tuple[CandidateEvidence, ...],
) -> CandidateEvidence:
    """Choose the simplest gate-passing model not distinguishable from the best.

    `difference_from_best` must be computed as candidate minus best using paired
    grouped bootstrap on the same frozen test queries. The best candidate uses
    the exact interval [0, 0]. Test evidence must not influence earlier tuning.
    """

    eligible = tuple(item for item in candidates if item.all_release_gates_pass)
    if not eligible:
        raise RuntimeError("no candidate passed all recommendation release gates")

    best_score = max(item.ndcg_at_3 for item in eligible)
    best = tuple(item for item in eligible if item.ndcg_at_3 == best_score)
    if not any(
        item.difference_from_best_ci_low == 0.0
        and item.difference_from_best_ci_high == 0.0
        for item in best
    ):
        raise ValueError("at least one empirical best candidate must use CI [0, 0]")

    indistinguishable = tuple(
        item for item in eligible if item.statistically_indistinguishable_from_best
    )
    if not indistinguishable:
        raise RuntimeError("no gate-passing candidate is statistically comparable to best")

    return min(
        indistinguishable,
        key=lambda item: (item.complexity_rank, -item.ndcg_at_3, item.name),
    )


__all__ = ["CandidateEvidence", "select_final_candidate"]
