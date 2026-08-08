"""Fail-closed routing after action relevance ranking."""

from __future__ import annotations

from .contracts import (
    ActionScore,
    RecommendationFeatures,
    RouteStatus,
    SafetyThresholds,
)


def route_ranked_actions(
    features: RecommendationFeatures,
    ranked_actions: tuple[ActionScore, ...],
    thresholds: SafetyThresholds,
) -> tuple[RouteStatus, tuple[str, ...]]:
    """Route to RECOMMEND only when evidence and ranking are sufficiently clear."""

    reasons: list[str] = []
    if not ranked_actions:
        return RouteStatus.NO_FEASIBLE_ACTION, ("NO_FEASIBLE_ACTION",)

    if features.hybrid_uncertainty > thresholds.maximum_hybrid_uncertainty:
        reasons.append("HYBRID_UNCERTAINTY_TOO_HIGH")
    if (
        features.seed_disagreement is not None
        and thresholds.maximum_seed_disagreement is not None
        and features.seed_disagreement > thresholds.maximum_seed_disagreement
    ):
        reasons.append("HYBRID_SEED_DISAGREEMENT_TOO_HIGH")
    if features.label_conflict > thresholds.maximum_label_conflict:
        reasons.append("WEAK_LABEL_CONFLICT_TOO_HIGH")
    if features.ood_score > thresholds.maximum_ood_score:
        reasons.append("OUT_OF_DISTRIBUTION_SCORE_TOO_HIGH")

    top1 = ranked_actions[0]
    top2_score = ranked_actions[1].score if len(ranked_actions) > 1 else 0.0
    margin = top1.score - top2_score
    insufficient_evidence: list[str] = []
    if top1.score < thresholds.minimum_top1_score:
        insufficient_evidence.append("TOP1_RELEVANCE_TOO_LOW")
    if len(ranked_actions) > 1 and margin < thresholds.minimum_top1_margin:
        reasons.append("TOP_ACTION_MARGIN_TOO_SMALL")

    if insufficient_evidence:
        return RouteStatus.INSUFFICIENT_EVIDENCE, tuple(insufficient_evidence)
    if reasons:
        return RouteStatus.HUMAN_REVIEW, tuple(reasons)
    return RouteStatus.RECOMMEND, ("SAFE_AUTOMATIC_RECOMMENDATION",)


__all__ = ["route_ranked_actions"]
