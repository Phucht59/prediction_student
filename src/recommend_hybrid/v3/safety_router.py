"""Runtime safety using only inference-available C0 + ranking signals."""

from __future__ import annotations

from .contracts import ActionScore, RecommendationFeatures, RouteStatus, SafetyThresholds


def route_ranked_actions(
    features: RecommendationFeatures,
    ranked_actions: tuple[ActionScore, ...],
    thresholds: SafetyThresholds,
) -> tuple[RouteStatus, tuple[str, ...]]:
    if not ranked_actions:
        return RouteStatus.NO_FEASIBLE_ACTION, ("NO_FEASIBLE_ACTION",)

    reasons: list[str] = []
    if features.uncertainty > thresholds.maximum_uncertainty:
        reasons.append("HYBRID_UNCERTAINTY_TOO_HIGH")
    top1 = ranked_actions[0]
    if top1.score < thresholds.minimum_top1_score:
        return RouteStatus.INSUFFICIENT_EVIDENCE, ("TOP1_RELEVANCE_TOO_LOW",)
    if len(ranked_actions) > 1:
        margin = top1.score - ranked_actions[1].score
        if margin < thresholds.minimum_top1_margin:
            reasons.append("TOP_ACTION_MARGIN_TOO_SMALL")
    if reasons:
        return RouteStatus.HUMAN_REVIEW, tuple(reasons)
    return RouteStatus.RECOMMEND, ("SAFE_AUTOMATIC_RECOMMENDATION",)
