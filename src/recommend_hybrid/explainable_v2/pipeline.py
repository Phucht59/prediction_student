"""End-to-end orchestration for risk-guided explainable recommendation V2."""

from __future__ import annotations

from .contracts import (
    RecommendationDecision,
    RecommendationFeatures,
    RiskBand,
    RiskThresholds,
    RouteStatus,
    SafetyThresholds,
)
from .feasibility import feasible_actions
from .ranker import ActionRanker
from .risk_policy import stratify_risk
from .safety_router import route_ranked_actions


class ExplainableRecommendationPipeline:
    """Use frozen Hybrid risk as authority, then rank only feasible actions."""

    def __init__(
        self,
        ranker: ActionRanker,
        risk_thresholds: RiskThresholds,
        safety_thresholds: SafetyThresholds,
        *,
        top_k: int = 3,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.ranker = ranker
        self.risk_thresholds = risk_thresholds
        self.safety_thresholds = safety_thresholds
        self.top_k = top_k

    def recommend(self, features: RecommendationFeatures) -> RecommendationDecision:
        risk_band = stratify_risk(features, self.risk_thresholds)
        if risk_band is RiskBand.LOW:
            return RecommendationDecision(
                student_key=features.student_key,
                course_key=features.course_key,
                stage=features.stage,
                risk_band=risk_band,
                route=RouteStatus.NO_ACTION,
                ranked_actions=(),
                reason_codes=("FROZEN_HYBRID_LOW_RISK",),
            )
        if risk_band is RiskBand.BORDERLINE:
            return RecommendationDecision(
                student_key=features.student_key,
                course_key=features.course_key,
                stage=features.stage,
                risk_band=risk_band,
                route=RouteStatus.MONITOR,
                ranked_actions=(),
                reason_codes=("FROZEN_HYBRID_BORDERLINE_OR_UNCERTAIN",),
            )

        evaluations = feasible_actions(features)
        eligible = tuple(item.action for item in evaluations if item.eligible)
        if not eligible:
            reasons = tuple(
                reason
                for item in evaluations
                for reason in item.reason_codes
                if reason != "FEASIBLE"
            )
            return RecommendationDecision(
                student_key=features.student_key,
                course_key=features.course_key,
                stage=features.stage,
                risk_band=risk_band,
                route=RouteStatus.HUMAN_REVIEW,
                ranked_actions=(),
                reason_codes=reasons or ("NO_FEASIBLE_AUTOMATIC_ACTION",),
            )

        ranked = self.ranker.score(features, eligible)
        route, reasons = route_ranked_actions(features, ranked, self.safety_thresholds)
        selected = ranked[: self.top_k]
        return RecommendationDecision(
            student_key=features.student_key,
            course_key=features.course_key,
            stage=features.stage,
            risk_band=risk_band,
            route=route,
            ranked_actions=selected,
            reason_codes=reasons,
        )


__all__ = ["ExplainableRecommendationPipeline"]
