"""End-to-end V3 pipeline: C0 risk → feasibility → rank → route → plan."""

from __future__ import annotations

from typing import Protocol

from .contracts import (
    ActionScore,
    CanonicalAction,
    RecommendationDecision,
    RecommendationFeatures,
    RiskRoute,
    RiskThresholds,
    RouteStatus,
    SafetyThresholds,
)
from .feasibility import feasible_actions
from .plan_builder import build_personalized_plan
from .risk_router import stratify_risk
from .safety_router import route_ranked_actions


class ActionRanker(Protocol):
    def score(
        self,
        features: RecommendationFeatures,
        eligible_actions: tuple[CanonicalAction, ...],
    ) -> tuple[ActionScore, ...]: ...


class RecommendationV3Pipeline:
    def __init__(
        self,
        ranker: ActionRanker,
        risk_thresholds: RiskThresholds,
        safety_thresholds: SafetyThresholds,
        *,
        review_k: int = 3,
    ) -> None:
        if review_k <= 0:
            raise ValueError("review_k must be positive")
        self.ranker = ranker
        self.risk_thresholds = risk_thresholds
        self.safety_thresholds = safety_thresholds
        self.review_k = review_k

    def recommend(self, features: RecommendationFeatures) -> RecommendationDecision:
        risk_route = stratify_risk(features, self.risk_thresholds)
        evaluations = feasible_actions(features)
        eligible = tuple(item.action for item in evaluations if item.eligible)
        if not eligible:
            reasons = tuple(
                reason
                for item in evaluations
                for reason in item.reason_codes
                if not reason.startswith("ELIGIBLE")
            )
            route = (
                RouteStatus.INSUFFICIENT_EVIDENCE
                if risk_route is RiskRoute.NO_AUTOMATIC
                else RouteStatus.NO_FEASIBLE_ACTION
            )
            return RecommendationDecision(
                student_key=features.student_key,
                course_key=features.course_key,
                stage=features.stage,
                risk_route=risk_route,
                route=route,
                ranked_actions=(),
                plan=None,
                reason_codes=reasons or ("NO_FEASIBLE_AUTOMATIC_ACTION",),
            )

        ranked = self.ranker.score(features, eligible)
        if risk_route is RiskRoute.NO_AUTOMATIC:
            return RecommendationDecision(
                student_key=features.student_key,
                course_key=features.course_key,
                stage=features.stage,
                risk_route=risk_route,
                route=RouteStatus.INSUFFICIENT_EVIDENCE,
                ranked_actions=(),
                plan=None,
                reason_codes=("C0_BELOW_OPERATING_THRESHOLD",),
            )
        if risk_route is RiskRoute.HUMAN_REVIEW:
            selected = ranked[: self.review_k]
            return RecommendationDecision(
                student_key=features.student_key,
                course_key=features.course_key,
                stage=features.stage,
                risk_route=risk_route,
                route=RouteStatus.HUMAN_REVIEW,
                ranked_actions=selected,
                plan=build_personalized_plan(selected[0], features),
                reason_codes=("C0_MARGIN_OR_UNCERTAINTY_REQUIRES_REVIEW",),
            )

        route, reasons = route_ranked_actions(features, ranked, self.safety_thresholds)
        if route is RouteStatus.RECOMMEND:
            selected = ranked[:1]
            plan = build_personalized_plan(selected[0], features)
        elif route is RouteStatus.HUMAN_REVIEW:
            selected = ranked[: self.review_k]
            plan = build_personalized_plan(selected[0], features)
        else:
            selected = ()
            plan = None
        return RecommendationDecision(
            student_key=features.student_key,
            course_key=features.course_key,
            stage=features.stage,
            risk_route=risk_route,
            route=route,
            ranked_actions=selected,
            plan=plan,
            reason_codes=reasons,
        )
