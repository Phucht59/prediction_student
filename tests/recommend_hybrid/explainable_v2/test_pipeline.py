from __future__ import annotations

from src.recommend_hybrid.contracts import Stage
from src.recommend_hybrid.explainable_v2 import (
    CanonicalAction,
    ExplainableRecommendationPipeline,
    FixedActionRanker,
    RecommendationFeatures,
    RiskBand,
    RiskThresholds,
    RouteStatus,
    SafetyThresholds,
)


def thresholds() -> tuple[RiskThresholds, SafetyThresholds]:
    return (
        RiskThresholds(
            low=0.35,
            high=0.65,
            maximum_automatic_uncertainty=0.40,
            maximum_seed_disagreement=0.10,
        ),
        SafetyThresholds(
            minimum_top1_score=0.60,
            minimum_top1_margin=0.10,
            maximum_hybrid_uncertainty=0.40,
            maximum_seed_disagreement=0.10,
            maximum_label_conflict=0.30,
            maximum_ood_score=0.95,
        ),
    )


def features(**overrides) -> RecommendationFeatures:
    payload = {
        "student_key": "student-1",
        "course_key": "AAA:2014J",
        "stage": Stage.EARLY_35,
        "cutoff_day": 40,
        "risk_probability": 0.80,
        "hybrid_uncertainty": 0.10,
        "seed_disagreement": 0.02,
        "course_progress": 0.35,
        "assessment_progress": 0.30,
        "assessments_due": 2,
        "missing_assessment_count": 1,
        "due_soon_count": 0,
        "assessment_window_open": True,
        "time_to_deadline_days": 10,
        "inactivity_streak": 8,
        "active_day_rate": 0.20,
        "recent_activity_trend": -0.40,
        "regularity_score": 0.25,
        "content_coverage": 0.40,
        "knowledge_gap_evidence": True,
        "quiz_activity": 0.10,
        "quiz_available": True,
        "vle_access_available": True,
        "study_material_available": True,
        "label_conflict": 0.10,
        "ood_score": 0.20,
    }
    payload.update(overrides)
    return RecommendationFeatures(**payload)


def pipeline(scores: dict[CanonicalAction, float]) -> ExplainableRecommendationPipeline:
    risk, safety = thresholds()
    return ExplainableRecommendationPipeline(
        FixedActionRanker(scores),
        risk,
        safety,
        top_k=3,
    )


def complete_scores() -> dict[CanonicalAction, float]:
    return {
        CanonicalAction.ASSESSMENT_COMPLETION: 0.90,
        CanonicalAction.RECOVER_ENGAGEMENT: 0.70,
        CanonicalAction.STUDY_REGULARITY: 0.55,
        CanonicalAction.TARGETED_CONTENT_REVIEW: 0.40,
        CanonicalAction.QUIZ_RETRIEVAL_PRACTICE: 0.30,
    }


def test_public_route_status_contract_is_exact() -> None:
    assert {status.value for status in RouteStatus} == {
        "RECOMMEND",
        "INSUFFICIENT_EVIDENCE",
        "HUMAN_REVIEW",
        "NO_FEASIBLE_ACTION",
    }


def test_low_risk_routes_to_insufficient_evidence() -> None:
    decision = pipeline(complete_scores()).recommend(features(risk_probability=0.20))
    assert decision.risk_band is RiskBand.LOW
    assert decision.route is RouteStatus.INSUFFICIENT_EVIDENCE
    assert decision.ranked_actions == ()


def test_uncertain_high_probability_routes_to_human_review() -> None:
    decision = pipeline(complete_scores()).recommend(
        features(risk_probability=0.85, hybrid_uncertainty=0.60)
    )
    assert decision.risk_band is RiskBand.BORDERLINE
    assert decision.route is RouteStatus.HUMAN_REVIEW


def test_high_risk_with_clear_scores_recommends_top_three() -> None:
    decision = pipeline(complete_scores()).recommend(features())
    assert decision.risk_band is RiskBand.HIGH
    assert decision.route is RouteStatus.RECOMMEND
    assert [item.action for item in decision.ranked_actions] == [
        CanonicalAction.ASSESSMENT_COMPLETION,
        CanonicalAction.RECOVER_ENGAGEMENT,
        CanonicalAction.STUDY_REGULARITY,
    ]


def test_small_top_action_margin_fails_closed() -> None:
    scores = complete_scores()
    scores[CanonicalAction.RECOVER_ENGAGEMENT] = 0.85
    decision = pipeline(scores).recommend(features())
    assert decision.route is RouteStatus.HUMAN_REVIEW
    assert "TOP_ACTION_MARGIN_TOO_SMALL" in decision.reason_codes


def test_high_risk_without_any_feasible_action_uses_exact_status() -> None:
    decision = pipeline(complete_scores()).recommend(
        features(
            assessments_due=0,
            missing_assessment_count=0,
            due_soon_count=0,
            assessment_window_open=False,
            time_to_deadline_days=0,
            vle_access_available=False,
            regularity_score=None,
            active_day_rate=None,
            inactivity_streak=None,
            content_coverage=None,
            knowledge_gap_evidence=False,
            study_material_available=False,
            quiz_available=False,
            quiz_activity=None,
        )
    )
    assert decision.route is RouteStatus.NO_FEASIBLE_ACTION
    assert decision.ranked_actions == ()


def test_unavailable_seed_disagreement_does_not_trigger_threshold() -> None:
    decision = pipeline(complete_scores()).recommend(
        features(seed_disagreement=None)
    )
    assert decision.route is RouteStatus.RECOMMEND


def test_unavailable_seed_threshold_is_a_valid_frozen_contract() -> None:
    risk, safety = thresholds()
    safety = SafetyThresholds(
        **{**safety.__dict__, "maximum_seed_disagreement": None}
    )
    decision = ExplainableRecommendationPipeline(
        FixedActionRanker(complete_scores()), risk, safety
    ).recommend(features(seed_disagreement=None))
    assert decision.route is RouteStatus.RECOMMEND


def test_high_risk_low_relevance_is_insufficient_evidence() -> None:
    decision = pipeline(
        {action: 0.2 for action in CanonicalAction}
    ).recommend(features())
    assert decision.route is RouteStatus.INSUFFICIENT_EVIDENCE
    assert decision.ranked_actions == ()


def test_contraindicated_action_is_never_ranked() -> None:
    decision = pipeline(complete_scores()).recommend(
        features(contraindications=frozenset({"EXTENSION_PENDING"}))
    )
    assert all(
        item.action is not CanonicalAction.ASSESSMENT_COMPLETION
        for item in decision.ranked_actions
    )
