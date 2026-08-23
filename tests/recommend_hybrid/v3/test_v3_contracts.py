"""V3 contract and pipeline tests. No training, no Gemini, no Panel B."""

from __future__ import annotations

from src.prediction.contracts import PredictionResult
from src.recommend_hybrid.v3.contracts import (
    ActionScore,
    CanonicalAction,
    RecommendationFeatures,
    RiskThresholds,
    RouteStatus,
    SafetyThresholds,
    Stage,
    map_prediction_state,
)
from src.recommend_hybrid.v3.feasibility import evaluate_action
from src.recommend_hybrid.v3.pipeline import RecommendationV3Pipeline
from src.recommend_hybrid.v3.plan_builder import build_personalized_plan
from src.recommend_hybrid.v3.prediction_adapter import binary_entropy, prediction_result_to_v3_fields
from src.recommend_hybrid.v3.ranker import RuleScoreRanker
from src.recommend_hybrid.v3.risk_router import stratify_risk


def _features(**kwargs) -> RecommendationFeatures:
    base = dict(
        student_key="1",
        course_key="AAA::2013J",
        record_id="abc",
        stage=Stage.EARLY_35,
        cutoff_day=90,
        risk_probability=0.7,
        predicted_risk=1,
        prediction_threshold=0.4,
        uncertainty=0.2,
        course_progress=0.35,
        missing_assessment_count=1,
        due_soon_count=1,
        completion_rate=0.5,
        quiz_available=True,
        vle_access_available=True,
        study_material_available=True,
        active_day_rate=0.2,
        regularity_score=0.3,
        content_coverage=0.4,
        inactivity_streak=8,
        quiz_activity=0.1,
        time_to_deadline_days=5,
    )
    base.update(kwargs)
    return RecommendationFeatures(**base)


def test_100pct_cannot_map_to_intervention():
    try:
        map_prediction_state("100pct")
    except ValueError as exc:
        assert "100pct" in str(exc) or "intervention" in str(exc).lower()
    else:
        raise AssertionError("100pct must be rejected")


def test_stage_mapping():
    assert map_prediction_state("20pct") is Stage.EARLY_20
    assert map_prediction_state("75pct") is Stage.LATE_75


def test_prediction_adapter_uses_c0_only():
    result = PredictionResult(
        dataset="oulad",
        record_id="rid",
        stage_or_endpoint="20pct",
        risk_probability=0.8,
        predicted_risk=1,
        threshold=0.4,
        uncertainty=None,
        model_id="hybrid",
        metadata={"student_key": "99", "course_key": "BBB::2014J", "cutoff_day": 50},
    )
    fields = prediction_result_to_v3_fields(result)
    assert fields["stage"] is Stage.EARLY_20
    assert abs(fields["uncertainty"] - binary_entropy(0.8)) < 1e-9
    assert fields["student_key"] == "99"
    assert "seed_disagreement" not in fields


def test_risk_router_handles_threshold_and_uncertainty():
    low = _features(risk_probability=0.2, prediction_threshold=0.4, uncertainty=0.1)
    assert stratify_risk(low, RiskThresholds(0.6, 0.05)).name == "NO_AUTOMATIC"
    review = _features(risk_probability=0.45, prediction_threshold=0.4, uncertainty=0.9)
    assert stratify_risk(review, RiskThresholds(0.6, 0.05)).name == "HUMAN_REVIEW"
    process = _features(risk_probability=0.8, prediction_threshold=0.4, uncertainty=0.2)
    assert stratify_risk(process, RiskThresholds(0.6, 0.05)).name == "PROCESS"


def test_feasibility_and_content_review_early_block():
    early = _features(stage=Stage.EARLY_20, course_progress=0.2)
    blocked = evaluate_action(CanonicalAction.TARGETED_CONTENT_REVIEW, early)
    assert blocked.eligible is False
    quiz = evaluate_action(CanonicalAction.QUIZ_RETRIEVAL_PRACTICE, early)
    assert quiz.eligible is True


def test_pipeline_recommend_top1_and_review_topk():
    pipe = RecommendationV3Pipeline(
        RuleScoreRanker(),
        RiskThresholds(0.85, 0.02),
        SafetyThresholds(0.05, 0.0, 0.95),
        review_k=3,
    )
    rec = pipe.recommend(_features(risk_probability=0.85, uncertainty=0.15, prediction_threshold=0.4))
    assert rec.route is RouteStatus.RECOMMEND
    assert len(rec.ranked_actions) == 1
    assert rec.plan is not None
    assert rec.plan.action
    review = pipe.recommend(_features(risk_probability=0.45, prediction_threshold=0.4, uncertainty=0.9))
    assert review.route is RouteStatus.HUMAN_REVIEW
    assert 1 <= len(review.ranked_actions) <= 3
    below = pipe.recommend(_features(risk_probability=0.1, prediction_threshold=0.4, uncertainty=0.2, missing_assessment_count=0, due_soon_count=0))
    assert below.route is RouteStatus.INSUFFICIENT_EVIDENCE
    assert below.ranked_actions == ()


def test_plan_uses_observed_evidence_only():
    features = _features()
    plan = build_personalized_plan(ActionScore(CanonicalAction.ASSESSMENT_COMPLETION, 0.8), features)
    text = " ".join(plan.observed_evidence)
    assert "final_result" not in text
    assert "stage=" in text
