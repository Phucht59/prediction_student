"""Required V3 finalization invariants. No Gemini calls. No Panel B."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

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
from src.recommend_hybrid.v3.feasibility import evaluate_action, feasible_actions
from src.recommend_hybrid.v3.metrics import (
    evaluate_runtime_equivalent_ranking,
)
from src.recommend_hybrid.v3.pipeline import RecommendationV3Pipeline
from src.recommend_hybrid.v3.plan_builder import build_personalized_plan
from src.recommend_hybrid.v3.ranker import RuleScoreRanker
from src.recommend_hybrid.v3.risk_router import stratify_risk

ROOT = Path(__file__).resolve().parents[3]
V3_SRC = ROOT / "src" / "recommend_hybrid" / "v3"
V3_SCRIPTS = ROOT / "scripts" / "recommend_hybrid" / "v3"


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


def _pipe(ranker=None) -> RecommendationV3Pipeline:
    return RecommendationV3Pipeline(
        ranker or RuleScoreRanker(),
        RiskThresholds(0.85, 0.02),
        SafetyThresholds(0.05, 0.0, 0.95),
        review_k=3,
    )


class ForcedIneligibleRanker:
    def score(self, features, eligible_actions):
        # If pipeline is correct this ranker never sees ineligible actions.
        assert CanonicalAction.TARGETED_CONTENT_REVIEW not in eligible_actions or features.stage is not Stage.EARLY_20
        return tuple(
            ActionScore(action, 0.9 - 0.05 * index) for index, action in enumerate(eligible_actions)
        )


def test_infeasible_action_never_emitted():
    early = _features(stage=Stage.EARLY_20, course_progress=0.2)
    assert evaluate_action(CanonicalAction.TARGETED_CONTENT_REVIEW, early).eligible is False
    decision = _pipe(ForcedIneligibleRanker()).recommend(early)
    emitted = {item.action for item in decision.ranked_actions}
    assert CanonicalAction.TARGETED_CONTENT_REVIEW not in emitted
    for item in feasible_actions(early):
        if not item.eligible:
            assert item.action not in emitted


def test_runtime_equivalent_ranking_filters_infeasible():
    frame = pd.DataFrame(
        [
            {"query_id": "q1", "action_id": "A", "score": 0.99, "relevance": 3, "eligible": False},
            {"query_id": "q1", "action_id": "B", "score": 0.50, "relevance": 2, "eligible": True},
            {"query_id": "q1", "action_id": "C", "score": 0.10, "relevance": 0, "eligible": True},
            {"query_id": "q2", "action_id": "A", "score": 0.80, "relevance": 0, "eligible": False},
        ]
    )
    metrics = evaluate_runtime_equivalent_ranking(frame)
    assert metrics.query_count == 1
    assert metrics.invalid_action_rate == 0.0
    assert metrics.precision_at_1 == 1.0
    assert metrics.unique_top1_actions == 1


def test_invalid_action_metric_zero_when_pipeline_invariant_holds():
    frame = pd.DataFrame(
        [
            {"query_id": "q1", "action_id": "ASSESSMENT_COMPLETION", "score": 0.8, "relevance": 3, "eligible": True},
            {"query_id": "q1", "action_id": "RECOVER_ENGAGEMENT", "score": 0.2, "relevance": 0, "eligible": True},
            {"query_id": "q2", "action_id": "STUDY_REGULARITY", "score": 0.7, "relevance": 2, "eligible": True},
        ]
    )
    assert evaluate_runtime_equivalent_ranking(frame).invalid_action_rate == 0.0


def test_nullable_router_fields_cannot_typeerror():
    features = _features()
    stratify_risk(features, RiskThresholds(0.7, 0.05))
    # Retired H1 signal must not be required.
    assert not hasattr(features, "seed_disagreement") or getattr(features, "seed_disagreement", None) is None
    text = (V3_SRC / "risk_router.py").read_text(encoding="utf-8")
    assert "maximum_seed_disagreement" not in text
    assert "seed_disagreement" not in text


def test_100pct_cannot_intervene():
    with pytest.raises(ValueError):
        map_prediction_state("100pct")
    with pytest.raises(ValueError):
        map_prediction_state("FINAL-100")


def test_panel_b_not_imported_by_v3_development():
    forbidden = (
        "panel_b_real_external_reviews",
        "PANEL_B_FINAL_HELDOUT",
        "PANEL_B_EVALUATION_PROTOCOL",
    )
    scanned = list(V3_SRC.rglob("*.py")) + [
        V3_SCRIPTS / "fit_five_ebm.py",
        V3_SCRIPTS / "rebase_labels.py",
        V3_SCRIPTS / "build_v3_features.py",
        V3_SCRIPTS / "materialize_c0_downstream.py",
        V3_SCRIPTS / "diagnose_and_challenger.py",
    ]
    for path in scanned:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} references {token}"


def test_panel_c_not_imported_by_training():
    trainers = [
        V3_SCRIPTS / "fit_five_ebm.py",
        V3_SCRIPTS / "rebase_labels.py",
        V3_SCRIPTS / "diagnose_and_challenger.py",
        V3_SRC / "ranker.py",
        V3_SRC / "weak_labels.py",
    ]
    for path in trainers:
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        assert "panel_c_reviews" not in lowered
        assert "panel_c_sampled_cases" not in lowered
        assert "artifacts/recommend_hybrid/v3/panel_c" not in lowered


def test_human_review_returns_top3_feasible():
    decision = _pipe().recommend(_features(risk_probability=0.45, prediction_threshold=0.4, uncertainty=0.9))
    assert decision.route is RouteStatus.HUMAN_REVIEW
    assert 1 <= len(decision.ranked_actions) <= 3
    eligible = {item.action for item in feasible_actions(_features()) if item.eligible}
    for item in decision.ranked_actions:
        assert item.action in eligible


def test_recommend_returns_top1_feasible():
    decision = _pipe().recommend(_features(risk_probability=0.85, uncertainty=0.15, prediction_threshold=0.4))
    assert decision.route is RouteStatus.RECOMMEND
    assert len(decision.ranked_actions) == 1
    assert decision.plan is not None
    assert evaluate_action(decision.ranked_actions[0].action, _features()).eligible is True


def test_plan_builder_is_wired_and_pre_cutoff_only():
    features = _features()
    decision = _pipe().recommend(features)
    assert decision.plan is not None
    blob = " ".join(decision.plan.observed_evidence) + decision.plan.reason + decision.plan.what_to_do
    assert "final_result" not in blob
    assert "risk_probability" not in blob
    plan = build_personalized_plan(ActionScore(CanonicalAction.ASSESSMENT_COMPLETION, 0.8), features)
    assert "stage=" in " ".join(plan.observed_evidence)


def test_gemini_absent_from_runtime_and_seed_disagreement_not_required():
    for path in V3_SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "google.generativeai" not in text
        assert "genai.Client" not in text
        assert "generativelanguage.googleapis.com" not in text
    fields = _features()
    assert "seed_disagreement" not in fields.__dataclass_fields__
