from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.recommendation.evaluation.metrics import clip_score
from src.recommendation.feasibility.rules_v2 import evaluate_feasibility_v2
from src.recommendation.finalization.authority import (
    ACTION_STATUS,
    APPROVED_FEATURES,
    validate_required_artifacts,
    validate_scientific_authority,
)
from src.recommendation.models.features import APPROVED_FEATURES as FEATURES
from src.recommendation.ranking.ranker import rank_actions, top_k_actions
from src.recommendation.service import FORBIDDEN_INPUT, RecommendationService
from src.recommendation.weak_supervision.matrix import FINAL_ACTIONS


ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "artifacts/recommendation/final/FINAL_RECOMMENDATION_FREEZE_MANIFEST.json"
TRUTH = ROOT / "artifacts/recommendation/final/THESIS_RECOMMENDATION_SOURCE_OF_TRUTH.json"
PHASE9 = ROOT / "artifacts/recommendation/evaluation/phase9_manifest.json"


def test_required_authority_and_five_actions():
    assert validate_required_artifacts(ROOT) == []
    assert validate_scientific_authority(ROOT) == []
    assert list(FINAL_ACTIONS) == [
        "assessment_recovery", "re_engagement", "study_planning", "progress_monitoring", "retrieval_practice",
    ]
    assert ACTION_STATUS["retrieval_practice"] == "REVIEW"
    assert ACTION_STATUS["progress_monitoring"] == "PASS_WITH_WARNING"
    assert FEATURES == (
        "stage_code", "risk_probability", "inactive_streak", "active_days_ratio", "recent_activity",
        "activity_trend", "assessment_completion", "missing_assessments", "quiz_activity", "vle_available",
    )


def test_feature_order_matches_phase8_manifest():
    phase8 = json.loads((ROOT / "artifacts/recommendation/models/phase8_model_manifest.json").read_text(encoding="utf-8"))
    assert phase8["features"] == list(FEATURES)
    assert "course_progress" not in phase8["features"]
    assert "confidence" not in phase8["features"]


def test_clipping_a4_v2_and_a5_review_ranking():
    assert np.allclose(clip_score(np.array([-1.0, 4.0])), [0.0, 3.0])
    status, reason, _ = evaluate_feasibility_v2(
        {"case_id": "c", "stage": "50pct", "missing_assessments": 1, "vle_available": True, "quiz_activity": 0},
        "A4",
    )
    assert status == "FEASIBLE" and reason == "PROGRESS_STATE_OBSERVED"
    rows = [
        {"action_id": "assessment_recovery", "raw_score": 1.0, "relevance_score": 1.0, "feasibility_status": "INFEASIBLE"},
        {"action_id": "retrieval_practice", "raw_score": 2.4, "relevance_score": 2.4, "feasibility_status": "FEASIBLE"},
        {"action_id": "study_planning", "raw_score": 2.0, "relevance_score": 2.0, "feasibility_status": "FEASIBLE"},
        {"action_id": "re_engagement", "raw_score": 0.5, "relevance_score": 0.5, "feasibility_status": "FEASIBLE"},
        {"action_id": "progress_monitoring", "raw_score": 0.2, "relevance_score": 0.2, "feasibility_status": "FEASIBLE"},
    ]
    ranked = rank_actions(rows, top_k=3)
    assert top_k_actions(ranked, 3)[0] == "retrieval_practice"
    assert next(row for row in ranked if row["action_id"] == "retrieval_practice")["release_status"] == "REVIEW_REQUIRED"
    assert ranked[0]["plan_status"] == "REVIEW"


def test_runtime_schema_rejects_future_and_missing_fields():
    service_errors = RecommendationService.validate_state(
        {"stage": "FINAL-100", "risk_probability": 1.2, "final_result": "Fail"},
    )
    assert any("forbidden" in item or "missing" in item or "final" in item or "range" in item for item in service_errors)
    assert "final_result" in FORBIDDEN_INPUT


def test_source_of_truth_matches_phase9_artifact():
    if not TRUTH.exists():
        pytest.skip("source-of-truth not written yet")
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    phase9 = json.loads(PHASE9.read_text(encoding="utf-8"))
    assert truth["phase9"]["metrics"]["EBM"]["ndcg@3"] == phase9["metrics"]["EBM"]["ndcg@3"]
    assert truth["phase9"]["panel_b_overlap_with_training"] == 0
    assert truth["phase9"]["evaluation_name"] == "AUTOMATED_REFERENCE_EVALUATION"
    assert truth["phase8"]["models"]["retrieval_practice"]["quality_status"] == "REVIEW"


def test_freeze_manifest_is_deterministic_given_same_checksums():
    if not FREEZE.exists():
        pytest.skip("freeze manifest not written yet")
    first = json.loads(FREEZE.read_text(encoding="utf-8"))
    second = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert first["content"]["checksums"] == second["content"]["checksums"]
    assert first["content"]["features"] == list(FEATURES)
    assert set(first["content"]["actions"]) == set(FINAL_ACTIONS)


def test_bulk_inference_outputs_when_present():
    scores = ROOT / "artifacts/recommendation/final/oulad_recommendation_scores.parquet"
    plans = ROOT / "artifacts/recommendation/final/oulad_recommendation_plans.parquet"
    if not scores.exists():
        pytest.skip("bulk inference not built")
    score_frame = pd.read_parquet(scores, columns=["case_id", "action_id", "stage", "raw_score", "relevance_score"])
    plan_frame = pd.read_parquet(plans, columns=["case_id", "stage", "plan_status"])
    assert score_frame["case_id"].nunique() == len(plan_frame)
    assert (score_frame.groupby("case_id").size() == 5).all()
    assert "FINAL" not in set(score_frame["stage"].astype(str))
    assert score_frame["raw_score"].notna().all()
    assert score_frame["relevance_score"].between(0, 3).all()


def test_no_secret_literals_in_finalization_code():
    roots = [
        ROOT / "src/recommendation/finalization",
        ROOT / "src/recommendation/service.py",
        ROOT / "src/recommendation/persistence",
        ROOT / "scripts/recommendation/validate_final_freeze.py",
        ROOT / "scripts/recommendation/recommend_student.py",
        ROOT / "database/migrations",
    ]
    banned = ("AIza", "sk-ant-", "BEGIN RSA PRIVATE KEY")
    for path in roots:
        files = [path] if path.is_file() else list(path.rglob("*.py")) + list(path.rglob("*.sql"))
        for file in files:
            text = file.read_text(encoding="utf-8")
            assert "GEMINI_API_KEY=" not in text
            assert "DB_PASSWORD=supersecret" not in text
            assert not any(token in text for token in banned)
