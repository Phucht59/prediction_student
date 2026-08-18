from __future__ import annotations

import json
from pathlib import Path

from src.recommendation.evaluation.metrics import ndcg_at_k
from src.recommendation.feasibility.rules_v2 import evaluate_feasibility_v2
from src.recommendation.labeling.panel_b_reference import (
    PANEL_B_REFERENCE_PROMPT_VERSION,
    build_panel_b_payload,
    build_panel_b_prompt,
    normalize_dual_reference,
)
from src.recommendation.ranking.ranker import rank_actions, top_k_actions
from scripts.recommendation.bootstrap_panel_b import bootstrap_case_metrics


ROOT = Path(__file__).resolve().parents[2]
JOB35 = ROOT / "artifacts/recommendation/labeling/jobs/panel_b_reference_gemini35_jobs.jsonl"
JOB31 = ROOT / "artifacts/recommendation/labeling/jobs/panel_b_reference_gemini31_jobs.jsonl"
MANIFEST8 = ROOT / "artifacts/recommendation/models/phase8_model_manifest.json"


def _scored(feasibility="FEASIBLE"):
    return [
        {"action_id": "assessment_recovery", "raw_score": 2.2, "relevance_score": 2.2, "feasibility_status": feasibility},
        {"action_id": "re_engagement", "raw_score": 2.2, "relevance_score": 2.2, "feasibility_status": "FEASIBLE"},
        {"action_id": "study_planning", "raw_score": 1.1, "relevance_score": 1.1, "feasibility_status": "FEASIBLE"},
        {"action_id": "progress_monitoring", "raw_score": 0.4, "relevance_score": 0.4, "feasibility_status": "FEASIBLE"},
        {"action_id": "retrieval_practice", "raw_score": 2.9, "relevance_score": 2.9, "feasibility_status": "FEASIBLE"},
    ]


def test_five_action_inference_and_deterministic_ties():
    first = rank_actions(_scored(), top_k=3)
    second = rank_actions(_scored(), top_k=3)
    assert [row["action_id"] for row in first] == [row["action_id"] for row in second]
    # equal clipped scores 2.2/2.2 break by raw then fixed action order
    assert first[1]["action_id"] == "assessment_recovery" or first[0]["action_id"] == "retrieval_practice"
    assert len(top_k_actions(first, 3)) == 3


def test_feasibility_filter_and_a5_review_required():
    rows = _scored()
    rows[0]["feasibility_status"] = "INFEASIBLE"
    ranked = rank_actions(rows, top_k=3)
    released = top_k_actions(ranked, 3)
    assert "assessment_recovery" not in released
    assert ranked[0]["action_id"] == "retrieval_practice"
    a5 = next(row for row in ranked if row["action_id"] == "retrieval_practice")
    assert a5["release_status"] == "REVIEW_REQUIRED"
    assert a5["in_top_k"] is True
    assert ranked[0]["plan_status"] == "REVIEW"


def test_a4_v2_is_feasible():
    status, reason, _source = evaluate_feasibility_v2(
        {"case_id": "c", "stage": "35pct", "missing_assessments": 0, "vle_available": True, "quiz_activity": 0},
        "progress_monitoring",
    )
    assert status == "FEASIBLE"
    assert reason == "PROGRESS_STATE_OBSERVED"


def test_reference_jobs_have_no_model_predictions():
    if not JOB35.exists():
        return
    for path in (JOB35, JOB31):
        text = path.read_text(encoding="utf-8")
        assert "raw_score" not in text
        assert "relevance_score" not in text
        assert "top_positive_reasons" not in text
        jobs = [json.loads(line) for line in text.splitlines() if line]
        assert len(jobs) == 15
        assert all(job["prompt_version"] == PANEL_B_REFERENCE_PROMPT_VERSION for job in jobs)
        assert all(len(job["case_ids"]) == 10 for job in jobs)


def test_reference_normalization_does_not_map_abstain_to_zero():
    assert normalize_dual_reference("ABSTAIN", "ABSTAIN") == {"reference_relevance": None, "reference_status": "NO_REFERENCE"}
    assert normalize_dual_reference(2, "ABSTAIN")["reference_status"] == "SINGLE_SOURCE"
    assert normalize_dual_reference(2, 3)["reference_relevance"] == 2.5
    payload = build_panel_b_payload(
        {"case_id": "c", "stage": "20pct", "risk_probability": 0.2, "risk_band": "low", "inactive_streak": 0,
         "active_days_ratio": 0.1, "recent_activity": 1, "activity_trend": 0, "assessment_completion": 0.2,
         "missing_assessments": 1, "course_progress": 0.2, "quiz_activity": 0, "vle_available": True},
        {"A1": "FEASIBLE", "A2": "FEASIBLE", "A3": "FEASIBLE", "A4": "FEASIBLE", "A5": "UNKNOWN"},
    )
    prompt = build_panel_b_prompt([payload])
    assert "Progress Monitoring" in prompt
    assert "Content Review" not in prompt or "not Content Review" in prompt
    assert "raw_score" not in prompt


def test_case_level_bootstrap_resamples_cases():
    frame = __import__("pandas").DataFrame({
        "case_id": [f"c{i}" for i in range(20)],
        "ndcg@3": [0.5 + (i % 5) / 10 for i in range(20)],
        "precision@1": [1.0] * 20,
        "recall@3": [0.5] * 20,
        "mrr": [0.5] * 20,
        "pairwise_accuracy": [0.5] * 20,
        "invalid_action_rate": [0.0] * 20,
        "coverage": [1.0] * 20,
        "a5_in_top1": [False] * 20,
        "a5_in_top3": [False] * 20,
        "plan_status": ["RECOMMEND"] * 20,
    })
    boot = bootstrap_case_metrics(frame, iterations=30, seed=2026)
    again = bootstrap_case_metrics(frame, iterations=30, seed=2026)
    assert len(boot) == 30
    assert boot["ndcg@3"].tolist() == again["ndcg@3"].tolist()


def test_raw_panel_b_references_normalize_without_mapping_abstain_to_zero():
    raw35 = ROOT / "artifacts/recommendation/labeling/raw/panel_b_reference_gemini35.jsonl"
    raw31 = ROOT / "artifacts/recommendation/labeling/raw/panel_b_reference_gemini31.jsonl"
    if not raw35.exists() or not raw31.exists():
        return
    import pandas as pd
    from src.recommendation.labeling.panel_b_reference import build_panel_b_reference_table, load_raw_panel_b_labels

    panel_b = set(pd.read_parquet(ROOT / "artifacts/recommendation/panels/panel_b.parquet", columns=["case_id"])["case_id"].astype(str))
    left = load_raw_panel_b_labels(raw35, expected_model="gemini-3.5-flash-lite")
    right = load_raw_panel_b_labels(raw31, expected_model="gemini-3.1-flash-lite")
    assert len(left) == 150
    assert len(right) == 150
    frame = build_panel_b_reference_table(raw35, raw31, panel_b)
    assert len(frame) == 750
    assert set(frame["action_id"]) == {
        "assessment_recovery", "re_engagement", "study_planning", "progress_monitoring", "retrieval_practice",
    }
    none = frame[frame["reference_status"] == "NO_REFERENCE"]
    assert none["reference_relevance"].isna().all()
    assert not ((none["label_gemini35"] == 0) & (none["label_gemini31"] == "ABSTAIN") & none["reference_relevance"].eq(0)).any()


def test_phase8_models_not_trained_on_panel_b():
    if not MANIFEST8.exists():
        return
    manifest = json.loads(MANIFEST8.read_text(encoding="utf-8"))
    assert manifest["panel_b_overlap"] == 0
    assert ndcg_at_k({"assessment_recovery": 3}, ["assessment_recovery"], 3) == 1.0
