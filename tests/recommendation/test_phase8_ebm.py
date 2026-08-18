from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.recommendation.evaluation.metrics import clip_score, ndcg_at_k
from src.recommendation.feasibility.rules_v2 import a4_feasibility_audit, evaluate_feasibility_v2
from src.recommendation.models.datasets import build_action_training, eligible_training_mask
from src.recommendation.models.features import APPROVED_FEATURES, audit_course_progress, encode_state_features
from src.recommendation.models.train import make_folds
from src.recommendation.weak_supervision.matrix import FINAL_ACTIONS


ROOT = Path(__file__).resolve().parents[2]
PANEL_A = ROOT / "artifacts/recommendation/panels/panel_a.parquet"
PANEL_B = ROOT / "artifacts/recommendation/panels/panel_b.parquet"
SILVER = ROOT / "artifacts/recommendation/weak_supervision/silver_labels.parquet"
TRAIN_DIR = ROOT / "artifacts/recommendation/models/training"
MANIFEST = ROOT / "artifacts/recommendation/models/phase8_model_manifest.json"


def test_no_weak_evidence_excluded_and_review_included():
    silver = pd.read_parquet(SILVER)
    mask = eligible_training_mask(silver)
    assert not silver.loc[mask, "silver_status"].eq("NO_WEAK_EVIDENCE").any()
    a5 = silver[silver["action_id"] == "retrieval_practice"]
    assert a5.loc[eligible_training_mask(a5), "silver_status"].eq("REVIEW").all()
    a1 = silver[silver["action_id"] == "assessment_recovery"]
    assert int(eligible_training_mask(a1).sum()) == 141


def test_target_is_expected_relevance_and_confidence_not_a_feature():
    assert "expected_relevance" not in APPROVED_FEATURES
    assert "silver_confidence" not in APPROVED_FEATURES
    assert "confidence" not in APPROVED_FEATURES
    assert "entropy" not in APPROVED_FEATURES
    assert "hard_label" not in APPROVED_FEATURES
    assert "case_id" not in APPROVED_FEATURES
    assert "student_id" not in APPROVED_FEATURES


def test_course_progress_is_redundant_stage():
    panel = pd.read_parquet(PANEL_A)
    audit = audit_course_progress(panel)
    assert audit["exclude"] is True
    assert audit["status"] == "FEATURE_EXCLUDED_REDUNDANT_STAGE"
    features = encode_state_features(panel)
    assert list(features.columns) == list(APPROVED_FEATURES)
    assert "course_progress" not in features.columns
    assert "risk_band" not in features.columns


def test_training_counts_and_panel_b_exclusion():
    if not TRAIN_DIR.exists():
        pytest.skip("training datasets not built")
    panel_b = set(pd.read_parquet(PANEL_B, columns=["case_id"])["case_id"].astype(str))
    expected = {"assessment_recovery": 141, "re_engagement": 500, "study_planning": 500, "progress_monitoring": 500, "retrieval_practice": 311}
    for action, n in expected.items():
        key = {"assessment_recovery": "A1", "re_engagement": "A2", "study_planning": "A3", "progress_monitoring": "A4", "retrieval_practice": "A5"}[action]
        frame = pd.read_parquet(TRAIN_DIR / f"{key}_training.parquet")
        assert len(frame) == n
        assert frame["action_id"].eq(action).all()
        assert set(frame["case_id"].astype(str)).isdisjoint(panel_b)
        assert list(col for col in APPROVED_FEATURES if col in frame.columns) == list(APPROVED_FEATURES)
        assert frame["silver_status"].isin(["VALID", "REVIEW"]).all()


def test_prediction_clipping_and_deterministic_folds():
    raw = np.array([-0.2, 1.4, 3.8])
    assert np.allclose(clip_score(raw), [0.0, 1.4, 3.0])
    first = make_folds(141, n_splits=5, seed=2026)
    second = make_folds(141, n_splits=5, seed=2026)
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(first, second))
    seen = np.concatenate([val for _, val in first])
    assert len(np.unique(seen)) == 141


def test_cv_group_leakage_not_applicable_because_identities_are_unique():
    panel = pd.read_parquet(PANEL_A)
    assert panel["student_id"].nunique() == 500
    assert panel["enrollment_identity"].nunique() == 500
    assert not panel["student_id"].duplicated().any()


def test_a4_feasibility_migration():
    audit = a4_feasibility_audit()
    assert audit["old_rule"]["status"] == "UNKNOWN"
    assert audit["new_rule"]["status"] == "FEASIBLE"
    assert audit["historical_artifact_mutated"] is False
    row = {"case_id": "x", "stage": "20pct", "missing_assessments": 1, "vle_available": True, "quiz_activity": 0}
    status, reason, _source = evaluate_feasibility_v2(row, "A4")
    assert status == "FEASIBLE"
    assert reason == "PROGRESS_STATE_OBSERVED"


def test_five_ebm_models_and_manifest_when_present():
    if not MANIFEST.exists():
        pytest.skip("Phase 8 models not frozen")
    import json
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(manifest["models"]) == set(FINAL_ACTIONS)
    assert manifest["panel_b_overlap"] == 0
    assert manifest["models"]["retrieval_practice"]["quality_status"] == "REVIEW"
    assert "PASS_WITH_WARNING" in manifest["models"]["progress_monitoring"]["quality_status"]
    assert manifest["features"] == list(APPROVED_FEATURES)
    for action in FINAL_ACTIONS:
        path = ROOT / manifest["models"][action]["artifact_path"]
        assert path.exists()
    again = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert json.dumps(manifest, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_ndcg_does_not_treat_missing_reference_as_zero():
    reference = {"assessment_recovery": 3.0, "re_engagement": 1.0}
    score = ndcg_at_k(reference, ["study_planning", "assessment_recovery"], 3)
    assert score is not None
    assert score > 0
