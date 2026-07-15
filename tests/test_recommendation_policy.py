"""Phase D safety and lineage tests for the only active recommendation policy."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.governed_recommendation import (
    POLICY_VERSION,
    WORKLOAD_CAP_MINUTES,
    action_catalog,
    advisor_decision,
    assess_snapshot,
    build_governed_recommendation,
    feature_registry,
    follow_up,
    prediction_snapshot,
    validate_recommendation,
    validate_scores,
)


MODEL_BUNDLE = {
    "model_bundle_id": "test-phase-e-bundle",
    "model_version": "N0_five_seed_development_frozen",
    "feature_contract_hash": "features",
    "preprocessor_hash": "preprocessor",
    "checkpoint_bundle_hash": "checkpoints",
}
NOW = datetime(2026, 7, 15, tzinfo=timezone.utc)
POLICY = {
    "minimum_max_model_score": 0.55,
    "maximum_entropy": 1.05,
    "max_seed_disagreement": 0.4,
    "freshness_seconds": 30 * 24 * 3600,
}


def _snapshot(seed_scores=None, **kwargs):
    return prediction_snapshot(
        student_source_reference="student-mat:development:1",
        features=kwargs.pop("features", {"G1": 8.0, "G2": 7.0}),
        seed_scores=seed_scores or [[.72, .20, .08]] * 5,
        r0_reference_class=kwargs.pop("r0_reference_class", 0),
        model_bundle=MODEL_BUNDLE,
        policy_version=POLICY_VERSION,
        input_snapshot_timestamp=kwargs.pop("input_snapshot_timestamp", NOW.isoformat()),
        prediction_timestamp=NOW.isoformat(),
    )


def test_n0_ensemble_is_exact_arithmetic_mean_and_r0_has_no_fake_confidence():
    snapshot = _snapshot([[.7, .2, .1], [.6, .3, .1], [.8, .1, .1], [.7, .2, .1], [.7, .2, .1]])
    assert np.allclose(snapshot["class_scores"], np.mean(snapshot["ensemble_seed_predictions"], axis=0))
    assert snapshot["r0"] == {"probability_available": False, "uncertainty_available": False, "deterministic_rule": True}
    assert "confidence" not in snapshot["r0"]


@pytest.mark.parametrize("scores", [
    [[.7, .2, .2]] * 5,
    [[float("nan"), .2, .1]] * 5,
    [[1.1, 0, -.1]] * 5,
    [[.7, .3, .0]] * 4,
])
def test_invalid_probability_contract_fails_fast(scores):
    with pytest.raises(ValueError):
        validate_scores(scores)


def test_missing_or_target_feature_is_never_defaulted_or_accepted():
    with pytest.raises(ValueError, match="Missing or invalid"):
        _snapshot(features={"G1": 8.0})
    with pytest.raises(ValueError, match="only G1/G2"):
        _snapshot(features={"G1": 8.0, "G2": 7.0, "G3": 4.0})
    with pytest.raises(ValueError, match="only G1/G2"):
        _snapshot(features={"G1": 8.0, "G2": 7.0, "outcome": "pass"})


def test_stale_and_disagreement_snapshots_require_advisor_or_block():
    stale = _snapshot(input_snapshot_timestamp=(NOW - timedelta(days=31)).isoformat())
    assert assess_snapshot(stale, POLICY, now=NOW)["recommendation_review_status"] == "stale_prediction"
    disagreement = _snapshot(r0_reference_class=2)
    assessment = assess_snapshot(disagreement, POLICY, now=NOW)
    assert assessment["recommendation_review_status"] == "advisor_review_required"
    assert "n0_r0_disagreement" in assessment["reasons"]


def test_context_features_are_registry_disabled_and_no_ratio_exists():
    registry = {row["feature_name"]: row for row in feature_registry()}
    assert registry["G1"]["allowed_for_recommendation"]
    assert registry["G2_minus_G1"]["allowed_for_prediction"] is False
    for name in ["studytime", "failures", "schoolsup", "famsup", "activities", "internet", "absences"]:
        assert registry[name]["allowed_for_recommendation"] is False
        assert registry[name]["reason"] == "timing_or_semantic_contract_unverified"
    assert "absence_ratio" not in str(action_catalog())


def test_canonical_builder_is_deterministic_structured_noncausal_and_advisor_first():
    snapshot = _snapshot()
    assessment = assess_snapshot(snapshot, POLICY, now=NOW)
    first = build_governed_recommendation(snapshot, assessment)
    second = build_governed_recommendation(snapshot, assessment)
    assert first == second
    assert first["recommendation_review_status"] == "advisor_review_required"
    assert first["goals"] and first["actions"]
    assert "does not establish" in first["explanation"]["non_causal_limitation"]
    assert sum(action["weekly_workload_minutes"] for action in first["actions"]) <= WORKLOAD_CAP_MINUTES
    validate_recommendation(first)


def test_action_catalog_and_conflict_controls_reject_duplicate_and_missing_prerequisite():
    ids = [row["action_id"] for row in action_catalog()]
    assert len(ids) == len(set(ids))
    snapshot = _snapshot()
    rec = build_governed_recommendation(snapshot, assess_snapshot(snapshot, POLICY, now=NOW))
    rec["actions"].append(dict(rec["actions"][0]))
    with pytest.raises(ValueError, match="Action safety"):
        validate_recommendation(rec)


def test_advisor_and_follow_up_lifecycles_have_required_fields():
    snapshot = _snapshot()
    rec = build_governed_recommendation(snapshot, assess_snapshot(snapshot, POLICY, now=NOW))
    decision = advisor_decision(rec["recommendation_revision_id"], "request_more_information", "advisor:1", "Need topic confirmation")
    event = follow_up(rec["actions"][0]["action_id"], "2026-08-01")
    assert decision["decision"] == "request_more_information"
    assert event["completion_status"] == "scheduled"


def test_blocked_snapshot_creates_no_actions_and_no_true_outcome_is_needed():
    snapshot = _snapshot()
    snapshot["features"].pop("G2")
    assessment = assess_snapshot(snapshot, POLICY, now=NOW)
    rec = build_governed_recommendation(snapshot, assessment)
    assert assessment["recommendation_review_status"] == "insufficient_information"
    assert rec["actions"] == [] and rec["goals"] == []
    validate_recommendation(rec)
