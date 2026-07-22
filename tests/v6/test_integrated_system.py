from __future__ import annotations

import copy
import hashlib
import json

import pandas as pd
import pytest

from src.studies.v6.contract import ARTIFACT_ROOT, ROOT
from src.studies.v6.decision_policy import apply_decision_policy
from src.studies.v6.recommendation import (
    generate_plan,
    recommendation_input,
    validate_plan,
    validate_recommendation_input,
)
from src.studies.v6.risk_profile import validate_risk_profile
from src.studies.v6.service import replay_recommendation, validate_lineage


@pytest.fixture(scope="module")
def profile():
    return pd.read_parquet(ARTIFACT_ROOT / "prediction/risk_profiles.parquet").iloc[0].to_dict()


def test_risk_profile_schema_probability_and_lineage(profile):
    validate_risk_profile(profile)
    assert profile["checkpoint_sha256"] and profile["feature_contract_sha256"]
    assert 0 <= profile["risk_percentile"] <= 1


def test_confidence_disagreement_and_abstention_rules(profile):
    changed = copy.deepcopy(profile)
    changed["confidence_level"] = "LOW_CONFIDENCE"
    changed["decision_status"] = "ABSTAIN_REVIEW_REQUIRED"
    policy = apply_decision_policy(changed)
    assert policy["requires_expert_review"]
    assert policy["risk_mechanism"] == "UNCERTAIN_RISK"


def test_prediction_output_is_recommendation_input(profile):
    value = recommendation_input(profile)
    validate_recommendation_input(value)
    assert value["schema_version"] == "recommendation_input_v2"


def test_wrong_schema_rejected(profile):
    value = recommendation_input(profile)
    value["schema_version"] = "wrong"
    with pytest.raises(ValueError):
        validate_recommendation_input(value)


def test_stale_profile_rejected(profile):
    with pytest.raises(ValueError):
        recommendation_input(
            profile, current_state_cutoff_day=int(profile["cutoff_day"]) + 1
        )


def test_invalid_probability_rejected(profile):
    changed = copy.deepcopy(profile)
    changed["probability_at_risk"] = 2.0
    with pytest.raises(ValueError):
        validate_risk_profile(changed)


def test_plan_has_reasons_lineage_workload_and_no_duplicates(profile):
    plan = generate_plan(profile)
    validate_plan(plan)
    actions = plan["recommended_actions"]
    assert all(action["reason_codes"] for action in actions)
    assert len({action["action_id"] for action in actions}) == len(actions)
    assert plan["expected_weekly_minutes"] <= 180
    assert plan["risk_profile_lineage_id"] == profile["lineage_id"]


def test_changed_mechanism_changes_recommendation(profile):
    engagement = copy.deepcopy(profile)
    engagement.update(
        confidence_level="HIGH_CONFIDENCE",
        decision_status="PREDICTED",
        withdrawal_risk_horizon=0.8,
        probability_fail=0.2,
        probability_pass=0.7,
        probability_distinction=0.1,
    )
    academic = copy.deepcopy(engagement)
    academic.update(
        withdrawal_risk_horizon=0.2,
        probability_fail=0.8,
        probability_pass=0.15,
        probability_distinction=0.05,
    )
    assert generate_plan(engagement)["risk_mechanism"] != generate_plan(academic)[
        "risk_mechanism"
    ]


def test_increased_uncertainty_raises_escalation(profile):
    changed = copy.deepcopy(profile)
    changed["confidence_level"] = "LOW_CONFIDENCE"
    changed["decision_status"] = "ABSTAIN_REVIEW_REQUIRED"
    assert generate_plan(changed)["requires_expert_review"]


def test_end_to_end_lineage_and_replay():
    plan = json.loads(
        (ARTIFACT_ROOT / "recommendation/plans.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert validate_lineage(plan["plan_id"])["status"] == "PASS"
    assert replay_recommendation(plan["plan_id"])["exact"]


def test_expert_labels_are_pending_not_fabricated():
    value = json.loads(
        (ARTIFACT_ROOT / "recommendation/expert_evaluation/metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert value["status"] in {"PENDING_EXPERT_LABELS", "COMPLETE"}
    assert value["synthetic_labels_created"] is False


def test_checksum_manifest_replays_exactly():
    manifest = json.loads(
        (ARTIFACT_ROOT / "checksums/manifest.json").read_text(encoding="utf-8")
    )
    for record in manifest["files"]:
        path = ROOT / record["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
