from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from src.studies.v6_2.expert import (
    EXPERT_ROOT,
    EXPERT_SCHEMA,
    ReviewImportError,
    _read_review,
    _validate_reviews,
)
from src.studies.v6_2.recommendation import (
    MAX_ACTIONS,
    MAX_WEEKLY_MINUTES,
    WITHDRAWAL_STATUS,
    generate_plan,
    validate_observed_state,
    validate_plan,
)


@pytest.fixture(scope="module")
def profile():
    return pd.read_parquet(
        "artifacts/v6/prediction/risk_profiles.parquet"
    ).iloc[0].to_dict()


@pytest.fixture()
def observed(profile):
    state = {
        "schema_version": "observed_learning_state_v6_2",
        "record_key": None,
        "cutoff_day": int(profile["cutoff_day"]),
        "values": {
            "activity_level": 0.2,
            "inactivity_streak": 3.0,
            "assessment_progress": 0.25,
            "grade_trend": -0.1,
        },
        "lineage": {
            name: {"source_max_day": int(profile["cutoff_day"])}
            for name in (
                "activity_level",
                "inactivity_streak",
                "assessment_progress",
                "grade_trend",
            )
        },
        "post_cutoff_used": False,
        "sensitive_attributes_used": False,
    }
    from src.studies.v6_2.contract import canonical_sha256

    state["lineage_sha256"] = canonical_sha256(state)
    # Derive the opaque record key from a generated plan-safe fixture.
    from src.studies.v6_2.recommendation import _pseudonym

    state["record_key"] = _pseudonym(str(profile["record_id"]))
    state["lineage_sha256"] = canonical_sha256(
        {key: value for key, value in state.items() if key != "lineage_sha256"}
    )
    return state


def _confident(profile):
    value = copy.deepcopy(profile)
    value["confidence_level"] = "HIGH_CONFIDENCE"
    value["decision_status"] = "PREDICTED"
    value["deep_ml_disagreement"] = 0.05
    return value


def test_observed_behavior_is_invariant_to_prediction_probabilities(profile, observed):
    first = _confident(profile)
    second = copy.deepcopy(first)
    second["probability_fail"] = 0.8
    second["probability_pass"] = 0.15
    second["probability_distinction"] = 0.05
    assert observed["values"] == copy.deepcopy(observed)["values"]
    assert generate_plan(first, observed)["observed_evidence"] == generate_plan(
        second, observed
    )["observed_evidence"]


def test_post_cutoff_and_invalid_lineage_force_abstention(profile, observed):
    changed = copy.deepcopy(observed)
    changed["lineage"]["activity_level"]["source_max_day"] += 1
    from src.studies.v6_2.contract import canonical_sha256

    changed["lineage_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "lineage_sha256"}
    )
    assert "POST_CUTOFF_LINEAGE" in validate_observed_state(changed)
    plan = generate_plan(_confident(profile), changed)
    assert plan["plan_status"] == "ABSTAINED"
    assert plan["recommended_actions"] == []


def test_missing_activity_cannot_fabricate_vle_reason(profile, observed):
    changed = copy.deepcopy(observed)
    changed["values"]["activity_level"] = None
    changed["values"]["inactivity_streak"] = None
    from src.studies.v6_2.contract import canonical_sha256

    changed["lineage_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "lineage_sha256"}
    )
    plan = generate_plan(_confident(profile), changed)
    assert "VLE_ENGAGEMENT" not in {
        action["action_id"] for action in plan["recommended_actions"]
    }
    assert "LOW_VLE_ENGAGEMENT" not in plan["reason_codes"]


def test_low_confidence_and_high_disagreement_abstain(profile, observed):
    changed = _confident(profile)
    changed["confidence_level"] = "LOW_CONFIDENCE"
    assert generate_plan(changed, observed)["plan_status"] == "ABSTAINED"
    changed = _confident(profile)
    changed["deep_ml_disagreement"] = 0.30
    assert generate_plan(changed, observed)["plan_status"] == "ABSTAINED"


def test_poor_withdrawal_head_cannot_trigger_mechanism_action(profile, observed):
    low = _confident(profile)
    low.update(
        withdrawal_hazard_current=0.0,
        withdrawal_risk_horizon=0.0,
        probability_fail=0.10,
        probability_pass=0.80,
        probability_distinction=0.10,
    )
    high = copy.deepcopy(low)
    high.update(withdrawal_hazard_current=0.99, withdrawal_risk_horizon=0.99)
    first = generate_plan(low, observed)
    second = generate_plan(high, observed)
    assert first["risk_mechanism"] == second["risk_mechanism"]
    assert first["recommended_actions"] == second["recommended_actions"]
    assert first["priority"] == second["priority"]
    assert WITHDRAWAL_STATUS == "EXPLORATORY_DISABLED_FOR_RECOMMENDATION"


def test_workload_lineage_schema_and_determinism(profile, observed):
    first = generate_plan(_confident(profile), observed)
    second = generate_plan(_confident(profile), observed)
    validate_plan(first)
    assert first == second
    assert len(first["recommended_actions"]) <= MAX_ACTIONS
    assert first["expected_weekly_minutes"] <= MAX_WEEKLY_MINUTES
    assert all(action["reason_codes"] for action in first["recommended_actions"])


def test_sensitive_attributes_absent_from_plan(profile, observed):
    plan = generate_plan(_confident(profile), observed)
    text = str(plan).lower()
    for field in (
        "id_student",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
    ):
        assert field not in text


def _case_frame():
    return pd.DataFrame(
        [{"case_id": "CASE-001", "proposed_action_ids": "STUDY_SCHEDULE"}]
    )


def _valid_reviews():
    plan = pd.DataFrame(
        [
            {
                "schema_version": EXPERT_SCHEMA,
                "reviewer_id": "E01",
                "case_id": "CASE-001",
                "q1_plan_score": 4,
                "q3_missing_action": "NO",
                "q3_missing_action_text": "",
                "q4_escalation": "CORRECT",
                "q5_reason_support": "SUPPORTED",
                "q6_safety_workload": "SAFE",
                "q6_safety_note": "",
            }
        ]
    )
    action = pd.DataFrame(
        [
            {
                "schema_version": EXPERT_SCHEMA,
                "reviewer_id": "E01",
                "case_id": "CASE-001",
                "action_id": "STUDY_SCHEDULE",
                "q2_action_relevance": "APPROVE",
            }
        ]
    )
    return plan, action


def test_expert_import_accepts_strict_valid_contract():
    plan, action = _valid_reviews()
    validated_plan, validated_action = _validate_reviews(
        plan, action, _case_frame()
    )
    assert validated_plan.q1_plan_score.iloc[0] == 4
    assert len(validated_action) == 1


def test_exported_xlsx_is_readable_and_blank_not_fabricated():
    plan, action = _read_review(EXPERT_ROOT / "expert_review_form.xlsx")
    cases = pd.read_csv(EXPERT_ROOT / "expert_review_cases.csv", dtype=object)
    validated_plan, validated_action = _validate_reviews(plan, action, cases)
    assert validated_plan.empty
    assert validated_action.empty


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda p, a: p.__setitem__("schema_version", "wrong"), "schema version"),
        (lambda p, a: p.__setitem__("reviewer_id", "Alice"), "pseudonymous"),
        (lambda p, a: p.__setitem__("q1_plan_score", 7), "integer 1-5"),
        (
            lambda p, a: a.__setitem__("q2_action_relevance", "MAYBE"),
            "q2_action_relevance",
        ),
        (lambda p, a: a.__setitem__("action_id", "UNKNOWN"), "not proposed"),
    ],
)
def test_expert_import_rejects_invalid_values(mutation, error):
    plan, action = _valid_reviews()
    mutation(plan, action)
    with pytest.raises(ReviewImportError, match=error):
        _validate_reviews(plan, action, _case_frame())


def test_expert_import_rejects_duplicates_and_unknown_case():
    plan, action = _valid_reviews()
    duplicated = pd.concat([plan, plan], ignore_index=True)
    with pytest.raises(ReviewImportError, match="Duplicate"):
        _validate_reviews(duplicated, action, _case_frame())
    plan, action = _valid_reviews()
    plan.loc[0, "case_id"] = "CASE-999"
    action.loc[0, "case_id"] = "CASE-999"
    with pytest.raises(ReviewImportError, match="unknown case"):
        _validate_reviews(plan, action, _case_frame())


def test_additive_database_migration_has_no_fake_review_rows():
    text = Path(
        "database/final/migrations/011_create_v6_2_expert_review_validation.sql"
    ).read_text(encoding="utf-8").upper()
    assert "INSERT INTO RECOMMENDATION.EXPERT" not in text
    assert "DROP TABLE" not in text
    assert "TRUNCATE" not in text
