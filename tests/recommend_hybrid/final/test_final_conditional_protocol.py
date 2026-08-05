from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / "configs/recommend_hybrid/final/conditional_action_protocol.yaml"


def protocol() -> dict:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_protocol_is_conditional_and_runtime_disabled():
    payload = protocol()
    assert payload["module_boundary"] == "conditional_hybrid_action_ranker"
    assert payload["end_to_end_recommendability_in_scope"] is False
    assert payload["external_ml_ranker_allowed"] is False
    assert payload["runtime_authorized"] is False


def test_protocol_freezes_canonical_action_identity_order():
    payload = protocol()
    assert payload["action_schema"]["canonical_order"] == [
        "ASSESSMENT_COMPLETION",
        "STUDY_REGULARITY",
        "VLE_ENGAGEMENT",
        "QUIZ_OR_RETRIEVAL_PRACTICE",
        "CONTENT_REVIEW",
    ]
    assert payload["action_schema"]["score_vector_length"] == 5
    assert payload["action_schema"]["identity_mapping_required"] is True
    assert (
        payload["action_schema"]["positional_mapping_to_eligible_subset_forbidden"]
        is True
    )


def test_protocol_forbids_caller_score_fallback():
    payload = protocol()
    contract = payload["scoring_contract"]
    assert contract["explicit_policy_authorization_required"] is True
    assert contract["integrated_head_output_required"] is True
    assert contract["caller_action_payload_score_fallback_allowed"] is False
