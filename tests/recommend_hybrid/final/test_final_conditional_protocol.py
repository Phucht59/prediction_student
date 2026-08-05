from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]


def test_protocol_is_conditional_and_runtime_disabled():
    payload = yaml.safe_load(
        (
            ROOT
            / "configs/recommend_hybrid/final/conditional_action_protocol.yaml"
        ).read_text(encoding="utf-8")
    )
    assert payload["module_boundary"] == "conditional_hybrid_action_ranker"
    assert payload["end_to_end_recommendability_in_scope"] is False
    assert payload["external_ml_ranker_allowed"] is False
    assert payload["runtime_authorized"] is False


def test_public_api_requires_integrated_head_authority():
    payload = yaml.safe_load(
        (
            ROOT
            / "configs/recommend_hybrid/final/conditional_action_protocol.yaml"
        ).read_text(encoding="utf-8")
    )
    public_api = payload["public_api"]
    assert public_api["execution_context"] == "offline_evaluation"
    assert public_api["score_authority"] == "integrated_conditional_action_head"
    assert public_api["caller_authored_action_scores_allowed"] is False
    assert public_api["canonical_action_order"] == [
        "ASSESSMENT_COMPLETION",
        "STUDY_REGULARITY",
        "VLE_ENGAGEMENT",
        "QUIZ_OR_RETRIEVAL_PRACTICE",
        "CONTENT_REVIEW",
    ]
