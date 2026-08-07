import json

from scripts.recommend_hybrid.explainable_v2.gemini_external_review_smoke_v3 import (
    _validate_semantics,
    build_request_body,
    build_review_schema,
)


def _case():
    return {
        "case_id": "case_aaaaaaaaaaaaaaaaaaaaaaaa",
        "panel_id": "PANEL_A",
        "candidate_actions": [
            "ASSESSMENT_COMPLETION",
            "STUDY_REGULARITY",
        ],
        "observed_pre_cutoff_evidence": {
            "missing_assessment_count": 1,
            "regularity_score": 0.8,
        },
        "availability_flags": {
            "vle_available": True,
            "quiz_available": False,
        },
        "contraindications": ["QUIZ_RETRIEVAL_PRACTICE"],
    }


def test_schema_locks_action_set_and_review_count():
    schema = build_review_schema(_case())
    reviews = schema["properties"]["reviews"]
    assert reviews["minItems"] == 2
    assert reviews["maxItems"] == 2
    assert reviews["items"]["properties"]["action_id"]["enum"] == [
        "ASSESSMENT_COMPLETION",
        "STUDY_REGULARITY",
    ]


def test_schema_locks_evidence_ids_to_public_case_fields():
    schema = build_review_schema(_case())
    evidence_enum = (
        schema["properties"]["reviews"]["items"]["properties"]["evidence_ids"]
        ["items"]["enum"]
    )
    assert "missing_assessment_count" in evidence_enum
    assert "regularity_score" in evidence_enum
    assert "vle_available" in evidence_enum
    assert "contraindications" in evidence_enum
    assert "student_id" not in evidence_enum
    assert "query_id" not in evidence_enum


def test_request_uses_locked_system_instruction_and_structured_output():
    case = _case()
    schema = build_review_schema(case)
    body = build_request_body(case, "LOCKED PROMPT", schema)

    assert body["systemInstruction"]["parts"][0]["text"] == "LOCKED PROMPT"
    assert "system_instruction" not in body
    response_format = body["generationConfig"]["responseFormat"]["text"]
    assert response_format["mimeType"] == "application/json"
    assert response_format["schema"] == schema

    serialized = json.dumps(body)
    assert "GEMINI_API_KEY" not in serialized
    assert "x-goog-api-key" not in serialized


def test_structured_schema_forbids_extra_review_properties():
    schema = build_review_schema(_case())
    assert schema["additionalProperties"] is False
    item_schema = schema["properties"]["reviews"]["items"]
    assert item_schema["additionalProperties"] is False


def test_semantic_validation_canonicalizes_candidate_action_order():
    case = _case()
    reversed_reviews = {
        "reviews": [
            {
                "action_id": "STUDY_REGULARITY",
                "relevance_score": 1,
                "abstain": False,
                "evidence_ids": ["regularity_score"],
                "rationale": "Regularity evidence provides weak support for this action.",
                "contraindication_detected": False,
                "safety_flag": False,
            },
            {
                "action_id": "ASSESSMENT_COMPLETION",
                "relevance_score": 3,
                "abstain": False,
                "evidence_ids": ["missing_assessment_count"],
                "rationale": "The missing assessment directly supports assessment completion.",
                "contraindication_detected": False,
                "safety_flag": False,
            },
        ]
    }

    ordered = _validate_semantics(case, reversed_reviews)
    assert [item["action_id"] for item in ordered] == case["candidate_actions"]


def test_semantic_validation_rejects_duplicate_evidence_ids():
    case = _case()
    payload = {
        "reviews": [
            {
                "action_id": "ASSESSMENT_COMPLETION",
                "relevance_score": 3,
                "abstain": False,
                "evidence_ids": [
                    "missing_assessment_count",
                    "missing_assessment_count",
                ],
                "rationale": "The missing assessment directly supports this action.",
                "contraindication_detected": False,
                "safety_flag": False,
            },
            {
                "action_id": "STUDY_REGULARITY",
                "relevance_score": 1,
                "abstain": False,
                "evidence_ids": ["regularity_score"],
                "rationale": "Regularity evidence provides weak support for this action.",
                "contraindication_detected": False,
                "safety_flag": False,
            },
        ]
    }

    try:
        _validate_semantics(case, payload)
    except RuntimeError as exc:
        assert "Duplicate evidence_ids" in str(exc)
    else:
        raise AssertionError("Duplicate evidence_ids must fail closed")
