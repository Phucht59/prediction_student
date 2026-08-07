import json

from scripts.recommend_hybrid.explainable_v2.dispatch_gemini_panel_a_batch01_v3 import (
    allowed_evidence_ids,
    build_request_body,
    build_review_schema,
    extract_reviews,
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
            "regularity_score": 0.85,
        },
        "availability_flags": {
            "vle_available": True,
            "quiz_available": False,
        },
        "contraindications": ["QUIZ_RETRIEVAL_PRACTICE"],
    }


def test_production_schema_locks_action_count_and_fields():
    schema = build_review_schema(_case())
    reviews = schema["properties"]["reviews"]
    assert reviews["minItems"] == 2
    assert reviews["maxItems"] == 2
    assert reviews["items"]["additionalProperties"] is False
    assert reviews["items"]["properties"]["action_id"]["enum"] == [
        "ASSESSMENT_COMPLETION",
        "STUDY_REGULARITY",
    ]


def test_production_request_uses_verified_gemini_response_format():
    case = _case()
    schema = build_review_schema(case)
    body = build_request_body(case, "LOCKED", schema)
    assert body["systemInstruction"]["parts"][0]["text"] == "LOCKED"
    text_format = body["generationConfig"]["responseFormat"]["text"]
    assert text_format["mimeType"] == "APPLICATION_JSON"
    assert text_format["schema"] == schema


def test_public_evidence_whitelist_contains_no_identity_fields():
    allowed = allowed_evidence_ids(_case())
    assert "missing_assessment_count" in allowed
    assert "regularity_score" in allowed
    assert "vle_available" in allowed
    assert "contraindications" in allowed
    assert "student_id" not in allowed
    assert "query_id" not in allowed


def test_response_order_is_canonicalized_to_candidate_actions():
    provider_response = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "reviews": [
                                        {
                                            "action_id": "STUDY_REGULARITY",
                                            "relevance_score": 1,
                                            "abstain": False,
                                            "evidence_ids": ["regularity_score"],
                                            "rationale": (
                                                "Regularity evidence gives weak "
                                                "support for this action."
                                            ),
                                            "contraindication_detected": False,
                                            "safety_flag": False,
                                        },
                                        {
                                            "action_id": "ASSESSMENT_COMPLETION",
                                            "relevance_score": 3,
                                            "abstain": False,
                                            "evidence_ids": [
                                                "missing_assessment_count"
                                            ],
                                            "rationale": (
                                                "The missing assessment directly "
                                                "supports completion."
                                            ),
                                            "contraindication_detected": False,
                                            "safety_flag": False,
                                        },
                                    ]
                                }
                            )
                        }
                    ]
                },
            }
        ],
        "usageMetadata": {"totalTokenCount": 100},
    }
    reviews, finish_reason, usage = extract_reviews(_case(), provider_response)
    assert [item["action_id"] for item in reviews] == _case()["candidate_actions"]
    assert finish_reason == "STOP"
    assert usage["totalTokenCount"] == 100


def test_non_stop_finish_reason_fails_closed():
    provider_response = {
        "candidates": [
            {
                "finishReason": "MAX_TOKENS",
                "content": {"parts": [{"text": '{"reviews": []}'}]},
            }
        ]
    }
    try:
        extract_reviews(_case(), provider_response)
    except RuntimeError as exc:
        assert "did not finish cleanly" in str(exc)
    else:
        raise AssertionError("Non-STOP finishReason must fail closed")
