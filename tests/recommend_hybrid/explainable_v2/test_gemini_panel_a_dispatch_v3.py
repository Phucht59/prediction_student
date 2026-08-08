from pathlib import Path
from collections import deque
import json

from scripts.recommend_hybrid.explainable_v2 import dispatch_gemini_panel_a_batch01_v3 as dispatcher

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


def test_v4_source_gate_accepts_current_canonical_batch():
    cases = dispatcher.load_cases()
    dispatcher.validate_v4_source_gate(cases)


def test_quarantine_is_not_resume_source():
    dispatcher.assert_quarantine_is_not_resume_source()


def test_retry_after_seconds_numeric():
    assert dispatcher.retry_after_seconds({"Retry-After": "7"}) == 7.0


def test_rolling_rate_limit_caps_all_http_attempts_below_13_per_minute():
    timestamps = deque([0.0] * dispatcher.MAX_REQUESTS_PER_MINUTE)
    clock = [0.0]
    sleeps = []

    def sleep_and_advance(delay):
        sleeps.append(delay)
        clock[0] += delay

    dispatcher.wait_for_rate_limit_slot(
        timestamps=timestamps,
        monotonic=lambda: clock[0],
        sleeper=sleep_and_advance,
    )

    assert dispatcher.MAX_REQUESTS_PER_MINUTE == 12
    assert sleeps == [
        dispatcher.RATE_LIMIT_WINDOW_SECONDS
        + dispatcher.RATE_LIMIT_SAFETY_SECONDS
    ]
    assert len(timestamps) == 1


def test_safe_response_headers_are_allowlisted_only():
    headers = {
        "Content-Type": "application/json",
        "Date": "Fri, 07 Aug 2026 12:00:00 GMT",
        "x-goog-request-id": "safe-id",
        "Authorization": "secret",
        "x-goog-api-key": "secret",
    }
    safe = dispatcher.safe_response_headers(headers)
    assert safe["content-type"] == "application/json"
    assert safe["date"] == "Fri, 07 Aug 2026 12:00:00 GMT"
    assert safe["x-goog-request-id"] == "safe-id"
    assert "Authorization" not in safe
    assert "x-goog-api-key" not in safe


def test_mixed_review_family_is_explicitly_allowlisted():
    assert dispatcher.ALLOWED_REVIEW_MODELS == {
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
    }
    assert (
        dispatcher.MODEL_MIXING_POLICY
        == "EXPLICIT_GEMINI_3_5_FLASH_FAMILY_MIXED_REVIEWERS"
    )


def test_completed_batch_rejects_unapproved_review_model():
    records = [
        {
            "model_name": "some-other-model",
            "model_version": "x",
        }
    ]
    try:
        dispatcher.verify_completed_batch_before_import(
            normalized_records=records,
            prompt_hash="0" * 64,
        )
    except RuntimeError as exc:
        assert "unapproved model_name" in str(exc)
    else:
        raise AssertionError("unapproved review model was not rejected")


def test_dispatcher_source_contains_verifier_gated_import():
    source = Path(dispatcher.__file__).read_text(encoding="utf-8")
    verify_pos = source.rindex("verify_completed_batch_before_import(")
    copy_pos = source.index("shutil.copyfile(NORMALIZED_PATH, IMPORT_RAW_PATH)")
    assert verify_pos < copy_pos
