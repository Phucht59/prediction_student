import hashlib

from scripts.recommend_hybrid.explainable_v2.import_llm_annotations import validate_record


def _valid_record():
    return {
        "case_id": "case_aaaaaaaaaaaaaaaaaaaaaaaa",
        "panel_id": "PANEL_A",
        "action_id": "ASSESSMENT_COMPLETION",
        "relevance_score": 3,
        "abstain": False,
        "evidence_ids": ["missing_assessment_count"],
        "rationale": "A missing assessment directly supports this action.",
        "contraindication_detected": False,
        "safety_flag": False,
        "reviewer_id": "gemini_external_reviewer_01",
        "reviewer_configuration_id": "gemini-3.6-flash_temperature_0_v1",
        "reviewer_type": "REAL_EXTERNAL_LLM_REVIEW",
        "provider": "Google Gemini API",
        "model_name": "gemini-3.6-flash",
        "request_id": "clientreq_001",
        "response_id": "providerresp_001",
        "batch_id": "panel_a_batch_01",
        "prompt_version": "external_reviewer_v1",
        "prompt_sha256": "a" * 64,
        "request_batch_sha256": "b" * 64,
        "raw_request_sha256": "c" * 64,
        "raw_response_sha256": "d" * 64,
        "response_record_index": 0,
        "response_record_sha256": hashlib.sha256(b"record").hexdigest(),
        "created_at": "2026-08-07T07:00:00+00:00",
    }


def _validate(rec):
    return validate_record(
        rec,
        known_cases={"case_aaaaaaaaaaaaaaaaaaaaaaaa"},
        case_panels={"case_aaaaaaaaaaaaaaaaaaaaaaaa": "PANEL_A"},
        case_candidate_actions={
            "case_aaaaaaaaaaaaaaaaaaaaaaaa": ["ASSESSMENT_COMPLETION"]
        },
        case_allowed_evidence_ids={
            "case_aaaaaaaaaaaaaaaaaaaaaaaa": {
                "missing_assessment_count",
                "assessments_due",
                "vle_available",
                "contraindications",
            }
        },
        approved_providers={"Google Gemini API"},
        locked_prompt_hash="a" * 64,
        locked_prompt_version="external_reviewer_v1",
        envelope_registry={},
    )


def test_missing_panel_id_rejected():
    rec = _valid_record()
    del rec["panel_id"]
    ok, code, _ = _validate(rec)
    assert ok is False
    assert code == "MISSING_REQUIRED_FIELD"


def test_missing_response_id_rejected():
    rec = _valid_record()
    rec["response_id"] = ""
    ok, code, _ = _validate(rec)
    assert ok is False
    assert code == "EMPTY_REQUIRED_FIELD"


def test_missing_prompt_sha256_rejected():
    rec = _valid_record()
    del rec["prompt_sha256"]
    ok, code, _ = _validate(rec)
    assert ok is False
    assert code == "MISSING_REQUIRED_FIELD"


def test_wrong_prompt_sha256_rejected():
    rec = _valid_record()
    rec["prompt_sha256"] = "f" * 64
    ok, code, _ = _validate(rec)
    assert ok is False
    assert code == "PROMPT_HASH_MISMATCH"


def test_missing_raw_response_sha256_rejected():
    rec = _valid_record()
    rec["raw_response_sha256"] = ""
    ok, code, _ = _validate(rec)
    assert ok is False
    assert code == "EMPTY_REQUIRED_FIELD"


def test_invalid_raw_response_sha256_rejected():
    rec = _valid_record()
    rec["raw_response_sha256"] = "not-a-sha"
    ok, code, _ = _validate(rec)
    assert ok is False
    assert code == "INVALID_RAW_RESPONSE_SHA256"


def test_unknown_evidence_id_rejected():
    rec = _valid_record()
    rec["evidence_ids"] = ["invented_hidden_feature"]
    ok, code, _ = _validate(rec)
    assert ok is False
    assert code == "UNKNOWN_EVIDENCE_ID"


def test_missing_evidence_when_not_abstaining_rejected():
    rec = _valid_record()
    rec["evidence_ids"] = []
    ok, code, _ = _validate(rec)
    assert ok is False
    assert code == "MISSING_EVIDENCE"


def test_naive_created_at_rejected():
    rec = _valid_record()
    rec["created_at"] = "2026-08-07T07:00:00"
    ok, code, _ = _validate(rec)
    assert ok is False
    assert code == "INVALID_CREATED_AT"


def test_wrong_prompt_version_rejected():
    rec = _valid_record()
    rec["prompt_version"] = "other_prompt"
    ok, code, _ = _validate(rec)
    assert ok is False
    assert code == "PROMPT_VERSION_MISMATCH"


def test_valid_schema_reaches_envelope_gate():
    rec = _valid_record()
    ok, code, _ = _validate(rec)
    assert ok is False
    assert code == "ENVELOPE_NOT_FOUND"
