import json

from src.recommend_hybrid.explainable_v2.provenance import (
    response_record_sha256,
    sha256_bytes,
)
from src.recommend_hybrid.explainable_v2.provider_envelope import (
    verify_provider_envelope,
)


def _build_valid_batch(tmp_path):
    provider = "Google Gemini API"
    batch_id = "panel_a_batch_01"
    model = "gemini-3.6-flash"
    prompt_hash = "7" * 64

    batch_dir = tmp_path / provider / batch_id
    req_raw_dir = batch_dir / "raw_requests"
    resp_raw_dir = batch_dir / "raw_responses"
    req_raw_dir.mkdir(parents=True)
    resp_raw_dir.mkdir(parents=True)

    batch_snapshot = b'{"case_id":"case_bbbbbbbbbbbbbbbbbbbbbbbb"}\n'
    batch_snapshot_path = batch_dir / "request_batch_snapshot.jsonl"
    batch_snapshot_path.write_bytes(batch_snapshot)
    batch_hash = sha256_bytes(batch_snapshot)

    raw_request = b'{"request":"exact"}'
    raw_response = b'{"responseId":"resp_002","modelVersion":"gemini-3.6-flash"}'
    req_file = req_raw_dir / "case_bbbbbbbbbbbbbbbbbbbbbbbb.json"
    resp_file = resp_raw_dir / "case_bbbbbbbbbbbbbbbbbbbbbbbb.json"
    req_file.write_bytes(raw_request)
    resp_file.write_bytes(raw_response)

    rec = {
        "case_id": "case_bbbbbbbbbbbbbbbbbbbbbbbb",
        "panel_id": "PANEL_A",
        "action_id": "STUDY_REGULARITY",
        "relevance_score": 2,
        "abstain": False,
        "evidence_ids": ["regularity_score"],
        "rationale": "Observed regularity evidence supports this action.",
        "contraindication_detected": False,
        "safety_flag": False,
        "reviewer_id": "gemini_external_reviewer_01",
        "reviewer_configuration_id": "gemini-3.6-flash_default_sampling_v1",
        "reviewer_type": "REAL_EXTERNAL_LLM_REVIEW",
        "provider": provider,
        "model_name": model,
        "model_version": "gemini-3.6-flash",
        "request_id": "clientreq_002",
        "response_id": "resp_002",
        "batch_id": batch_id,
        "prompt_version": "external_reviewer_v1",
        "prompt_sha256": prompt_hash,
        "request_batch_sha256": batch_hash,
        "raw_request_sha256": sha256_bytes(raw_request),
        "raw_response_sha256": sha256_bytes(raw_response),
        "response_record_index": 1,
        "created_at": "2026-08-07T07:00:00+00:00",
    }
    rec["response_record_sha256"] = response_record_sha256(rec)

    request_envelope = {
        "provider": provider,
        "model_name": model,
        "batch_id": batch_id,
        "records": [
            {
                "case_id": rec["case_id"],
                "request_id": rec["request_id"],
                "raw_request_sha256": rec["raw_request_sha256"],
                "raw_request_file": "raw_requests/case_bbbbbbbbbbbbbbbbbbbbbbbb.json",
            }
        ],
    }
    response_envelope = {
        "provider": provider,
        "model_name": model,
        "batch_id": batch_id,
        "records": [
            {
                "index": rec["response_record_index"],
                "case_id": rec["case_id"],
                "request_id": rec["request_id"],
                "response_id": rec["response_id"],
                "action_id": rec["action_id"],
                "sha256": rec["response_record_sha256"],
                "raw_response_sha256": rec["raw_response_sha256"],
                "raw_response_file": "raw_responses/case_bbbbbbbbbbbbbbbbbbbbbbbb.json",
                "model_version": rec["model_version"],
            }
        ],
    }
    manifest = {
        "provider": provider,
        "model_name": model,
        "batch_id": batch_id,
        "prompt_sha256": prompt_hash,
        "request_batch_sha256": batch_hash,
        "request_batch_snapshot_file": "request_batch_snapshot.jsonl",
    }

    (batch_dir / "request_envelope.json").write_text(
        json.dumps(request_envelope), encoding="utf-8"
    )
    (batch_dir / "response_envelope.json").write_text(
        json.dumps(response_envelope), encoding="utf-8"
    )
    (batch_dir / "batch_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return rec, provider, batch_id, prompt_hash, resp_file, batch_snapshot_path


def test_record_level_provenance_passes(tmp_path):
    rec, provider, batch_id, prompt_hash, _, _ = _build_valid_batch(tmp_path)
    ok, code, _ = verify_provider_envelope(
        tmp_path, provider, batch_id, rec, locked_prompt_hash=prompt_hash
    )
    assert ok is True
    assert code == "OK"


def test_wrong_request_id_fails_closed(tmp_path):
    rec, provider, batch_id, prompt_hash, _, _ = _build_valid_batch(tmp_path)
    rec["request_id"] = "fake_request"
    rec["response_record_sha256"] = response_record_sha256(rec)
    ok, code, _ = verify_provider_envelope(
        tmp_path, provider, batch_id, rec, locked_prompt_hash=prompt_hash
    )
    assert ok is False
    assert code == "REQUEST_ID_MISMATCH"


def test_tampered_raw_response_fails_closed(tmp_path):
    rec, provider, batch_id, prompt_hash, resp_file, _ = _build_valid_batch(tmp_path)
    resp_file.write_bytes(b'{"tampered":true}')
    ok, code, _ = verify_provider_envelope(
        tmp_path, provider, batch_id, rec, locked_prompt_hash=prompt_hash
    )
    assert ok is False
    assert code == "RAW_RESPONSE_FILE_HASH_MISMATCH"


def test_wrong_response_id_fails_closed(tmp_path):
    rec, provider, batch_id, prompt_hash, _, _ = _build_valid_batch(tmp_path)
    rec["response_id"] = "fake_response"
    rec["response_record_sha256"] = response_record_sha256(rec)
    ok, code, _ = verify_provider_envelope(
        tmp_path, provider, batch_id, rec, locked_prompt_hash=prompt_hash
    )
    assert ok is False
    assert code == "RESPONSE_RECORD_MISMATCH"


def test_missing_exact_raw_request_file_fails_closed(tmp_path):
    rec, provider, batch_id, prompt_hash, _, _ = _build_valid_batch(tmp_path)
    req_path = (
        tmp_path
        / provider
        / batch_id
        / "raw_requests"
        / "case_bbbbbbbbbbbbbbbbbbbbbbbb.json"
    )
    req_path.unlink()
    ok, code, _ = verify_provider_envelope(
        tmp_path, provider, batch_id, rec, locked_prompt_hash=prompt_hash
    )
    assert ok is False
    assert code == "RAW_REQUEST_FILE_MISSING"


def test_tampered_request_batch_snapshot_fails_closed(tmp_path):
    rec, provider, batch_id, prompt_hash, _, snapshot = _build_valid_batch(tmp_path)
    snapshot.write_bytes(b'{"tampered":"batch"}\n')
    ok, code, _ = verify_provider_envelope(
        tmp_path, provider, batch_id, rec, locked_prompt_hash=prompt_hash
    )
    assert ok is False
    assert code == "REQUEST_BATCH_SNAPSHOT_HASH_MISMATCH"
