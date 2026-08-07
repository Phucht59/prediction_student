"""Provider-envelope verification for authentic external reviews."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.recommend_hybrid.explainable_v2.provenance import (
    is_sha256,
    response_record_sha256,
    sha256_bytes,
)


def _load_json(path: Path, code: str) -> tuple[dict[str, Any] | None, str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, code, f"{path.name}: {exc}"
    if not isinstance(data, dict):
        return None, code, f"{path.name} root must be a JSON object"
    return data, "OK", ""


def _safe_payload_path(batch_dir: Path, relative_name: Any) -> Path | None:
    if not isinstance(relative_name, str) or not relative_name.strip():
        return None
    relative = Path(relative_name)
    if relative.is_absolute():
        return None
    candidate = (batch_dir / relative).resolve()
    root = batch_dir.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _find_request_record(
    records: list[Any],
    request_id: str,
    case_id: str,
) -> dict[str, Any] | None:
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("request_id") == request_id and record.get("case_id") == case_id:
            return record
    return None


def _find_response_record(
    records: list[Any],
    request_id: str,
    response_id: str,
    case_id: str,
    action_id: str,
    record_sha256: str,
) -> dict[str, Any] | None:
    for record in records:
        if not isinstance(record, dict):
            continue
        if (
            record.get("request_id") == request_id
            and record.get("response_id") == response_id
            and record.get("case_id") == case_id
            and record.get("action_id") == action_id
            and record.get("sha256") == record_sha256
        ):
            return record
    return None


def verify_provider_envelope(
    envelope_root: Path,
    provider_name: str,
    batch_id: str,
    rec: dict[str, Any],
    locked_prompt_hash: str | None = None,
) -> tuple[bool, str, str]:
    """Verify record-level request/response provenance and exact raw payload hashes."""
    if not envelope_root.exists():
        return False, "MISSING_ENVELOPE_ROOT", f"Envelope directory '{envelope_root}' does not exist"

    batch_dir = envelope_root / provider_name / batch_id
    if not batch_dir.exists():
        return False, "ENVELOPE_NOT_FOUND", f"Envelope batch directory '{batch_dir}' not found"

    req_path = batch_dir / "request_envelope.json"
    resp_path = batch_dir / "response_envelope.json"
    manifest_path = batch_dir / "batch_manifest.json"

    if not req_path.exists() or not resp_path.exists() or not manifest_path.exists():
        return (
            False,
            "MISSING_ENVELOPE",
            "request_envelope.json, response_envelope.json, and batch_manifest.json are required",
        )

    req_env, code, msg = _load_json(req_path, "MALFORMED_REQUEST_ENVELOPE")
    if req_env is None:
        return False, code, msg
    resp_env, code, msg = _load_json(resp_path, "MALFORMED_RESPONSE_ENVELOPE")
    if resp_env is None:
        return False, code, msg
    manifest, code, msg = _load_json(manifest_path, "MALFORMED_BATCH_MANIFEST")
    if manifest is None:
        return False, code, msg

    for obj_name, obj in (
        ("request envelope", req_env),
        ("response envelope", resp_env),
        ("batch manifest", manifest),
    ):
        if obj.get("provider") != provider_name:
            return False, "ENVELOPE_PROVIDER_MISMATCH", f"{obj_name} provider mismatch"
        if obj.get("batch_id") != batch_id:
            return False, "BATCH_ID_MISMATCH", f"{obj_name} batch_id mismatch"

    model_name = str(rec.get("model_name", "")).strip()
    if not model_name:
        return False, "MISSING_MODEL_NAME", "model_name missing"
    for obj_name, obj in (
        ("request envelope", req_env),
        ("response envelope", resp_env),
        ("batch manifest", manifest),
    ):
        if obj.get("model_name") != model_name:
            return False, "MODEL_MISMATCH", f"{obj_name} model_name mismatch"

    prompt_hash = str(rec.get("prompt_sha256", "")).strip()
    if locked_prompt_hash is not None and prompt_hash != locked_prompt_hash:
        return False, "PROMPT_HASH_MISMATCH", "Record prompt_sha256 does not match locked prompt hash"
    if manifest.get("prompt_sha256") != prompt_hash:
        return False, "PROMPT_HASH_MISMATCH", "Batch manifest prompt_sha256 mismatch"

    batch_hash = str(rec.get("request_batch_sha256", "")).strip()
    if not is_sha256(batch_hash):
        return False, "INVALID_REQUEST_BATCH_SHA256", "request_batch_sha256 is missing or invalid"
    if manifest.get("request_batch_sha256") != batch_hash:
        return False, "REQUEST_BATCH_HASH_MISMATCH", "Batch manifest request_batch_sha256 mismatch"

    batch_snapshot_path = _safe_payload_path(
        batch_dir,
        manifest.get("request_batch_snapshot_file"),
    )
    if batch_snapshot_path is None or not batch_snapshot_path.is_file():
        return False, "REQUEST_BATCH_SNAPSHOT_MISSING", "Exact request-batch snapshot file missing"
    if sha256_bytes(batch_snapshot_path.read_bytes()) != batch_hash:
        return (
            False,
            "REQUEST_BATCH_SNAPSHOT_HASH_MISMATCH",
            "request_batch_sha256 does not match exact request-batch snapshot bytes",
        )

    request_id = str(rec.get("request_id", "")).strip()
    response_id = str(rec.get("response_id", "")).strip()
    case_id = str(rec.get("case_id", "")).strip()
    action_id = str(rec.get("action_id", "")).strip()
    rec_sha = str(rec.get("response_record_sha256", "")).strip()

    req_records = req_env.get("records")
    resp_records = resp_env.get("records")
    if not isinstance(req_records, list) or not req_records:
        return False, "MISSING_REQUEST_RECORDS", "Request envelope has no record-level provenance"
    if not isinstance(resp_records, list) or not resp_records:
        return False, "MISSING_RESPONSE_RECORDS", "Response envelope has no record-level provenance"

    request_record = _find_request_record(req_records, request_id=request_id, case_id=case_id)
    if request_record is None:
        return (
            False,
            "REQUEST_ID_MISMATCH",
            f"No request-envelope record for case_id={case_id}, request_id={request_id}",
        )

    raw_request_sha = str(rec.get("raw_request_sha256", "")).strip()
    if request_record.get("raw_request_sha256") != raw_request_sha:
        return False, "RAW_REQUEST_HASH_MISMATCH", "raw_request_sha256 does not match request envelope"

    raw_request_path = _safe_payload_path(batch_dir, request_record.get("raw_request_file"))
    if raw_request_path is None or not raw_request_path.is_file():
        return False, "RAW_REQUEST_FILE_MISSING", "Exact raw request payload file missing"
    if sha256_bytes(raw_request_path.read_bytes()) != raw_request_sha:
        return False, "RAW_REQUEST_FILE_HASH_MISMATCH", "Exact raw request payload hash mismatch"

    response_record = _find_response_record(
        resp_records,
        request_id=request_id,
        response_id=response_id,
        case_id=case_id,
        action_id=action_id,
        record_sha256=rec_sha,
    )
    if response_record is None:
        return False, "RESPONSE_RECORD_MISMATCH", "No matching response-envelope record"

    if response_record.get("index") != rec.get("response_record_index"):
        return False, "RESPONSE_INDEX_MISMATCH", "response_record_index mismatch"

    raw_response_sha = str(rec.get("raw_response_sha256", "")).strip()
    if response_record.get("raw_response_sha256") != raw_response_sha:
        return False, "RAW_RESPONSE_HASH_MISMATCH", "raw_response_sha256 does not match response envelope"

    raw_response_path = _safe_payload_path(batch_dir, response_record.get("raw_response_file"))
    if raw_response_path is None or not raw_response_path.is_file():
        return False, "RAW_RESPONSE_FILE_MISSING", "Exact raw provider response file missing"
    if sha256_bytes(raw_response_path.read_bytes()) != raw_response_sha:
        return False, "RAW_RESPONSE_FILE_HASH_MISMATCH", "Exact raw provider response hash mismatch"

    model_version = rec.get("model_version")
    if model_version is not None and response_record.get("model_version") != model_version:
        return False, "MODEL_VERSION_MISMATCH", "model_version mismatch"

    if response_record_sha256(rec) != rec_sha:
        return False, "RESPONSE_RECORD_HASH_MISMATCH", "response_record_sha256 is not canonical-record hash"

    return True, "OK", "Provider envelope verified"
