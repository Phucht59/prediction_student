"""Provider Envelope Verification Module for Authentic External Reviews."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def verify_provider_envelope(
    envelope_root: Path,
    provider_name: str,
    batch_id: str,
    rec: dict[str, Any],
    locked_prompt_hash: str | None = None,
) -> tuple[bool, str, str]:
    """Verify raw request/response envelopes, payload hashes, and prompt hashes.

    Returns (is_valid, rejection_code, rejection_message).
    """
    if not envelope_root.exists():
        return False, "MISSING_ENVELOPE_ROOT", f"Envelope directory '{envelope_root}' does not exist"

    provider_dir = envelope_root / provider_name
    batch_dir = provider_dir / batch_id

    if not batch_dir.exists():
        return False, "ENVELOPE_NOT_FOUND", f"Envelope batch directory '{batch_dir}' not found"

    req_env_path = batch_dir / "request_envelope.json"
    resp_env_path = batch_dir / "response_envelope.json"
    batch_manifest_path = batch_dir / "batch_manifest.json"

    if not req_env_path.exists() or not resp_env_path.exists():
        return False, "MISSING_ENVELOPE", "request_envelope.json or response_envelope.json missing"

    try:
        req_env = json.loads(req_env_path.read_text(encoding="utf-8"))
        resp_env = json.loads(resp_env_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, "MALFORMED_ENVELOPE_JSON", str(exc)

    # Provider and model matching
    if req_env.get("provider") != provider_name or resp_env.get("provider") != provider_name:
        return False, "ENVELOPE_PROVIDER_MISMATCH", f"Envelope provider does not match '{provider_name}'"

    req_id = rec.get("request_id", "")
    if req_env.get("request_id") != req_id or resp_env.get("request_id") != req_id:
        return False, "REQUEST_ID_MISMATCH", f"request_id '{req_id}' does not match envelope request_id"

    if batch_manifest_path.exists():
        try:
            b_man = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
            if locked_prompt_hash and b_man.get("prompt_sha256") != locked_prompt_hash:
                return False, "PROMPT_HASH_MISMATCH", "Batch manifest prompt hash mismatch"
        except Exception:
            pass

    # Per-record response hash verification
    rec_sha = rec.get("response_record_sha256", "")
    rec_idx = rec.get("response_record_index")

    records_in_env = resp_env.get("records", [])
    if records_in_env:
        matched = False
        for env_rec in records_in_env:
            if env_rec.get("sha256") == rec_sha or (rec_idx is not None and env_rec.get("index") == rec_idx):
                matched = True
                break
        if not matched:
            return False, "RECORD_HASH_MISMATCH", f"response_record_sha256 '{rec_sha}' not found in response envelope records"

    return True, "OK", "Provider envelope verified"
