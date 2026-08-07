"""Real Gemini 3.6 external-review smoke harness with isolated importer validation.

This script intentionally keeps smoke artifacts outside the canonical import area.
It never writes API keys or HTTP headers to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.recommend_hybrid.explainable_v2 import import_llm_annotations as imp
from src.recommend_hybrid.explainable_v2.provenance import (
    canonical_json_bytes,
    canonical_text_sha256,
    response_record_sha256,
    sha256_bytes,
)

PROVIDER = "Google Gemini API"
PROMPT_VERSION = "external_reviewer_v1"
DEFAULT_MODEL = "gemini-3.6-flash"

PROMPT_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/annotations/prompts/"
    "locked_external_reviewer_prompt_v1.txt"
)
BATCH_FILE = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/annotations/prompts/"
    "panel_a_request_batches/batch_01.jsonl"
)
SMOKE_ROOT = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/annotations/smoke_gemini_v3"
)


def _load_first_case() -> dict[str, Any]:
    for line in BATCH_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            case = json.loads(line)
            if not isinstance(case, dict):
                raise RuntimeError("First batch record is not a JSON object")
            return case
    raise RuntimeError("batch_01.jsonl contains no cases")


def _allowed_evidence_ids(case: dict[str, Any]) -> list[str]:
    observed = case.get("observed_pre_cutoff_evidence", {})
    availability = case.get("availability_flags", {})
    if not isinstance(observed, dict) or not isinstance(availability, dict):
        raise RuntimeError("Invalid case evidence structure")
    return sorted(set(observed) | set(availability) | {"contraindications"})


def build_review_schema(case: dict[str, Any]) -> dict[str, Any]:
    actions = case.get("candidate_actions", [])
    if not isinstance(actions, list) or not actions:
        raise RuntimeError("candidate_actions missing")

    evidence_ids = _allowed_evidence_ids(case)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reviews": {
                "type": "array",
                "minItems": len(actions),
                "maxItems": len(actions),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "action_id": {
                            "type": "string",
                            "enum": actions,
                            "description": "One of the supplied candidate actions.",
                        },
                        "relevance_score": {
                            "type": "integer",
                            "enum": [0, 1, 2, 3],
                        },
                        "abstain": {"type": "boolean"},
                        "evidence_ids": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": evidence_ids,
                            },
                        },
                        "rationale": {
                            "type": "string",
                            "minLength": 10,
                        },
                        "contraindication_detected": {"type": "boolean"},
                        "safety_flag": {"type": "boolean"},
                    },
                    "required": [
                        "action_id",
                        "relevance_score",
                        "abstain",
                        "evidence_ids",
                        "rationale",
                        "contraindication_detected",
                        "safety_flag",
                    ],
                },
            }
        },
        "required": ["reviews"],
    }


def build_request_body(
    case: dict[str, Any],
    prompt: str,
    response_schema: dict[str, Any],
) -> dict[str, Any]:
    user_payload = {
        "case": case,
        "instruction": (
            "Review this blinded case. Follow the system instruction and "
            "return exactly one review for every candidate action."
        ),
    }
    return {
        "systemInstruction": {
            "parts": [{"text": prompt}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": json.dumps(
                            user_payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "responseFormat": {
                "text": {
                    "mimeType": "application/json",
                    "schema": response_schema,
                }
            }
        },
    }


def _extract_structured_response(provider_response: dict[str, Any]) -> dict[str, Any]:
    candidates = provider_response.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise RuntimeError("Gemini response must contain exactly one candidate")

    candidate = candidates[0]
    finish_reason = str(candidate.get("finishReason", "")).strip()
    if finish_reason != "STOP":
        raise RuntimeError(
            f"Gemini candidate did not finish cleanly: finishReason={finish_reason!r}"
        )

    parts = candidate.get("content", {}).get("parts", [])
    if not isinstance(parts, list) or not parts:
        raise RuntimeError("Gemini response contains no text parts")

    text = "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict)
    ).strip()
    if not text:
        raise RuntimeError("Gemini structured response text is empty")

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise RuntimeError("Gemini structured response root is not an object")
    return parsed


def _validate_semantics(
    case: dict[str, Any],
    parsed: dict[str, Any],
) -> list[dict[str, Any]]:
    reviews = parsed.get("reviews")
    if not isinstance(reviews, list):
        raise RuntimeError("reviews is not an array")

    expected = list(case["candidate_actions"])
    returned = [review.get("action_id") for review in reviews if isinstance(review, dict)]

    if len(reviews) != len(expected):
        raise RuntimeError(
            f"Review count mismatch: expected {len(expected)}, got {len(reviews)}"
        )
    if len(returned) != len(reviews) or len(set(returned)) != len(returned):
        raise RuntimeError("Duplicate or malformed action_id in Gemini response")
    if set(returned) != set(expected):
        raise RuntimeError(
            f"Action set mismatch: expected={sorted(expected)}, returned={sorted(returned)}"
        )

    allowed_evidence = set(_allowed_evidence_ids(case))
    for review in reviews:
        evidence_ids = review.get("evidence_ids")
        if not isinstance(evidence_ids, list):
            raise RuntimeError("evidence_ids must be an array")
        unknown = set(evidence_ids) - allowed_evidence
        if unknown:
            raise RuntimeError(f"Unknown evidence ids: {sorted(unknown)}")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise RuntimeError("Duplicate evidence_ids are not allowed")
        if not review.get("abstain", False) and not evidence_ids:
            raise RuntimeError("Non-abstained review must cite evidence")
        rationale = str(review.get("rationale", "")).strip()
        if len(rationale) < 10:
            raise RuntimeError("Rationale too short")

    # Canonicalize provider output ordering to candidate_actions ordering.
    # This keeps normalized record indexes/hashes stable even if the provider
    # returns the same valid action set in a different array order.
    by_action = {review["action_id"]: review for review in reviews}
    return [by_action[action_id] for action_id in expected]


def _patch_importer_for_smoke(
    smoke_raw: Path,
    smoke_imports: Path,
    smoke_envelopes: Path,
    smoke_audit: Path,
) -> dict[str, Any]:
    names = (
        "RAW_DIR",
        "IMPORTS_DIR",
        "ENVELOPE_ROOT",
        "CAPABILITY_AUDIT_PATH",
        "ACCEPTED_RECORDS_PATH",
        "REJECTED_RECORDS_PATH",
    )
    original = {name: getattr(imp, name) for name in names}

    try:
        imp.RAW_DIR = smoke_raw
        imp.IMPORTS_DIR = smoke_imports
        imp.ENVELOPE_ROOT = smoke_envelopes
        imp.CAPABILITY_AUDIT_PATH = smoke_audit
        imp.ACCEPTED_RECORDS_PATH = smoke_imports / "accepted_records.parquet"
        imp.REJECTED_RECORDS_PATH = smoke_imports / "rejected_records.jsonl"

        return imp.import_annotations(
            raw_dir=smoke_raw,
            output_file=imp.ACCEPTED_RECORDS_PATH,
        )
    finally:
        for name, value in original.items():
            setattr(imp, name, value)


def _make_smoke_audit(path: Path, model: str, response_id: str) -> None:
    audit = {
        "audit_version": "external_llm_provider_smoke_v3",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "external_provider_status": "AVAILABLE",
        "scientific_status": "SMOKE_VERIFIED_NOT_CANONICAL",
        "verified_independent_source_count": 1,
        "environment_audit": {
            "GEMINI_API_KEY": True,
        },
        "evaluated_providers": [
            {
                "provider_name": PROVIDER,
                "connector_or_sdk": "Gemini REST generateContent",
                "authentication_available": True,
                "model_endpoint": model,
                "provider_response_id_observed": bool(response_id),
                "supports_raw_response_capture": True,
                "status": "AVAILABLE",
            }
        ],
    }
    path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_smoke(model: str, execute: bool) -> int:
    case = _load_first_case()
    prompt = PROMPT_PATH.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    prompt_hash = canonical_text_sha256(PROMPT_PATH)
    schema = build_review_schema(case)
    body = build_request_body(case, prompt, schema)
    raw_request = canonical_json_bytes(body)

    snapshot_bytes = (
        json.dumps(
            case,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")

    print(f"MODEL={model}")
    print(f"CASE_ID={case['case_id']}")
    print(f"PANEL_ID={case['panel_id']}")
    print(f"CANDIDATE_ACTIONS={case['candidate_actions']}")
    print(f"PROMPT_SHA256={prompt_hash}")
    print(f"RESPONSE_SCHEMA_SHA256={sha256_bytes(canonical_json_bytes(schema))}")
    print(f"REQUEST_BATCH_SHA256={sha256_bytes(snapshot_bytes)}")
    print("SAMPLING_CONFIG=MODEL_DEFAULTS")
    print("API_KEY_PERSISTED_TO_DISK=FALSE")

    if not execute:
        print("SMOKE_DRY_RUN=PASS")
        print("EXTERNAL_API_CALLS_MADE=FALSE")
        return 0

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")

    if SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)

    batch_id = "panel_a_batch_01_smoke_v3"
    envelope_root = SMOKE_ROOT / "external_reviews"
    batch_dir = envelope_root / PROVIDER / batch_id
    raw_request_dir = batch_dir / "raw_requests"
    raw_response_dir = batch_dir / "raw_responses"
    smoke_raw = SMOKE_ROOT / "imports" / "raw"
    smoke_imports = SMOKE_ROOT / "imports"
    raw_request_dir.mkdir(parents=True)
    raw_response_dir.mkdir(parents=True)
    smoke_raw.mkdir(parents=True)

    snapshot_path = batch_dir / "request_batch_snapshot.jsonl"
    snapshot_path.write_bytes(snapshot_bytes)
    request_batch_sha = sha256_bytes(snapshot_bytes)

    request_id = f"clientreq_{uuid.uuid4().hex}"
    raw_request_sha = sha256_bytes(raw_request)
    raw_request_name = f"{case['case_id']}.json"
    raw_request_path = raw_request_dir / raw_request_name
    raw_request_path.write_bytes(raw_request)

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    request = urllib.request.Request(
        endpoint,
        data=raw_request,
        method="POST",
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw_response = response.read()
            http_status = response.status
    except urllib.error.HTTPError as exc:
        error_body = exc.read()
        raise RuntimeError(
            f"Gemini HTTP {exc.code}: {error_body.decode('utf-8', errors='replace')}"
        ) from exc

    raw_response_sha = sha256_bytes(raw_response)
    raw_response_name = f"{case['case_id']}.json"
    raw_response_path = raw_response_dir / raw_response_name
    raw_response_path.write_bytes(raw_response)

    provider_response = json.loads(raw_response.decode("utf-8"))
    response_id = str(provider_response.get("responseId", "")).strip()
    model_version = str(provider_response.get("modelVersion", "")).strip()
    if not response_id:
        raise RuntimeError("Provider responseId missing")
    if not model_version:
        raise RuntimeError("Provider modelVersion missing")

    parsed = _extract_structured_response(provider_response)
    reviews = _validate_semantics(case, parsed)

    candidate = provider_response["candidates"][0]
    finish_reason = str(candidate.get("finishReason", "")).strip()
    usage_metadata = provider_response.get("usageMetadata", {})
    if not isinstance(usage_metadata, dict):
        usage_metadata = {}

    created_at = datetime.now(timezone.utc).isoformat()
    reviewer_config_payload = {
        "provider": PROVIDER,
        "requested_model": model,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "response_schema_sha256": sha256_bytes(canonical_json_bytes(schema)),
        "sampling": "model_defaults",
        "api": "generateContent",
    }
    reviewer_config_hash = sha256_bytes(canonical_json_bytes(reviewer_config_payload))
    reviewer_config_id = f"gemini_external_{reviewer_config_hash[:16]}"

    normalized: list[dict[str, Any]] = []
    response_env_records: list[dict[str, Any]] = []

    for index, review in enumerate(reviews):
        record = {
            "case_id": case["case_id"],
            "panel_id": case["panel_id"],
            "action_id": review["action_id"],
            "relevance_score": int(review["relevance_score"]),
            "abstain": bool(review["abstain"]),
            "evidence_ids": list(review["evidence_ids"]),
            "rationale": str(review["rationale"]),
            "contraindication_detected": bool(review["contraindication_detected"]),
            "safety_flag": bool(review["safety_flag"]),
            "reviewer_id": "gemini_external_reviewer_01",
            "reviewer_configuration_id": reviewer_config_id,
            "reviewer_type": "REAL_EXTERNAL_LLM_REVIEW",
            "provider": PROVIDER,
            "model_name": model,
            "model_version": model_version,
            "request_id": request_id,
            "response_id": response_id,
            "batch_id": batch_id,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": prompt_hash,
            "request_batch_sha256": request_batch_sha,
            "raw_request_sha256": raw_request_sha,
            "raw_response_sha256": raw_response_sha,
            "response_record_index": index,
            "created_at": created_at,
        }
        record["response_record_sha256"] = response_record_sha256(record)
        normalized.append(record)

        response_env_records.append(
            {
                "index": index,
                "case_id": case["case_id"],
                "request_id": request_id,
                "response_id": response_id,
                "action_id": record["action_id"],
                "sha256": record["response_record_sha256"],
                "raw_response_sha256": raw_response_sha,
                "raw_response_file": f"raw_responses/{raw_response_name}",
                "model_version": model_version,
            }
        )

    request_envelope = {
        "provider": PROVIDER,
        "model_name": model,
        "batch_id": batch_id,
        "records": [
            {
                "case_id": case["case_id"],
                "request_id": request_id,
                "raw_request_sha256": raw_request_sha,
                "raw_request_file": f"raw_requests/{raw_request_name}",
            }
        ],
    }
    response_envelope = {
        "provider": PROVIDER,
        "model_name": model,
        "batch_id": batch_id,
        "records": response_env_records,
    }
    manifest = {
        "schema_version": "gemini_external_review_batch_v3_1",
        "provider": PROVIDER,
        "model_name": model,
        "model_version": model_version,
        "batch_id": batch_id,
        "panel_id": case["panel_id"],
        "case_count": 1,
        "annotation_record_count": len(normalized),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "request_batch_sha256": request_batch_sha,
        "request_batch_snapshot_file": "request_batch_snapshot.jsonl",
        "source_batch_file": str(BATCH_FILE.relative_to(ROOT)).replace("\\", "/"),
        "source_batch_sha256": sha256_bytes(BATCH_FILE.read_bytes()),
        "response_schema_sha256": sha256_bytes(canonical_json_bytes(schema)),
        "reviewer_configuration_id": reviewer_config_id,
        "reviewer_configuration": reviewer_config_payload,
        "request_id_semantics": "client_generated_uuid4",
        "response_id_semantics": "provider_native_Gemini_responseId",
        "http_status": http_status,
        "finish_reason": finish_reason,
        "usage_metadata": usage_metadata,
        "endpoint": endpoint,
        "api_revision": "v1beta",
        "created_at": created_at,
    }

    (batch_dir / "request_envelope.json").write_text(
        json.dumps(request_envelope, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (batch_dir / "response_envelope.json").write_text(
        json.dumps(response_envelope, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (batch_dir / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    normalized_path = smoke_raw / "panel_a_batch_01_smoke_v3.jsonl"
    normalized_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in normalized) + "\n",
        encoding="utf-8",
    )

    smoke_audit = SMOKE_ROOT / "capability_audit_smoke.json"
    _make_smoke_audit(smoke_audit, model, response_id)

    first_import = _patch_importer_for_smoke(
        smoke_raw=smoke_raw,
        smoke_imports=smoke_imports,
        smoke_envelopes=envelope_root,
        smoke_audit=smoke_audit,
    )
    if (
        first_import.get("real_external_llm_review_count") != len(normalized)
        or first_import.get("invalid_count") != 0
    ):
        raise RuntimeError(f"Initial smoke import failed: {first_import}")

    original_response_bytes = raw_response_path.read_bytes()
    raw_response_path.write_bytes(b'{"tampered":true}\n')
    tampered_import = _patch_importer_for_smoke(
        smoke_raw=smoke_raw,
        smoke_imports=smoke_imports,
        smoke_envelopes=envelope_root,
        smoke_audit=smoke_audit,
    )
    if tampered_import.get("real_external_llm_review_count") != 0:
        raise RuntimeError(
            "Fail-closed tamper test failed: tampered response was accepted"
        )

    raw_response_path.write_bytes(original_response_bytes)
    restored_import = _patch_importer_for_smoke(
        smoke_raw=smoke_raw,
        smoke_imports=smoke_imports,
        smoke_envelopes=envelope_root,
        smoke_audit=smoke_audit,
    )
    if (
        restored_import.get("real_external_llm_review_count") != len(normalized)
        or restored_import.get("invalid_count") != 0
    ):
        raise RuntimeError(f"Restored smoke import failed: {restored_import}")

    checksums = []
    for path in sorted(p for p in batch_dir.rglob("*") if p.is_file()):
        if path.name == "checksums.sha256":
            continue
        rel = path.relative_to(batch_dir).as_posix()
        checksums.append(f"{sha256_bytes(path.read_bytes())}  {rel}")
    (batch_dir / "checksums.sha256").write_text(
        "\n".join(checksums) + "\n",
        encoding="utf-8",
    )

    print(f"HTTP_STATUS={http_status}")
    print(f"MODEL_VERSION={model_version}")
    print(f"PROVIDER_RESPONSE_ID={response_id}")
    print(f"REQUEST_ID={request_id}")
    print(f"REVIEW_COUNT={len(normalized)}")
    print("INITIAL_IMPORT=PASS")
    print("TAMPERED_RAW_RESPONSE_REJECTED=TRUE")
    print("RESTORED_IMPORT=PASS")
    print(f"SMOKE_ROOT={SMOKE_ROOT}")
    print("CANONICAL_IMPORT_ARTIFACTS_MODIFIED=FALSE")
    print("TRACKED_CAPABILITY_AUDIT_MODIFIED=FALSE")
    print("GEMINI_EXTERNAL_REVIEW_SMOKE_V3=PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Make exactly one real Gemini API request.",
    )
    args = parser.parse_args()
    return run_smoke(model=args.model, execute=args.execute)


if __name__ == "__main__":
    raise SystemExit(main())
