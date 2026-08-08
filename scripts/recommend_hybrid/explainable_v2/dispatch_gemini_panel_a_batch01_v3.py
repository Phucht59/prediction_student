"""Resume-safe production Gemini dispatcher for Panel A batch 01.

This script performs authentic external-review collection for the canonical
blinded request batch. It stores exact request/response bytes, record-level
provenance, and normalized annotations. It never persists API credentials.

Default mode is dry-run. Use --execute to make real API calls.
"""

from __future__ import annotations

import argparse
from collections import deque
import email.utils
import json
import os
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.explainable_v2.provenance import (
    canonical_json_bytes,
    canonical_text_sha256,
    response_record_sha256,
    sha256_bytes,
)
from src.recommend_hybrid.explainable_v2.provider_envelope import (
    verify_provider_envelope,
)

PROVIDER = "Google Gemini API"
PROMPT_VERSION = "external_reviewer_v1"
DEFAULT_MODEL = "gemini-3.6-flash"
BATCH_ID = "panel_a_batch_01"
EXPECTED_PANEL_ID = "PANEL_A"
EXPECTED_PANEL_CASE_COUNT = 300
MODEL_MIXING_POLICY = "EXPLICIT_GEMINI_3_5_FLASH_FAMILY_MIXED_REVIEWERS"
ALLOWED_REVIEW_MODELS = frozenset({"gemini-3.5-flash", "gemini-3.5-flash-lite"})
EXPECTED_CASE_EXPORT_CLASSIFICATION = "VERIFIED_OULAD_QUERY_LEVEL_LINEAGE_V4"
CASE_MANIFEST_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/annotations/exports/"
    "case_manifest.json"
)
QUARANTINE_ROOT = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/annotations/quarantine"
)
SAFE_RESPONSE_HEADER_NAMES = frozenset(
    {
        "content-type",
        "date",
        "x-goog-request-id",
        "x-request-id",
        "x-cloud-trace-context",
    }
)
MAX_REQUESTS_PER_MINUTE = 12
RATE_LIMIT_WINDOW_SECONDS = 60.0
RATE_LIMIT_SAFETY_SECONDS = 0.25
_REQUEST_TIMESTAMPS: deque[float] = deque()

PROMPT_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/annotations/prompts/"
    "locked_external_reviewer_prompt_v1.txt"
)
SOURCE_BATCH_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/annotations/prompts/"
    "panel_a_request_batches/batch_01.jsonl"
)
CAPABILITY_AUDIT_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/annotations/"
    "EXTERNAL_PROVIDER_CAPABILITY_AUDIT.json"
)
ENVELOPE_ROOT = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/annotations/external_reviews"
)
BATCH_DIR = ENVELOPE_ROOT / PROVIDER / BATCH_ID
RAW_REQUEST_DIR = BATCH_DIR / "raw_requests"
RAW_RESPONSE_DIR = BATCH_DIR / "raw_responses"
CASE_STATE_DIR = BATCH_DIR / "case_state"
NORMALIZED_PATH = BATCH_DIR / "normalized_records.jsonl"
REQUEST_ENVELOPE_PATH = BATCH_DIR / "request_envelope.json"
RESPONSE_ENVELOPE_PATH = BATCH_DIR / "response_envelope.json"
BATCH_MANIFEST_PATH = BATCH_DIR / "batch_manifest.json"
BATCH_SNAPSHOT_PATH = BATCH_DIR / "request_batch_snapshot.jsonl"
CHECKSUMS_PATH = BATCH_DIR / "checksums.sha256"

IMPORT_RAW_DIR = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/annotations/imports/raw"
)
IMPORT_RAW_PATH = IMPORT_RAW_DIR / "panel_a_batch_01_gemini.jsonl"



def validate_v4_source_gate(cases: list[dict[str, Any]]) -> None:
    manifest = json.loads(CASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("case_export_classification") != EXPECTED_CASE_EXPORT_CLASSIFICATION:
        raise RuntimeError(
            "Panel A source is not VERIFIED_OULAD_QUERY_LEVEL_LINEAGE_V4"
        )
    if manifest.get("query_level_evidence_invariant_across_actions") is not True:
        raise RuntimeError("Query-level evidence invariance is not verified")
    panel_count_field = f"{EXPECTED_PANEL_ID.lower()}_case_count"
    if int(manifest.get(panel_count_field, -1)) != EXPECTED_PANEL_CASE_COUNT:
        raise RuntimeError(
            f"V4 case manifest {panel_count_field} is not "
            f"{EXPECTED_PANEL_CASE_COUNT}"
        )
    if manifest.get("zero_student_overlap") is not True:
        raise RuntimeError("V4 Panel A/B student overlap gate failed")
    if manifest.get("zero_query_overlap") is not True:
        raise RuntimeError("V4 Panel A/B query overlap gate failed")
    if manifest.get("runtime_authorized") is not False:
        raise RuntimeError("V4 runtime_authorized must remain false")
    if len(cases) != 50:
        raise RuntimeError("V4 batch_01 must contain exactly 50 cases")

    allowed_top_level = {
        "case_id",
        "panel_id",
        "stage",
        "cutoff_day",
        "risk_band",
        "uncertainty_band",
        "routing_status",
        "observed_pre_cutoff_evidence",
        "candidate_actions",
        "availability_flags",
        "contraindications",
    }
    forbidden_recursive = {
        "query_id",
        "source_query_id",
        "student_key",
        "student_group_id",
        "source_student_group_id",
        "id_student",
        "course_key",
        "code_module",
        "code_presentation",
        "outer_fold",
    }

    def walk_keys(value: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(value, dict):
            for key, child in value.items():
                found.add(str(key))
                found.update(walk_keys(child))
        elif isinstance(value, list):
            for child in value:
                found.update(walk_keys(child))
        return found

    for case in cases:
        extras = set(case) - allowed_top_level
        if extras:
            raise RuntimeError(
                f"Unexpected public case keys for {case.get('case_id')}: {sorted(extras)}"
            )
        leaked = walk_keys(case) & forbidden_recursive
        if leaked:
            raise RuntimeError(
                f"Blinding violation for {case.get('case_id')}: {sorted(leaked)}"
            )


def assert_quarantine_is_not_resume_source() -> None:
    canonical = BATCH_DIR.resolve()
    quarantine = QUARANTINE_ROOT.resolve()
    try:
        canonical.relative_to(quarantine)
    except ValueError:
        pass
    else:
        raise RuntimeError("Canonical batch directory resolves inside quarantine")


def safe_response_headers(headers: Any) -> dict[str, str]:
    if headers is None:
        return {}
    result: dict[str, str] = {}
    for name in SAFE_RESPONSE_HEADER_NAMES:
        value = headers.get(name)
        if value is None:
            value = headers.get(name.title())
        if value is not None:
            result[name] = str(value)
    return result


def retry_after_seconds(headers: Any) -> float | None:
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if value is None:
        value = headers.get("retry-after")
    if value is None:
        return None
    raw = str(value).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        try:
            dt = email.utils.parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
        except Exception:
            return None


def wait_for_rate_limit_slot(
    *,
    timestamps: deque[float] | None = None,
    monotonic=time.monotonic,
    sleeper=time.sleep,
) -> None:
    """Enforce a conservative rolling limit for every HTTP attempt."""

    observed = _REQUEST_TIMESTAMPS if timestamps is None else timestamps
    while True:
        now = monotonic()
        while observed and now - observed[0] >= RATE_LIMIT_WINDOW_SECONDS:
            observed.popleft()
        if len(observed) < MAX_REQUESTS_PER_MINUTE:
            observed.append(now)
            return
        delay = (
            RATE_LIMIT_WINDOW_SECONDS
            - (now - observed[0])
            + RATE_LIMIT_SAFETY_SECONDS
        )
        print(f"RATE_LIMIT_WAIT_SECONDS={delay:.2f}")
        sleeper(max(delay, RATE_LIMIT_SAFETY_SECONDS))


def locked_model_versions_from_states(
    cases: list[dict[str, Any]],
) -> dict[str, str]:
    versions_by_model: dict[str, set[str]] = {}
    for case in cases:
        state = load_case_state(case["case_id"])
        if state is None or not completed_case_is_intact(state):
            continue
        model_name = str(state.get("model_name", "")).strip()
        model_version = str(state.get("model_version", "")).strip()
        if model_name not in ALLOWED_REVIEW_MODELS:
            raise RuntimeError(
                f"Existing canonical state uses unapproved review model: {model_name!r}"
            )
        if not model_version:
            raise RuntimeError(
                f"Existing canonical state for {model_name!r} has empty modelVersion"
            )
        versions_by_model.setdefault(model_name, set()).add(model_version)

    mixed_versions = {
        model_name: sorted(values)
        for model_name, values in versions_by_model.items()
        if len(values) > 1
    }
    if mixed_versions:
        raise RuntimeError(
            f"Multiple provider modelVersion values within same model: {mixed_versions}"
        )
    return {
        model_name: next(iter(values))
        for model_name, values in versions_by_model.items()
    }

def verify_completed_batch_before_import(
    *,
    normalized_records: list[dict[str, Any]],
    prompt_hash: str,
) -> None:
    if not normalized_records:
        raise RuntimeError("Cannot verify an empty completed batch")

    versions_by_model: dict[str, set[str]] = {}
    for record in normalized_records:
        model_name = str(record.get("model_name", "")).strip()
        model_version = str(record.get("model_version", "")).strip()
        if model_name not in ALLOWED_REVIEW_MODELS:
            raise RuntimeError(
                f"Completed batch contains unapproved model_name: {model_name!r}"
            )
        if not model_version:
            raise RuntimeError("Completed batch contains empty modelVersion")
        versions_by_model.setdefault(model_name, set()).add(model_version)

    bad = {
        model_name: sorted(values)
        for model_name, values in versions_by_model.items()
        if len(values) > 1
    }
    if bad:
        raise RuntimeError(
            f"Completed batch mixes provider versions within same model: {bad}"
        )

    for record in normalized_records:
        ok, code, message = verify_provider_envelope(
            ENVELOPE_ROOT,
            PROVIDER,
            BATCH_ID,
            record,
            locked_prompt_hash=prompt_hash,
        )
        if not ok:
            raise RuntimeError(
                f"Provider-envelope verification failed: {code}: {message}"
            )

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        SOURCE_BATCH_PATH.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case, dict):
            raise RuntimeError(f"Batch line {line_number} is not a JSON object")
        if case.get("panel_id") != EXPECTED_PANEL_ID:
            raise RuntimeError(
                f"Batch line {line_number} is not {EXPECTED_PANEL_ID}: "
                f"{case.get('panel_id')}"
            )
        actions = case.get("candidate_actions")
        if not isinstance(actions, list) or not actions:
            raise RuntimeError(f"Batch line {line_number} has no candidate_actions")
        cases.append(case)

    if len(cases) != 50:
        raise RuntimeError(f"Expected exactly 50 cases in batch_01, found {len(cases)}")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise RuntimeError("Duplicate case_id detected in batch_01")
    return cases


def allowed_evidence_ids(case: dict[str, Any]) -> list[str]:
    observed = case.get("observed_pre_cutoff_evidence", {})
    availability = case.get("availability_flags", {})
    if not isinstance(observed, dict) or not isinstance(availability, dict):
        raise RuntimeError("Invalid blinded case evidence structure")
    return sorted(set(observed) | set(availability) | {"contraindications"})


def build_review_schema(case: dict[str, Any]) -> dict[str, Any]:
    actions = list(case["candidate_actions"])
    evidence_ids = allowed_evidence_ids(case)
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
                    "mimeType": "APPLICATION_JSON",
                    "schema": response_schema,
                }
            }
        },
    }


def extract_reviews(
    case: dict[str, Any],
    provider_response: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    candidates = provider_response.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise RuntimeError("Gemini response must contain exactly one candidate")

    candidate = candidates[0]
    finish_reason = str(candidate.get("finishReason", "")).strip()
    if finish_reason != "STOP":
        raise RuntimeError(
            f"Gemini candidate did not finish cleanly: {finish_reason!r}"
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
        raise RuntimeError("Gemini structured response is empty")

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise RuntimeError("Structured response root is not an object")

    reviews = parsed.get("reviews")
    if not isinstance(reviews, list):
        raise RuntimeError("reviews is not an array")

    expected = list(case["candidate_actions"])
    returned = [
        review.get("action_id")
        for review in reviews
        if isinstance(review, dict)
    ]
    if len(reviews) != len(expected):
        raise RuntimeError(
            f"Review count mismatch: expected {len(expected)}, got {len(reviews)}"
        )
    if len(returned) != len(reviews) or len(set(returned)) != len(returned):
        raise RuntimeError("Duplicate or malformed action_id")
    if set(returned) != set(expected):
        raise RuntimeError(
            f"Action set mismatch: expected={expected}, returned={returned}"
        )

    allowed = set(allowed_evidence_ids(case))
    by_action: dict[str, dict[str, Any]] = {}
    for review in reviews:
        evidence_ids = review.get("evidence_ids")
        if not isinstance(evidence_ids, list):
            raise RuntimeError("evidence_ids must be an array")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise RuntimeError("Duplicate evidence_ids")
        unknown = set(evidence_ids) - allowed
        if unknown:
            raise RuntimeError(f"Unknown evidence_ids: {sorted(unknown)}")
        if not review.get("abstain", False) and not evidence_ids:
            raise RuntimeError("Non-abstained review must cite evidence")
        rationale = str(review.get("rationale", "")).strip()
        if len(rationale) < 10:
            raise RuntimeError("Rationale too short")
        by_action[review["action_id"]] = review

    usage = provider_response.get("usageMetadata", {})
    if not isinstance(usage, dict):
        usage = {}
    return [by_action[action] for action in expected], finish_reason, usage


def load_capability_audit() -> dict[str, Any]:
    data = json.loads(CAPABILITY_AUDIT_PATH.read_text(encoding="utf-8"))
    if data.get("external_provider_status") != "AVAILABLE":
        raise RuntimeError(
            "Capability audit is not AVAILABLE. Run the verified smoke "
            "capability-promotion step first."
        )
    providers = {
        item.get("provider_name"): item
        for item in data.get("evaluated_providers", [])
        if isinstance(item, dict)
    }
    gemini = providers.get(PROVIDER)
    if not gemini or gemini.get("status") != "AVAILABLE":
        raise RuntimeError("Google Gemini API is not AVAILABLE in capability audit")
    return data


def case_state_path(case_id: str) -> Path:
    return CASE_STATE_DIR / f"{case_id}.json"


def load_case_state(case_id: str) -> dict[str, Any] | None:
    path = case_state_path(case_id)
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("status") != "COMPLETE":
        return None
    return state


def save_case_state(case_id: str, state: dict[str, Any]) -> None:
    path = case_state_path(case_id)
    path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def completed_case_is_intact(state: dict[str, Any]) -> bool:
    request_path = BATCH_DIR / state["raw_request_file"]
    response_path = BATCH_DIR / state["raw_response_file"]
    if not request_path.is_file() or not response_path.is_file():
        return False
    if sha256_bytes(request_path.read_bytes()) != state["raw_request_sha256"]:
        return False
    if sha256_bytes(response_path.read_bytes()) != state["raw_response_sha256"]:
        return False
    records = state.get("normalized_records")
    return isinstance(records, list) and bool(records)


def request_with_retry(
    endpoint: str,
    api_key: str,
    raw_request: bytes,
    max_attempts: int,
    base_delay: float,
) -> tuple[bytes, int, dict[str, str]]:
    last_error: Exception | None = None
    retryable = {408, 429, 500, 502, 503, 504}

    for attempt in range(1, max_attempts + 1):
        wait_for_rate_limit_slot()
        request = urllib.request.Request(
            endpoint,
            data=raw_request,
            method="POST",
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
        )
        retry_after: float | None = None
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return (
                    response.read(),
                    response.status,
                    safe_response_headers(response.headers),
                )
        except urllib.error.HTTPError as exc:
            error_bytes = exc.read()
            body = error_bytes.decode("utf-8", errors="replace")
            if exc.code not in retryable:
                raise RuntimeError(f"Gemini HTTP {exc.code}: {body}") from exc
            retry_after = retry_after_seconds(exc.headers)
            last_error = RuntimeError(f"Gemini HTTP {exc.code}: {body}")
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc

        if attempt < max_attempts:
            exponential = min(base_delay * (2 ** (attempt - 1)), 60.0)
            delay = max(exponential, retry_after or 0.0)
            delay += random.uniform(0.0, min(1.0, max(delay, 0.1) * 0.1))
            print(
                f"RETRY attempt={attempt + 1}/{max_attempts} "
                f"after_seconds={delay:.2f}"
            )
            time.sleep(delay)

    raise RuntimeError(f"Gemini request failed after retries: {last_error}")

def rebuild_batch_artifacts(
    cases: list[dict[str, Any]],
    model: str,
    prompt_hash: str,
    request_batch_sha: str,
    source_batch_sha: str,
) -> tuple[int, int]:
    request_records: list[dict[str, Any]] = []
    response_records: list[dict[str, Any]] = []
    normalized_records: list[dict[str, Any]] = []
    completed_cases = 0

    global_index = 0
    model_versions: set[str] = set()
    model_names: set[str] = set()
    response_schema_hashes: set[str] = set()

    for case in cases:
        state = load_case_state(case["case_id"])
        if state is None or not completed_case_is_intact(state):
            continue

        completed_cases += 1
        state_model_name = str(state.get("model_name", "")).strip()
        if state_model_name not in ALLOWED_REVIEW_MODELS:
            raise RuntimeError(
                f"Unapproved review model in canonical state: {state_model_name!r}"
            )
        model_names.add(state_model_name)
        model_versions.add(state["model_version"])
        response_schema_hashes.add(state["response_schema_sha256"])

        request_records.append(
            {
                "case_id": case["case_id"],
                "request_id": state["request_id"],
                "raw_request_sha256": state["raw_request_sha256"],
                "raw_request_file": state["raw_request_file"],
            }
        )

        for record in state["normalized_records"]:
            canonical = dict(record)
            canonical["response_record_index"] = global_index
            canonical["response_record_sha256"] = response_record_sha256(canonical)
            normalized_records.append(canonical)
            response_records.append(
                {
                    "index": global_index,
                    "case_id": canonical["case_id"],
                    "request_id": canonical["request_id"],
                    "response_id": canonical["response_id"],
                    "action_id": canonical["action_id"],
                    "sha256": canonical["response_record_sha256"],
                    "raw_response_sha256": canonical["raw_response_sha256"],
                    "raw_response_file": state["raw_response_file"],
                    "model_version": canonical["model_version"],
                }
            )
            global_index += 1

    if not model_names.issubset(ALLOWED_REVIEW_MODELS):
        raise RuntimeError(
            f"Unexpected review models in batch: {sorted(model_names)}"
        )

    envelope_model_name = (
        next(iter(model_names))
        if len(model_names) == 1
        else "MIXED_GEMINI_3_5_FLASH_FAMILY"
    )

    request_envelope = {
        "provider": PROVIDER,
        "model_name": envelope_model_name,
        "model_names_observed": sorted(model_names),
        "model_mixing_policy": MODEL_MIXING_POLICY,
        "batch_id": BATCH_ID,
        "records": request_records,
    }
    response_envelope = {
        "provider": PROVIDER,
        "model_name": envelope_model_name,
        "model_names_observed": sorted(model_names),
        "model_mixing_policy": MODEL_MIXING_POLICY,
        "batch_id": BATCH_ID,
        "records": response_records,
    }
    manifest = {
        "schema_version": "gemini_external_review_batch_v3_1",
        "provider": PROVIDER,
        "model_name": envelope_model_name,
        "model_names_observed": sorted(model_names),
        "model_versions_observed": sorted(model_versions),
        "model_mixing_policy": MODEL_MIXING_POLICY,
        "batch_id": BATCH_ID,
        "panel_id": EXPECTED_PANEL_ID,
        "source_case_count": len(cases),
        "completed_case_count": completed_cases,
        "annotation_record_count": len(normalized_records),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "request_batch_sha256": request_batch_sha,
        "request_batch_snapshot_file": "request_batch_snapshot.jsonl",
        "source_batch_file": str(SOURCE_BATCH_PATH.relative_to(ROOT)).replace(
            "\\",
            "/",
        ),
        "source_batch_sha256": source_batch_sha,
        "response_schema_sha256_values": sorted(response_schema_hashes),
        "request_id_semantics": "client_generated_uuid4",
        "response_id_semantics": "provider_native_Gemini_responseId",
        "sampling": "model_defaults",
        "endpoint": (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        ),
        "updated_at": utc_now(),
        "complete": completed_cases == len(cases),
    }

    REQUEST_ENVELOPE_PATH.write_text(
        json.dumps(request_envelope, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    RESPONSE_ENVELOPE_PATH.write_text(
        json.dumps(response_envelope, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    BATCH_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    NORMALIZED_PATH.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in normalized_records
        )
        + ("\n" if normalized_records else ""),
        encoding="utf-8",
    )

    if completed_cases == len(cases):
        checksum_lines: list[str] = []
        for path in sorted(p for p in BATCH_DIR.rglob("*") if p.is_file()):
            if path.name == "checksums.sha256":
                continue
            rel = path.relative_to(BATCH_DIR).as_posix()
            checksum_lines.append(f"{sha256_bytes(path.read_bytes())}  {rel}")
        CHECKSUMS_PATH.write_text(
            "\n".join(checksum_lines) + "\n",
            encoding="utf-8",
        )

        verify_completed_batch_before_import(
            normalized_records=normalized_records,
            prompt_hash=prompt_hash,
        )
        IMPORT_RAW_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(NORMALIZED_PATH, IMPORT_RAW_PATH)

    return completed_cases, len(normalized_records)


def dispatch(
    model: str,
    execute: bool,
    limit: int | None,
    max_attempts: int,
    base_delay: float,
    inter_request_delay: float,
) -> int:
    cases = load_cases()
    if model not in ALLOWED_REVIEW_MODELS:
        raise RuntimeError(
            f"Requested model {model!r} is outside approved review family: "
            f"{sorted(ALLOWED_REVIEW_MODELS)}"
        )
    validate_v4_source_gate(cases)
    assert_quarantine_is_not_resume_source()
    prompt = PROMPT_PATH.read_text(encoding="utf-8").replace(
        "\r\n",
        "\n",
    ).replace("\r", "\n")
    prompt_hash = canonical_text_sha256(PROMPT_PATH)

    source_batch_bytes = SOURCE_BATCH_PATH.read_bytes()
    source_batch_sha = sha256_bytes(source_batch_bytes)

    print(f"PROVIDER={PROVIDER}")
    print(f"MODEL={model}")
    print(f"BATCH_ID={BATCH_ID}")
    print(f"SOURCE_CASE_COUNT={len(cases)}")
    print(f"PROMPT_SHA256={prompt_hash}")
    print(f"SOURCE_BATCH_SHA256={source_batch_sha}")
    print("SAMPLING_CONFIG=MODEL_DEFAULTS")
    print("API_KEY_PERSISTED_TO_DISK=FALSE")

    if not execute:
        print("PRODUCTION_DISPATCH_DRY_RUN=PASS")
        print("EXTERNAL_API_CALLS_MADE=FALSE")
        return 0

    load_capability_audit()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    RAW_REQUEST_DIR.mkdir(parents=True, exist_ok=True)
    RAW_RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    CASE_STATE_DIR.mkdir(parents=True, exist_ok=True)

    if BATCH_SNAPSHOT_PATH.exists():
        existing = BATCH_SNAPSHOT_PATH.read_bytes()
        if existing != source_batch_bytes:
            raise RuntimeError(
                "Existing request_batch_snapshot.jsonl differs from source "
                "batch. Refusing to mix annotation runs."
            )
    else:
        BATCH_SNAPSHOT_PATH.write_bytes(source_batch_bytes)

    request_batch_sha = sha256_bytes(BATCH_SNAPSHOT_PATH.read_bytes())
    if request_batch_sha != source_batch_sha:
        raise RuntimeError("Exact batch snapshot hash differs from source batch hash")

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )

    new_requests = 0
    skipped_complete = 0
    locked_model_versions = locked_model_versions_from_states(cases)

    for case_number, case in enumerate(cases, 1):
        case_id = case["case_id"]
        state = load_case_state(case_id)
        if state is not None:
            if not completed_case_is_intact(state):
                raise RuntimeError(
                    f"Completed state for {case_id} failed integrity check"
                )
            skipped_complete += 1
            print(
                f"[{case_number:02d}/50] SKIP_COMPLETE "
                f"case_id={case_id}"
            )
            continue

        if limit is not None and new_requests >= limit:
            break

        schema = build_review_schema(case)
        schema_sha = sha256_bytes(canonical_json_bytes(schema))
        body = build_request_body(case, prompt, schema)
        raw_request = canonical_json_bytes(body)
        raw_request_sha = sha256_bytes(raw_request)

        request_id = f"clientreq_{uuid.uuid4().hex}"
        request_filename = f"{case_id}.json"
        response_filename = f"{case_id}.json"
        request_path = RAW_REQUEST_DIR / request_filename
        response_path = RAW_RESPONSE_DIR / response_filename
        request_path.write_bytes(raw_request)

        print(
            f"[{case_number:02d}/50] REQUEST "
            f"case_id={case_id} actions={len(case['candidate_actions'])}"
        )

        raw_response, http_status, response_headers = request_with_retry(
            endpoint=endpoint,
            api_key=api_key,
            raw_request=raw_request,
            max_attempts=max_attempts,
            base_delay=base_delay,
        )
        response_path.write_bytes(raw_response)
        raw_response_sha = sha256_bytes(raw_response)

        provider_response = json.loads(raw_response.decode("utf-8"))
        response_id = str(provider_response.get("responseId", "")).strip()
        model_version = str(provider_response.get("modelVersion", "")).strip()
        if not response_id:
            raise RuntimeError(f"Gemini responseId missing for {case_id}")
        if not model_version:
            raise RuntimeError(f"Gemini modelVersion missing for {case_id}")
        locked_version = locked_model_versions.get(model)
        if locked_version is None:
            locked_model_versions[model] = model_version
        elif model_version != locked_version:
            raise RuntimeError(
                f"Gemini modelVersion changed within model {model!r}: "
                f"{locked_version!r} -> {model_version!r}"
            )

        reviews, finish_reason, usage = extract_reviews(case, provider_response)

        reviewer_config_payload = {
            "provider": PROVIDER,
            "requested_model": model,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": prompt_hash,
            "response_schema_sha256": schema_sha,
            "sampling": "model_defaults",
            "api": "generateContent",
        }
        config_hash = sha256_bytes(canonical_json_bytes(reviewer_config_payload))
        reviewer_config_id = f"gemini_external_{config_hash[:16]}"
        created_at = utc_now()

        state_records: list[dict[str, Any]] = []
        for local_index, review in enumerate(reviews):
            record = {
                "case_id": case_id,
                "panel_id": case["panel_id"],
                "action_id": review["action_id"],
                "relevance_score": int(review["relevance_score"]),
                "abstain": bool(review["abstain"]),
                "evidence_ids": list(review["evidence_ids"]),
                "rationale": str(review["rationale"]),
                "contraindication_detected": bool(
                    review["contraindication_detected"]
                ),
                "safety_flag": bool(review["safety_flag"]),
                "reviewer_id": "gemini_external_reviewer_01",
                "reviewer_configuration_id": reviewer_config_id,
                "reviewer_type": "REAL_EXTERNAL_LLM_REVIEW",
                "provider": PROVIDER,
                "model_name": model,
                "model_version": model_version,
                "request_id": request_id,
                "response_id": response_id,
                "batch_id": BATCH_ID,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": prompt_hash,
                "request_batch_sha256": request_batch_sha,
                "raw_request_sha256": raw_request_sha,
                "raw_response_sha256": raw_response_sha,
                "response_record_index": local_index,
                "created_at": created_at,
            }
            record["response_record_sha256"] = response_record_sha256(record)
            state_records.append(record)

        state_payload = {
            "schema_version": "gemini_external_review_case_state_v3_1",
            "status": "COMPLETE",
            "case_number": case_number,
            "case_id": case_id,
            "provider": PROVIDER,
            "model_name": model,
            "model_version": model_version,
            "request_id": request_id,
            "response_id": response_id,
            "http_status": http_status,
            "response_headers": response_headers,
            "finish_reason": finish_reason,
            "usage_metadata": usage,
            "response_schema_sha256": schema_sha,
            "raw_request_sha256": raw_request_sha,
            "raw_response_sha256": raw_response_sha,
            "raw_request_file": f"raw_requests/{request_filename}",
            "raw_response_file": f"raw_responses/{response_filename}",
            "normalized_records": state_records,
            "created_at": created_at,
        }
        save_case_state(case_id, state_payload)
        new_requests += 1

        print(
            f"[{case_number:02d}/50] COMPLETE "
            f"response_id={response_id} reviews={len(reviews)}"
        )

        rebuild_batch_artifacts(
            cases=cases,
            model=model,
            prompt_hash=prompt_hash,
            request_batch_sha=request_batch_sha,
            source_batch_sha=source_batch_sha,
        )

        if inter_request_delay > 0:
            time.sleep(inter_request_delay)

    completed_cases, annotation_count = rebuild_batch_artifacts(
        cases=cases,
        model=model,
        prompt_hash=prompt_hash,
        request_batch_sha=request_batch_sha,
        source_batch_sha=source_batch_sha,
    )

    print(f"NEW_API_REQUESTS={new_requests}")
    print(f"SKIPPED_COMPLETE_CASES={skipped_complete}")
    print(f"COMPLETED_CASES={completed_cases}/50")
    print(f"ANNOTATION_RECORDS={annotation_count}")
    print(f"BATCH_DIR={BATCH_DIR}")

    if completed_cases == 50:
        print(f"IMPORT_RAW_PATH={IMPORT_RAW_PATH}")
        print("PANEL_A_BATCH01_DISPATCH=COMPLETE")
    else:
        print("PANEL_A_BATCH01_DISPATCH=PARTIAL_RESUME_SAFE")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of NEW provider requests this invocation.",
    )
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--base-delay", type=float, default=2.0)
    parser.add_argument(
        "--inter-request-delay",
        type=float,
        default=5.1,
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be >= 1")
    if args.base_delay < 0 or args.inter_request_delay < 0:
        raise SystemExit("Delay values must be >= 0")

    return dispatch(
        model=args.model,
        execute=args.execute,
        limit=args.limit,
        max_attempts=args.max_attempts,
        base_delay=args.base_delay,
        inter_request_delay=args.inter_request_delay,
    )


if __name__ == "__main__":
    raise SystemExit(main())
