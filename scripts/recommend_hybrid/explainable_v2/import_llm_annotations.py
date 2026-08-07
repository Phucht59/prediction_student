"""Import, validate, and normalize authentic external LLM reviews.

Production acceptance is fail-closed:
- capability audit must mark at least one provider AVAILABLE,
- every record must satisfy the strict external-review schema,
- prompt hash/version must match the tracked locked prompt,
- case/panel/action/evidence must match public blinded exports,
- every accepted record must pass provider-envelope verification,
- raw request/response hashes must refer to exact captured payload bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.explainable_v2.independence_audit import compute_source_independence_audit
from src.recommend_hybrid.explainable_v2.provenance import canonical_text_sha256, is_sha256
from src.recommend_hybrid.explainable_v2.provider_envelope import verify_provider_envelope

EXPORT_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports"
CAPABILITY_AUDIT_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/annotations/EXTERNAL_PROVIDER_CAPABILITY_AUDIT.json"
)
RAW_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/imports/raw"
IMPORTS_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/imports"
ENVELOPE_ROOT = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/external_reviews"
LOCKED_PROMPT_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/annotations/prompts/"
    "locked_external_reviewer_prompt_v1.txt"
)

ACCEPTED_RECORDS_PATH = IMPORTS_DIR / "accepted_records.parquet"
REJECTED_RECORDS_PATH = IMPORTS_DIR / "rejected_records.jsonl"

LOCKED_PROMPT_VERSION = "external_reviewer_v1"

FORBIDDEN_TYPES = {
    "REAL_LLM_GENERATED_REVIEW",
    "AGENT_GENERATED_PSEUDO_REVIEW",
    "RULE_DERIVED_LABEL",
}
FORBIDDEN_MODELS = {
    "Antigravity-LLM-v2-ReviewerA",
    "Antigravity-LLM-v2-ReviewerB",
    "Antigravity-LLM-v2-ReviewerC",
    "ANTIGRAVITY_INTERNAL_RULE_AGENT",
}

KNOWN_ACTIONS = {
    "ASSESSMENT_COMPLETION",
    "RECOVER_ENGAGEMENT",
    "STUDY_REGULARITY",
    "TARGETED_CONTENT_REVIEW",
    "QUIZ_RETRIEVAL_PRACTICE",
}

REQUIRED_SCHEMA_FIELDS = {
    "case_id",
    "panel_id",
    "action_id",
    "relevance_score",
    "abstain",
    "evidence_ids",
    "rationale",
    "contraindication_detected",
    "safety_flag",
    "reviewer_id",
    "reviewer_configuration_id",
    "reviewer_type",
    "provider",
    "model_name",
    "request_id",
    "response_id",
    "batch_id",
    "prompt_version",
    "prompt_sha256",
    "request_batch_sha256",
    "raw_request_sha256",
    "raw_response_sha256",
    "response_record_index",
    "response_record_sha256",
    "created_at",
}

OUTPUT_COLUMNS = [
    "case_id",
    "panel_id",
    "action_id",
    "relevance_score",
    "abstain",
    "evidence_ids",
    "rationale",
    "contraindication_detected",
    "safety_flag",
    "reviewer_id",
    "reviewer_configuration_id",
    "reviewer_type",
    "provider",
    "model_name",
    "model_version",
    "request_id",
    "response_id",
    "batch_id",
    "prompt_version",
    "prompt_sha256",
    "request_batch_sha256",
    "raw_request_sha256",
    "raw_response_sha256",
    "response_record_index",
    "response_record_sha256",
    "created_at",
    "source_record_sha256",
    "eligible_for_final_snorkel",
    "classification",
]


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_timestamp(value: Any) -> bool:
    if not _is_nonempty_string(value):
        return False
    text = value.strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return False
    return dt.tzinfo is not None


def _load_capability_audit() -> tuple[str, set[str], str | None]:
    if not CAPABILITY_AUDIT_PATH.exists():
        return "UNAVAILABLE", set(), "Capability audit file missing"
    try:
        data = json.loads(CAPABILITY_AUDIT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return "UNAVAILABLE", set(), f"Capability audit malformed: {exc}"
    if not isinstance(data, dict):
        return "UNAVAILABLE", set(), "Capability audit root is not a JSON object"

    status = str(data.get("external_provider_status", "UNAVAILABLE"))
    providers = {
        str(item.get("provider_name"))
        for item in data.get("evaluated_providers", [])
        if isinstance(item, dict)
        and item.get("status") == "AVAILABLE"
        and _is_nonempty_string(item.get("provider_name"))
    }
    if status == "AVAILABLE" and not providers:
        return "UNAVAILABLE", set(), "Capability audit says AVAILABLE but lists no provider"
    return status, providers, None


def _load_case_registry() -> tuple[
    set[str],
    dict[str, str],
    dict[str, list[str]],
    dict[str, set[str]],
]:
    known_cases: set[str] = set()
    case_panels: dict[str, str] = {}
    case_candidate_actions: dict[str, list[str]] = {}
    case_allowed_evidence_ids: dict[str, set[str]] = {}

    for panel_file in ("panel_a_cases.jsonl", "panel_b_cases.jsonl"):
        path = EXPORT_DIR / panel_file
        if not path.exists():
            raise RuntimeError(f"Required blinded export missing: {path}")

        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"{path.name}:{line_number}: malformed JSON: {exc}") from exc

            case_id = case.get("case_id")
            panel_id = case.get("panel_id")
            actions = case.get("candidate_actions")
            if not _is_nonempty_string(case_id):
                raise RuntimeError(f"{path.name}:{line_number}: case_id missing")
            if panel_id not in {"PANEL_A", "PANEL_B"}:
                raise RuntimeError(f"{path.name}:{line_number}: invalid panel_id")
            if not isinstance(actions, list) or not actions:
                raise RuntimeError(f"{path.name}:{line_number}: candidate_actions missing")

            observed = case.get("observed_pre_cutoff_evidence", {})
            availability = case.get("availability_flags", {})
            if not isinstance(observed, dict) or not isinstance(availability, dict):
                raise RuntimeError(f"{path.name}:{line_number}: evidence structure invalid")

            known_cases.add(case_id)
            case_panels[case_id] = panel_id
            case_candidate_actions[case_id] = list(actions)
            case_allowed_evidence_ids[case_id] = (
                set(observed.keys()) | set(availability.keys()) | {"contraindications"}
            )

    return known_cases, case_panels, case_candidate_actions, case_allowed_evidence_ids


def validate_record(
    rec: dict[str, Any],
    known_cases: set[str] | None = None,
    case_panels: dict[str, str] | None = None,
    case_candidate_actions: dict[str, list[str]] | None = None,
    case_allowed_evidence_ids: dict[str, set[str]] | None = None,
    approved_providers: set[str] | None = None,
    locked_prompt_hash: str | None = None,
    locked_prompt_version: str | None = None,
    known_actions: set[str] | None = None,
    envelope_registry: dict | None = None,
    envelope_root: Path = ENVELOPE_ROOT,
) -> tuple[bool, str, str]:
    """Validate one normalized external-review record."""
    if not isinstance(rec, dict):
        return False, "MALFORMED_JSON", "Record is not a JSON object"

    # Diagnostic-precedence checks.
    #
    # These checks do NOT weaken the strict full-schema gate below. They only
    # preserve precise rejection codes for malformed/partial adversarial
    # records, so a record with an obvious semantic/provenance violation is
    # diagnosed by that violation before the generic missing-field error.
    provider = rec.get("provider")
    if (
        approved_providers is not None
        and _is_nonempty_string(provider)
        and provider not in approved_providers
    ):
        return False, "UNAPPROVED_PROVIDER", f"Provider '{provider}' is not approved"

    if "request_id" in rec and not _is_nonempty_string(rec.get("request_id")):
        return False, "MISSING_REQUEST_ID", "request_id is missing or empty"

    if (
        locked_prompt_hash is not None
        and "prompt_sha256" in rec
        and _is_nonempty_string(rec.get("prompt_sha256"))
        and rec.get("prompt_sha256") != locked_prompt_hash
    ):
        return False, "PROMPT_HASH_MISMATCH", "prompt_sha256 does not match locked prompt"

    case_id_pre = rec.get("case_id")
    if (
        known_cases is not None
        and _is_nonempty_string(case_id_pre)
        and case_id_pre not in known_cases
    ):
        return False, "UNKNOWN_CASE_ID", f"Unknown case_id '{case_id_pre}'"

    panel_id_pre = rec.get("panel_id")
    if (
        case_panels is not None
        and _is_nonempty_string(case_id_pre)
        and case_id_pre in case_panels
        and _is_nonempty_string(panel_id_pre)
        and case_panels[case_id_pre] != panel_id_pre
    ):
        return False, "PANEL_MISMATCH", f"panel_id '{panel_id_pre}' does not match case registry"

    action_id_pre = rec.get("action_id")
    if (
        case_candidate_actions is not None
        and _is_nonempty_string(case_id_pre)
        and case_id_pre in case_candidate_actions
        and _is_nonempty_string(action_id_pre)
        and action_id_pre not in case_candidate_actions[case_id_pre]
    ):
        return (
            False,
            "INELIGIBLE_ACTION",
            f"action_id '{action_id_pre}' not in {case_candidate_actions[case_id_pre]}",
        )

    if "relevance_score" in rec:
        relevance_pre = rec.get("relevance_score")
        if type(relevance_pre) is not int or relevance_pre not in (0, 1, 2, 3):
            return False, "INVALID_RELEVANCE_SCORE", "relevance_score must be integer 0..3"

    if "rationale" in rec and not str(rec.get("rationale", "")).strip():
        return False, "EMPTY_RATIONALE", "rationale is empty"

    missing = sorted(field for field in REQUIRED_SCHEMA_FIELDS if field not in rec)
    if missing:
        return False, "MISSING_REQUIRED_FIELD", f"Missing required fields: {missing}"

    for field in (
        "case_id",
        "panel_id",
        "action_id",
        "reviewer_id",
        "reviewer_configuration_id",
        "reviewer_type",
        "provider",
        "model_name",
        "request_id",
        "response_id",
        "batch_id",
        "prompt_version",
        "prompt_sha256",
        "request_batch_sha256",
        "raw_request_sha256",
        "raw_response_sha256",
        "response_record_sha256",
        "created_at",
    ):
        if not _is_nonempty_string(rec.get(field)):
            return False, "EMPTY_REQUIRED_FIELD", f"{field} is empty"

    reviewer_type = rec["reviewer_type"]
    model_name = rec["model_name"]
    provider = rec["provider"]
    case_id = rec["case_id"]
    panel_id = rec["panel_id"]
    action_id = rec["action_id"]

    if reviewer_type in FORBIDDEN_TYPES or model_name in FORBIDDEN_MODELS:
        return (
            False,
            "FORBIDDEN_TYPE_OR_MODEL",
            f"Forbidden reviewer_type '{reviewer_type}' or model '{model_name}'",
        )
    if reviewer_type != "REAL_EXTERNAL_LLM_REVIEW":
        return False, "INVALID_REVIEWER_TYPE", "reviewer_type must be REAL_EXTERNAL_LLM_REVIEW"

    if approved_providers is not None and provider not in approved_providers:
        return False, "UNAPPROVED_PROVIDER", f"Provider '{provider}' is not approved"

    if panel_id not in {"PANEL_A", "PANEL_B"}:
        return False, "INVALID_PANEL_ID", f"Invalid panel_id '{panel_id}'"

    if known_cases is not None and case_id not in known_cases:
        return False, "UNKNOWN_CASE_ID", f"Unknown case_id '{case_id}'"

    if case_panels is not None and case_id in case_panels:
        if case_panels[case_id] != panel_id:
            return False, "PANEL_MISMATCH", f"panel_id '{panel_id}' does not match case registry"

    actions = known_actions if known_actions is not None else KNOWN_ACTIONS
    if action_id not in actions:
        return False, "UNKNOWN_ACTION_ID", f"Unknown action_id '{action_id}'"

    if case_candidate_actions is not None and case_id in case_candidate_actions:
        candidates = case_candidate_actions[case_id]
        if action_id not in candidates:
            return False, "INELIGIBLE_ACTION", f"action_id '{action_id}' not in {candidates}"

    relevance = rec.get("relevance_score")
    if type(relevance) is not int or relevance not in (0, 1, 2, 3):
        return False, "INVALID_RELEVANCE_SCORE", "relevance_score must be integer 0..3"

    for bool_field in ("abstain", "contraindication_detected", "safety_flag"):
        if type(rec.get(bool_field)) is not bool:
            return False, "INVALID_BOOLEAN_FIELD", f"{bool_field} must be boolean"

    rationale = str(rec.get("rationale", "")).strip()
    if len(rationale) < 10:
        return False, "INVALID_RATIONALE", "rationale must contain at least 10 characters"

    evidence_ids = rec.get("evidence_ids")
    if not isinstance(evidence_ids, list) or any(
        not _is_nonempty_string(item) for item in evidence_ids
    ):
        return False, "INVALID_EVIDENCE_IDS", "evidence_ids must be a list of non-empty strings"
    if len(evidence_ids) != len(set(evidence_ids)):
        return False, "DUPLICATE_EVIDENCE_ID", "evidence_ids contains duplicates"
    if not rec["abstain"] and not evidence_ids:
        return False, "MISSING_EVIDENCE", "non-abstained review must cite evidence"

    if case_allowed_evidence_ids is not None and case_id in case_allowed_evidence_ids:
        invalid_ids = sorted(set(evidence_ids) - case_allowed_evidence_ids[case_id])
        if invalid_ids:
            return False, "UNKNOWN_EVIDENCE_ID", f"Unknown evidence_ids: {invalid_ids}"

    if locked_prompt_version is not None and rec["prompt_version"] != locked_prompt_version:
        return False, "PROMPT_VERSION_MISMATCH", "prompt_version does not match locked version"

    if not is_sha256(rec["prompt_sha256"]):
        return False, "INVALID_PROMPT_SHA256", "prompt_sha256 must be lowercase SHA-256"
    if locked_prompt_hash is not None and rec["prompt_sha256"] != locked_prompt_hash:
        return False, "PROMPT_HASH_MISMATCH", "prompt_sha256 does not match locked prompt"

    for field in (
        "request_batch_sha256",
        "raw_request_sha256",
        "raw_response_sha256",
        "response_record_sha256",
    ):
        if not is_sha256(rec[field]):
            return False, f"INVALID_{field.upper()}", f"{field} must be lowercase SHA-256"

    record_index = rec.get("response_record_index")
    if type(record_index) is not int or record_index < 0:
        return False, "INVALID_RESPONSE_RECORD_INDEX", "response_record_index must be >= 0"

    if not _valid_timestamp(rec["created_at"]):
        return False, "INVALID_CREATED_AT", "created_at must be timezone-aware ISO-8601"

    if envelope_registry is not None:
        key = (provider, rec["batch_id"])
        if key not in envelope_registry:
            return False, "ENVELOPE_NOT_FOUND", f"No envelope registry entry for {key}"
        env_data = envelope_registry[key]
        req_env = env_data.get("request_envelope", {})
        resp_env = env_data.get("response_envelope", {})
        if req_env.get("provider") != provider or resp_env.get("provider") != provider:
            return False, "ENVELOPE_PROVIDER_MISMATCH", "Envelope provider mismatch"
        if req_env.get("request_id") != rec["request_id"]:
            return False, "REQUEST_ID_MISMATCH", "Request-envelope request_id mismatch"
        if resp_env.get("request_id") != rec["request_id"]:
            return False, "REQUEST_ID_MISMATCH", "Response-envelope request_id mismatch"
        records = resp_env.get("records", [])
        if records and not any(
            item.get("sha256") == rec["response_record_sha256"]
            for item in records
            if isinstance(item, dict)
        ):
            return False, "RECORD_HASH_MISMATCH", "response_record_sha256 absent from envelope"
        return True, "OK", "Record is valid"

    is_env_ok, env_code, env_msg = verify_provider_envelope(
        envelope_root=envelope_root,
        provider_name=provider,
        batch_id=rec["batch_id"],
        rec=rec,
        locked_prompt_hash=locked_prompt_hash,
    )
    if not is_env_ok:
        return False, env_code, env_msg

    return True, "OK", "Record is valid"


def _empty_output() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _write_outputs(
    output_file: Path,
    df_out: pd.DataFrame,
    rejected_records: list[dict[str, Any]],
    provider_status: str,
    locked_prompt_hash: str | None,
    blocked_reason: str | None = None,
) -> dict[str, Any]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_file, index=False)

    REJECTED_RECORDS_PATH.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in rejected_records),
        encoding="utf-8",
    )

    indep_summary = compute_source_independence_audit(df_out)
    verified_count = len(df_out)

    manifest = {
        "schema_version": "recommend_hybrid_v2_import_v3",
        "import_status": "PASS" if verified_count > 0 else "BLOCKED_NO_VERIFIED_RAW_RESPONSES",
        "external_provider_status": provider_status,
        "locked_prompt_version": LOCKED_PROMPT_VERSION,
        "locked_prompt_sha256": locked_prompt_hash,
        "real_external_llm_review_count": verified_count,
        "verified_independent_source_count": indep_summary["verified_independent_source_count"],
        "unique_case_count": int(df_out["case_id"].nunique()) if not df_out.empty else 0,
        "unique_reviewer_count": int(df_out["reviewer_id"].nunique()) if not df_out.empty else 0,
        "panel_a_count": int((df_out["panel_id"] == "PANEL_A").sum()) if not df_out.empty else 0,
        "panel_b_count": int((df_out["panel_id"] == "PANEL_B").sum()) if not df_out.empty else 0,
        "invalid_count": len(rejected_records),
        "rejected_count": len(rejected_records),
        "abstention_count": int(df_out["abstain"].sum()) if not df_out.empty else 0,
        "independence_audit": indep_summary,
        "fail_closed_triggered": verified_count == 0,
        "reason": blocked_reason,
    }

    (IMPORTS_DIR / "import_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    quality_report = {
        "status": "PASS" if verified_count > 0 else "BLOCKED",
        "fail_closed": verified_count == 0,
        "total_imported_records": verified_count,
        "rejected_records_count": len(rejected_records),
        "validation_summary": manifest,
    }
    (IMPORTS_DIR / "import_quality_report.json").write_text(
        json.dumps(quality_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def import_annotations(
    raw_dir: Path = RAW_DIR,
    output_file: Path = ACCEPTED_RECORDS_PATH,
) -> dict[str, Any]:
    IMPORTS_DIR.mkdir(parents=True, exist_ok=True)

    provider_status, available_providers, audit_error = _load_capability_audit()

    if not LOCKED_PROMPT_PATH.exists():
        return _write_outputs(
            output_file,
            _empty_output(),
            [],
            provider_status,
            None,
            "Locked external-review prompt file missing",
        )
    locked_prompt_hash = canonical_text_sha256(LOCKED_PROMPT_PATH)

    try:
        (
            known_cases,
            case_panels,
            case_candidate_actions,
            case_allowed_evidence_ids,
        ) = _load_case_registry()
    except RuntimeError as exc:
        return _write_outputs(
            output_file,
            _empty_output(),
            [],
            provider_status,
            locked_prompt_hash,
            f"Blinded case registry invalid: {exc}",
        )

    raw_files = sorted(raw_dir.glob("*.jsonl")) if raw_dir.exists() else []
    if provider_status != "AVAILABLE" or not raw_files:
        reason = audit_error or "No verified external provider available or raw response files missing"
        return _write_outputs(
            output_file,
            _empty_output(),
            [],
            provider_status,
            locked_prompt_hash,
            reason,
        )

    verified_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for raw_file in raw_files:
        for line_number, line in enumerate(raw_file.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue

            source_record_sha = hashlib.sha256(line.encode("utf-8")).hexdigest()
            now_iso = datetime.now(timezone.utc).isoformat()

            try:
                rec = json.loads(line)
            except Exception as exc:
                rejected_records.append(
                    {
                        "source_file": raw_file.name,
                        "line_number": line_number,
                        "record_sha256": source_record_sha,
                        "case_id": None,
                        "rejection_code": "MALFORMED_JSON",
                        "rejection_message": str(exc),
                        "created_at": now_iso,
                    }
                )
                continue

            case_id = rec.get("case_id", "") if isinstance(rec, dict) else ""
            action_id = rec.get("action_id", "") if isinstance(rec, dict) else ""
            reviewer_id = rec.get("reviewer_id", "") if isinstance(rec, dict) else ""

            is_valid, code, msg = validate_record(
                rec,
                known_cases=known_cases,
                case_panels=case_panels,
                case_candidate_actions=case_candidate_actions,
                case_allowed_evidence_ids=case_allowed_evidence_ids,
                approved_providers=available_providers,
                locked_prompt_hash=locked_prompt_hash,
                locked_prompt_version=LOCKED_PROMPT_VERSION,
                known_actions=KNOWN_ACTIONS,
                envelope_root=ENVELOPE_ROOT,
            )

            if not is_valid:
                rejected_records.append(
                    {
                        "source_file": raw_file.name,
                        "line_number": line_number,
                        "record_sha256": source_record_sha,
                        "case_id": case_id or None,
                        "rejection_code": code,
                        "rejection_message": msg,
                        "created_at": now_iso,
                    }
                )
                continue

            duplicate_key = (case_id, action_id, reviewer_id)
            if duplicate_key in seen:
                rejected_records.append(
                    {
                        "source_file": raw_file.name,
                        "line_number": line_number,
                        "record_sha256": source_record_sha,
                        "case_id": case_id,
                        "rejection_code": "DUPLICATE_REVIEW",
                        "rejection_message": f"Duplicate reviewer-case-action key {duplicate_key}",
                        "created_at": now_iso,
                    }
                )
                continue
            seen.add(duplicate_key)

            verified_records.append(
                {
                    "case_id": case_id,
                    "panel_id": rec["panel_id"],
                    "action_id": action_id,
                    "relevance_score": rec["relevance_score"],
                    "abstain": rec["abstain"],
                    "evidence_ids": json.dumps(rec["evidence_ids"], ensure_ascii=False),
                    "rationale": rec["rationale"],
                    "contraindication_detected": rec["contraindication_detected"],
                    "safety_flag": rec["safety_flag"],
                    "reviewer_id": reviewer_id,
                    "reviewer_configuration_id": rec["reviewer_configuration_id"],
                    "reviewer_type": rec["reviewer_type"],
                    "provider": rec["provider"],
                    "model_name": rec["model_name"],
                    "model_version": rec.get("model_version"),
                    "request_id": rec["request_id"],
                    "response_id": rec["response_id"],
                    "batch_id": rec["batch_id"],
                    "prompt_version": rec["prompt_version"],
                    "prompt_sha256": rec["prompt_sha256"],
                    "request_batch_sha256": rec["request_batch_sha256"],
                    "raw_request_sha256": rec["raw_request_sha256"],
                    "raw_response_sha256": rec["raw_response_sha256"],
                    "response_record_index": rec["response_record_index"],
                    "response_record_sha256": rec["response_record_sha256"],
                    "created_at": rec["created_at"],
                    "source_record_sha256": source_record_sha,
                    "eligible_for_final_snorkel": True,
                    "classification": "VERIFIED_EXTERNAL_LLM_REVIEW",
                }
            )

    df_out = pd.DataFrame(verified_records, columns=OUTPUT_COLUMNS)
    return _write_outputs(
        output_file,
        df_out,
        rejected_records,
        provider_status,
        locked_prompt_hash,
        None if verified_records else "All external-review records were rejected",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = import_annotations()
    print(f"IMPORT_STATUS={result['import_status']}")
    print(f"REAL_EXTERNAL_LLM_REVIEW_COUNT={result['real_external_llm_review_count']}")
