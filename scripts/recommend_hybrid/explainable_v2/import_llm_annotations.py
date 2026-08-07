"""Import, validate, and normalize external LLM annotation responses with strict fail-closed security.

FAIL-CLOSED AUDIT GUARANTEES:
1. Checks EXTERNAL_PROVIDER_CAPABILITY_AUDIT.json. If external_provider_status == "UNAVAILABLE",
   import MUST fail-closed with status = BLOCKED_NO_VERIFIED_RAW_RESPONSES.
2. Production loop calls validate_record(...) as the ONLY path for record acceptance.
3. Every rejected record (including malformed JSON) is appended to rejected_records.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.explainable_v2.independence_audit import compute_source_independence_audit
from src.recommend_hybrid.explainable_v2.provider_envelope import verify_provider_envelope

EXPORT_MANIFEST_PATH = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports/case_manifest.json"
PRIVATE_MAPPING_PATH = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/private/private_case_mapping.json"
CAPABILITY_AUDIT_PATH = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/EXTERNAL_PROVIDER_CAPABILITY_AUDIT.json"
RAW_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/imports/raw"
IMPORTS_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/imports"
ENVELOPE_ROOT = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/external_reviews"

ACCEPTED_RECORDS_PATH = IMPORTS_DIR / "accepted_records.parquet"
REJECTED_RECORDS_PATH = IMPORTS_DIR / "rejected_records.jsonl"

FORBIDDEN_TYPES = {"REAL_LLM_GENERATED_REVIEW", "AGENT_GENERATED_PSEUDO_REVIEW", "RULE_DERIVED_LABEL"}
FORBIDDEN_MODELS = {"Antigravity-LLM-v2-ReviewerA", "Antigravity-LLM-v2-ReviewerB",
                    "Antigravity-LLM-v2-ReviewerC", "ANTIGRAVITY_INTERNAL_RULE_AGENT"}

REQUIRED_SCHEMA_FIELDS = [
    "case_id", "panel_id", "action_id", "relevance_score", "abstain",
    "evidence_ids", "rationale", "contraindication_detected", "safety_flag",
    "reviewer_id", "reviewer_configuration_id", "reviewer_type", "provider",
    "model_name", "request_id", "response_id", "batch_id", "prompt_version", "prompt_sha256",
    "raw_response_sha256", "created_at"
]


def validate_record(
    rec: dict,
    known_cases: set | None = None,
    case_panels: dict | None = None,
    case_candidate_actions: dict | None = None,
    approved_providers: set | None = None,
    locked_prompt_hash: str | None = None,
    known_actions: set | None = None,
    envelope_registry: dict | None = None,
    envelope_root: Path = ENVELOPE_ROOT,
) -> tuple[bool, str, str]:
    """Validate a single review record. Returns (is_valid, rejection_code, rejection_message)."""
    if not isinstance(rec, dict):
        return False, "MALFORMED_JSON", "Record is not a valid JSON dictionary"

    rt = rec.get("reviewer_type", "")
    mn = rec.get("model_name", "")
    prov = rec.get("provider", "")
    cid = rec.get("case_id", "")
    pid = rec.get("panel_id", "")
    aid = rec.get("action_id", "")
    req_id = rec.get("request_id", "")
    batch_id = rec.get("batch_id", "")

    if rt in FORBIDDEN_TYPES or mn in FORBIDDEN_MODELS:
        return False, "FORBIDDEN_TYPE_OR_MODEL", f"Forbidden reviewer_type '{rt}' or model '{mn}'"

    if rt != "REAL_EXTERNAL_LLM_REVIEW":
        return False, "INVALID_REVIEWER_TYPE", f"reviewer_type must be REAL_EXTERNAL_LLM_REVIEW, got '{rt}'"

    if approved_providers is not None and prov not in approved_providers:
        return False, "UNAPPROVED_PROVIDER", f"Provider '{prov}' not in approved runtime provider registry"

    if not req_id:
        return False, "MISSING_REQUEST_ID", "request_id is missing or empty"

    if known_cases is not None and cid not in known_cases:
        return False, "UNKNOWN_CASE_ID", f"case_id '{cid}' not found in exported case manifest"

    if case_panels is not None and cid in case_panels and pid and case_panels[cid] != pid:
        return False, "PANEL_MISMATCH", f"panel_id '{pid}' does not match case panel '{case_panels[cid]}'"

    if case_candidate_actions is not None and cid in case_candidate_actions:
        cands = case_candidate_actions[cid]
        if aid not in cands:
            return False, "INELIGIBLE_ACTION", f"action_id '{aid}' is not in candidate_actions {cands} for case '{cid}'"

    rel = rec.get("relevance_score")
    abstain = rec.get("abstain", False)
    if not abstain and (not isinstance(rel, int) or rel not in (0, 1, 2, 3)):
        return False, "INVALID_RELEVANCE_SCORE", f"relevance_score '{rel}' is invalid (must be integer 0..3 or abstain=True)"

    rat = str(rec.get("rationale", "")).strip()
    if not rat and not abstain:
        return False, "EMPTY_RATIONALE", "rationale is empty"

    if locked_prompt_hash is not None and rec.get("prompt_sha256"):
        if rec.get("prompt_sha256") != locked_prompt_hash:
            return False, "PROMPT_HASH_MISMATCH", f"prompt_sha256 '{rec.get('prompt_sha256')}' does not match locked prompt hash '{locked_prompt_hash}'"

    # Provider envelope validation
    if envelope_registry is not None or envelope_root.exists():
        key = (prov, batch_id)
        if envelope_registry is not None and key in envelope_registry:
            env_data = envelope_registry[key]
            req_e = env_data.get("request_envelope", {})
            resp_e = env_data.get("response_envelope", {})
            if req_e.get("provider") != prov or resp_e.get("provider") != prov:
                return False, "ENVELOPE_PROVIDER_MISMATCH", f"Envelope provider mismatch for key {key}"
            if req_e.get("request_id") != req_id or resp_e.get("request_id") != req_id:
                return False, "REQUEST_ID_MISMATCH", f"Envelope request_id mismatch for request_id {req_id}"

            rec_sha = rec.get("response_record_sha256", "")
            env_recs = resp_e.get("records", [])
            if env_recs:
                matched = any(r.get("sha256") == rec_sha for r in env_recs)
                if not matched:
                    return False, "RECORD_HASH_MISMATCH", f"Per-record response hash '{rec_sha}' not found in envelope"
        else:
            is_env_ok, env_code, env_msg = verify_provider_envelope(
                envelope_root=envelope_root,
                provider_name=prov,
                batch_id=batch_id,
                rec=rec,
                locked_prompt_hash=locked_prompt_hash,
            )
            if not is_env_ok:
                return False, env_code, env_msg

    return True, "OK", "Record is valid"


def import_annotations(raw_dir: Path = RAW_DIR, output_file: Path = ACCEPTED_RECORDS_PATH) -> dict:
    IMPORTS_DIR.mkdir(parents=True, exist_ok=True)

    provider_status = "UNAVAILABLE"
    available_providers = set()
    if CAPABILITY_AUDIT_PATH.exists():
        try:
            audit_data = json.loads(CAPABILITY_AUDIT_PATH.read_text(encoding="utf-8"))
            provider_status = audit_data.get("external_provider_status", "UNAVAILABLE")
            for prov in audit_data.get("evaluated_providers", []):
                if prov.get("status") == "AVAILABLE":
                    available_providers.add(prov.get("provider_name"))
        except Exception:
            pass

    known_cases = set()
    case_panels = {}
    case_candidate_actions = {}
    if EXPORT_MANIFEST_PATH.parent.exists():
        for pf in (EXPORT_MANIFEST_PATH.parent).glob("*.jsonl"):
            for line in pf.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        cdata = json.loads(line)
                        cid = cdata["case_id"]
                        known_cases.add(cid)
                        case_panels[cid] = cdata.get("panel_id", "")
                        case_candidate_actions[cid] = cdata.get("candidate_actions", [])
                    except Exception:
                        pass

    raw_files = list(raw_dir.glob("*.jsonl")) if raw_dir.exists() else []

    # FAIL-CLOSED: If external provider is UNAVAILABLE or 0 raw response files exist
    if provider_status != "AVAILABLE" or len(raw_files) == 0:
        empty_df = pd.DataFrame(columns=[
            "case_id", "panel_id", "action_id", "relevance_score", "abstain",
            "evidence_ids", "rationale", "contraindication_detected",
            "safety_flag", "reviewer_id", "reviewer_configuration_id",
            "reviewer_type", "provider", "model_name", "request_id",
            "response_id", "batch_id", "prompt_version", "prompt_sha256",
            "raw_response_sha256", "response_record_index", "response_record_sha256",
            "eligible_for_final_snorkel", "classification"
        ])
        empty_df.to_parquet(output_file, index=False)
        (IMPORTS_DIR / "normalized_annotations.parquet").unlink(missing_ok=True)

        indep_summary = compute_source_independence_audit(empty_df)

        manifest_data = {
            "schema_version": "recommend_hybrid_v2_import_v2",
            "import_status": "BLOCKED_NO_VERIFIED_RAW_RESPONSES",
            "external_provider_status": provider_status,
            "real_external_llm_review_count": 0,
            "verified_independent_source_count": 0,
            "unique_case_count": 0,
            "unique_reviewer_count": 0,
            "panel_a_count": 0,
            "panel_b_count": 0,
            "duplicate_count": 0,
            "invalid_count": 0,
            "unverified_count": 0,
            "abstention_count": 0,
            "independence_audit": indep_summary,
            "fail_closed_triggered": True,
            "reason": "No verified external LLM provider available or raw response files missing"
        }

        (IMPORTS_DIR / "import_manifest.json").write_text(
            json.dumps(manifest_data, indent=2), encoding="utf-8"
        )

        quality_report = {
            "status": "BLOCKED",
            "fail_closed": True,
            "total_imported_records": 0,
            "validation_summary": manifest_data,
        }
        (IMPORTS_DIR / "import_quality_report.json").write_text(
            json.dumps(quality_report, indent=2), encoding="utf-8"
        )
        REJECTED_RECORDS_PATH.write_text("", encoding="utf-8")

        return manifest_data

    verified_records = []
    rejected_records = []
    seen = set()

    for rf in raw_files:
        for line_no, line in enumerate(rf.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            rec_hash = hashlib.sha256(line.encode()).hexdigest()
            now_iso = datetime.now(timezone.utc).isoformat()
            try:
                rec = json.loads(line)
                cid = rec.get("case_id", "")
                aid = rec.get("action_id", "")
                rid = rec.get("reviewer_id", "")
                key = (cid, aid, rid)

                # Production loop calls validate_record as ONLY path for acceptance
                is_valid, code, msg = validate_record(
                    rec,
                    known_cases=known_cases,
                    case_panels=case_panels,
                    case_candidate_actions=case_candidate_actions,
                    approved_providers=available_providers,
                    envelope_root=ENVELOPE_ROOT,
                )

                if is_valid:
                    if key in seen:
                        rejected_records.append({
                            "source_file": rf.name,
                            "line_number": line_no,
                            "record_sha256": rec_hash,
                            "case_id": cid,
                            "rejection_code": "DUPLICATE_REVIEW",
                            "rejection_message": f"Duplicate reviewer-case-action key {key}",
                            "created_at": now_iso
                        })
                        continue
                    seen.add(key)

                    rec_idx = rec.get("response_record_index", 0)
                    rec_sha = rec.get("response_record_sha256", rec_hash)

                    verified_rec = {
                        "case_id": cid,
                        "panel_id": rec.get("panel_id", ""),
                        "action_id": aid,
                        "relevance_score": int(rec.get("relevance_score", -1)),
                        "abstain": bool(rec.get("abstain", False)),
                        "evidence_ids": json.dumps(rec.get("evidence_ids", [])),
                        "rationale": str(rec.get("rationale", "")),
                        "contraindication_detected": bool(rec.get("contraindication_detected", False)),
                        "safety_flag": bool(rec.get("safety_flag", False)),
                        "reviewer_id": rid,
                        "reviewer_configuration_id": str(rec.get("reviewer_configuration_id", rid)),
                        "reviewer_type": "REAL_EXTERNAL_LLM_REVIEW",
                        "provider": rec.get("provider", ""),
                        "model_name": rec.get("model_name", ""),
                        "request_id": rec.get("request_id", ""),
                        "response_id": rec.get("response_id", ""),
                        "batch_id": rec.get("batch_id", ""),
                        "prompt_version": str(rec.get("prompt_version", "v2.0_locked")),
                        "prompt_sha256": str(rec.get("prompt_sha256", "")),
                        "raw_response_sha256": rec_hash,
                        "response_record_index": rec_idx,
                        "response_record_sha256": rec_sha,
                        "eligible_for_final_snorkel": True,
                        "classification": "VERIFIED_EXTERNAL_LLM_REVIEW",
                    }
                    verified_records.append(verified_rec)
                else:
                    rejected_records.append({
                        "source_file": rf.name,
                        "line_number": line_no,
                        "record_sha256": rec_hash,
                        "case_id": cid or None,
                        "rejection_code": code,
                        "rejection_message": msg,
                        "created_at": now_iso
                    })

            except Exception as exc:
                rejected_records.append({
                    "source_file": rf.name,
                    "line_number": line_no,
                    "record_sha256": rec_hash,
                    "case_id": None,
                    "rejection_code": "MALFORMED_JSON",
                    "rejection_message": str(exc),
                    "created_at": now_iso
                })

    df_out = pd.DataFrame(verified_records)
    if not df_out.empty:
        df_out.to_parquet(output_file, index=False)
    else:
        empty_df = pd.DataFrame(columns=[
            "case_id", "panel_id", "action_id", "relevance_score", "abstain",
            "evidence_ids", "rationale", "contraindication_detected",
            "safety_flag", "reviewer_id", "reviewer_configuration_id",
            "reviewer_type", "provider", "model_name", "request_id",
            "response_id", "batch_id", "prompt_version", "prompt_sha256",
            "raw_response_sha256", "response_record_index", "response_record_sha256",
            "eligible_for_final_snorkel", "classification"
        ])
        empty_df.to_parquet(output_file, index=False)

    REJECTED_RECORDS_PATH.write_text(
        "\n".join(json.dumps(r) for r in rejected_records), encoding="utf-8"
    )

    indep_summary = compute_source_independence_audit(df_out)

    manifest_data = {
        "schema_version": "recommend_hybrid_v2_import_v2",
        "import_status": "PASS" if len(verified_records) > 0 else "BLOCKED_NO_VERIFIED_RAW_RESPONSES",
        "external_provider_status": provider_status,
        "real_external_llm_review_count": len(verified_records),
        "verified_independent_source_count": indep_summary["verified_independent_source_count"],
        "unique_case_count": len(df_out["case_id"].unique()) if not df_out.empty else 0,
        "unique_reviewer_count": len(df_out["reviewer_id"].unique()) if not df_out.empty else 0,
        "invalid_count": len(rejected_records),
        "rejected_count": len(rejected_records),
        "independence_audit": indep_summary,
        "fail_closed_triggered": len(verified_records) == 0,
    }

    (IMPORTS_DIR / "import_manifest.json").write_text(
        json.dumps(manifest_data, indent=2), encoding="utf-8"
    )

    quality_report = {
        "status": "PASS" if len(verified_records) > 0 else "BLOCKED",
        "fail_closed": len(verified_records) == 0,
        "total_imported_records": len(verified_records),
        "rejected_records_count": len(rejected_records),
        "validation_summary": manifest_data,
    }
    (IMPORTS_DIR / "import_quality_report.json").write_text(
        json.dumps(quality_report, indent=2), encoding="utf-8"
    )

    return manifest_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    m = import_annotations()
    print(f"IMPORT_STATUS={m['import_status']}")
    print(f"REAL_EXTERNAL_LLM_REVIEW_COUNT={m['real_external_llm_review_count']}")
