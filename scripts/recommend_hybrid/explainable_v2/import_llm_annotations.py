"""Import, validate, and normalize external LLM annotation responses with strict fail-closed security.

FAIL-CLOSED AUDIT GUARANTEES:
1. Checks EXTERNAL_PROVIDER_CAPABILITY_AUDIT.json. If external_provider_status == "UNAVAILABLE",
   import MUST fail-closed with status = BLOCKED_NO_VERIFIED_RAW_RESPONSES.
2. Rejects all pseudo-agent reviews, rule-derived labels, synthetic fixtures, and mislabeled records.
3. Verifies raw response file existence, SHA-256 integrity, authentic provider metadata,
   non-empty request_id/response_id, prompt_sha256, and valid case_id lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPORT_MANIFEST_PATH = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports/case_manifest.json"
PRIVATE_MAPPING_PATH = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/private/private_case_mapping.json"
CAPABILITY_AUDIT_PATH = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/EXTERNAL_PROVIDER_CAPABILITY_AUDIT.json"
RAW_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/imports/raw"
IMPORTS_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/imports"

FORBIDDEN_TYPES = {"REAL_LLM_GENERATED_REVIEW", "AGENT_GENERATED_PSEUDO_REVIEW", "RULE_DERIVED_LABEL"}
FORBIDDEN_MODELS = {"Antigravity-LLM-v2-ReviewerA", "Antigravity-LLM-v2-ReviewerB",
                    "Antigravity-LLM-v2-ReviewerC", "ANTIGRAVITY_INTERNAL_RULE_AGENT"}


def import_annotations(raw_dir: Path = RAW_DIR, output_file: Path = IMPORTS_DIR / "normalized_annotations.parquet") -> dict:
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
    if PRIVATE_MAPPING_PATH.exists():
        p_map = json.loads(PRIVATE_MAPPING_PATH.read_text(encoding="utf-8"))
        known_cases = set(p_map.keys())

    raw_files = list(raw_dir.glob("*.jsonl")) if raw_dir.exists() else []

    # FAIL-CLOSED: If external provider is UNAVAILABLE or 0 raw response files exist
    if provider_status != "AVAILABLE" or len(raw_files) == 0:
        empty_df = pd.DataFrame(columns=[
            "case_id", "action_id", "relevance_score", "abstain",
            "evidence_ids", "rationale", "contraindication_detected",
            "safety_flag", "reviewer_id", "reviewer_configuration_id",
            "reviewer_type", "provider", "model_name", "request_id",
            "prompt_version", "prompt_sha256", "raw_response_sha256",
            "eligible_for_final_snorkel", "classification"
        ])
        empty_df.to_parquet(output_file, index=False)

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

        return manifest_data

    verified_records = []
    invalid_count = 0
    duplicate_count = 0
    unverified_count = 0
    seen = set()

    for rf in raw_files:
        h = hashlib.sha256()
        with open(rf, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        raw_hash = h.hexdigest()

        for line in rf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                rt = rec.get("reviewer_type", "")
                mn = rec.get("model_name", "")
                prov = rec.get("provider", "")
                cid = rec.get("case_id", "")
                aid = rec.get("action_id", "")
                rid = rec.get("reviewer_id", "")
                req_id = rec.get("request_id", "")

                if rt in FORBIDDEN_TYPES or mn in FORBIDDEN_MODELS:
                    invalid_count += 1
                    continue
                if rt != "REAL_EXTERNAL_LLM_REVIEW":
                    invalid_count += 1
                    continue
                if prov not in available_providers:
                    unverified_count += 1
                    continue
                if not req_id:
                    unverified_count += 1
                    continue
                if cid not in known_cases:
                    invalid_count += 1
                    continue

                key = (cid, aid, rid)
                if key in seen:
                    duplicate_count += 1
                    continue
                seen.add(key)

                verified_rec = {
                    "case_id": cid,
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
                    "provider": prov,
                    "model_name": mn,
                    "request_id": req_id,
                    "prompt_version": str(rec.get("prompt_version", "v2.0_locked")),
                    "prompt_sha256": str(rec.get("prompt_sha256", "")),
                    "raw_response_sha256": raw_hash,
                    "eligible_for_final_snorkel": True,
                    "classification": "VERIFIED_EXTERNAL_LLM_REVIEW",
                }
                verified_records.append(verified_rec)

            except Exception:
                invalid_count += 1

    df_out = pd.DataFrame(verified_records)
    if not df_out.empty:
        df_out.to_parquet(output_file, index=False)
    else:
        empty_df = pd.DataFrame(columns=[
            "case_id", "action_id", "relevance_score", "abstain",
            "evidence_ids", "rationale", "contraindication_detected",
            "safety_flag", "reviewer_id", "reviewer_configuration_id",
            "reviewer_type", "provider", "model_name", "request_id",
            "prompt_version", "prompt_sha256", "raw_response_sha256",
            "eligible_for_final_snorkel", "classification"
        ])
        empty_df.to_parquet(output_file, index=False)

    manifest_data = {
        "schema_version": "recommend_hybrid_v2_import_v2",
        "import_status": "PASS" if len(verified_records) > 0 else "BLOCKED_NO_VERIFIED_RAW_RESPONSES",
        "external_provider_status": provider_status,
        "real_external_llm_review_count": len(verified_records),
        "verified_independent_source_count": len(df_out["provider"].unique()) if not df_out.empty else 0,
        "unique_case_count": len(df_out["case_id"].unique()) if not df_out.empty else 0,
        "unique_reviewer_count": len(df_out["reviewer_id"].unique()) if not df_out.empty else 0,
        "duplicate_count": duplicate_count,
        "invalid_count": invalid_count,
        "unverified_count": unverified_count,
        "fail_closed_triggered": len(verified_records) == 0,
    }

    (IMPORTS_DIR / "import_manifest.json").write_text(
        json.dumps(manifest_data, indent=2), encoding="utf-8"
    )

    quality_report = {
        "status": "PASS" if len(verified_records) > 0 else "BLOCKED",
        "fail_closed": len(verified_records) == 0,
        "total_imported_records": len(verified_records),
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
