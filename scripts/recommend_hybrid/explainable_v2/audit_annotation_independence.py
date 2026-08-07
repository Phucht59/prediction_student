"""Audit annotation independence — detect pseudo reviews masquerading as real LLM."""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/imports/raw"
PSEUDO_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/pseudo_agent_experiments"
REQUIRED_PROVENANCE = {"provider", "request_id", "response_id"}
FAKE_MODEL_NAMES = {"Antigravity-LLM-v2-ReviewerA", "Antigravity-LLM-v2-ReviewerB",
                     "Antigravity-LLM-v2-ReviewerC", "ANTIGRAVITY_INTERNAL_RULE_AGENT"}


def audit() -> dict:
    result = {
        "verified_independent_source_count": 0,
        "same_base_model_reviewer_count": 0,
        "exact_duplicate_rate": 0.0,
        "near_duplicate_rationale_rate": 0.0,
        "action_only_predictability": "UNTESTED",
        "stage_only_predictability": "UNTESTED",
        "cyclic_index_predictability": "UNTESTED",
        "missing_provider_count": 0,
        "missing_request_id_count": 0,
        "missing_response_id_count": 0,
        "fake_model_name_count": 0,
        "mislabeled_as_real_llm_count": 0,
        "raw_response_files_count": 0,
        "agent_pseudo_annotation_count": 0,
        "real_external_annotation_count": 0,
        "annotation_provenance_gate": "FAIL",
        "failures": [],
    }

    # Check raw dir for annotations claimed as real
    records = []
    if RAW_DIR.exists():
        for f in RAW_DIR.glob("*.jsonl"):
            result["raw_response_files_count"] += 1
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                except json.JSONDecodeError:
                    result["failures"].append(f"Invalid JSON in {f.name}")

    for rec in records:
        model_name = rec.get("model_name", "")
        reviewer_type = rec.get("reviewer_type", "")
        provider = rec.get("provider", None)
        request_id = rec.get("request_id", None)
        response_id = rec.get("response_id", None)

        if model_name in FAKE_MODEL_NAMES:
            result["fake_model_name_count"] += 1
            result["failures"].append(f"Fake model name: {model_name}")

        if reviewer_type == "REAL_LLM_GENERATED_REVIEW":
            result["mislabeled_as_real_llm_count"] += 1
            result["failures"].append(f"Mislabeled record: reviewer_type=REAL_LLM_GENERATED_REVIEW is forbidden")

        if reviewer_type == "AGENT_GENERATED_PSEUDO_REVIEW":
            result["agent_pseudo_annotation_count"] += 1

        if not provider:
            result["missing_provider_count"] += 1
        if not request_id:
            result["missing_request_id_count"] += 1
        if not response_id:
            result["missing_response_id_count"] += 1

        if (provider and provider not in ("NONE", "NONE_INTERNAL", None)
                and request_id and response_id
                and reviewer_type == "REAL_EXTERNAL_LLM_REVIEW"):
            result["real_external_annotation_count"] += 1

    # Check duplicate rates
    rationales = [r.get("rationale", "") for r in records]
    if rationales:
        total = len(rationales)
        exact_dups = total - len(set(rationales))
        result["exact_duplicate_rate"] = round(exact_dups / total, 4) if total else 0.0

    result["verified_independent_source_count"] = result["real_external_annotation_count"]

    # Gate
    if (result["mislabeled_as_real_llm_count"] == 0
            and result["fake_model_name_count"] == 0
            and result["real_external_annotation_count"] > 0):
        result["annotation_provenance_gate"] = "PASS"
    elif result["real_external_annotation_count"] == 0:
        result["annotation_provenance_gate"] = "BLOCKED"
        result["failures"].append("VERIFIED_EXTERNAL_LLM_REVIEW_COUNT=0 — annotation gate BLOCKED")
    else:
        result["annotation_provenance_gate"] = "FAIL"

    return result


if __name__ == "__main__":
    r = audit()
    print(json.dumps(r, indent=2))
    if r["annotation_provenance_gate"] == "PASS":
        sys.exit(0)
    elif r["annotation_provenance_gate"] == "BLOCKED":
        sys.exit(2)
    else:
        sys.exit(1)
