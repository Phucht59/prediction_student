"""Independent scientific completion verifier.

Exit codes:
  0 = VERIFIED_COMPLETE
  2 = BLOCKED_EXTERNAL_DEPENDENCY (waiting for verified external LLM reviews)
  1 = INVALID_OR_FAILED
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANDIDATES_PATH = ROOT / "artifacts/recommend_hybrid/explainable_v2/features/action_candidates.parquet"
EXPORT_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports"
PRIVATE_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/private"
RAW_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/imports/raw"
ACCEPTED_RECORDS_PATH = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/imports/accepted_records.parquet"
CAPABILITY_AUDIT_PATH = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/EXTERNAL_PROVIDER_CAPABILITY_AUDIT.json"
MODEL_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/models"


def verify() -> tuple[int, dict]:
    report = {
        "runtime_authorized": False,
        "checks": {},
        "failures": [],
        "scientific_status": "PENDING",
    }

    def fail(key: str, msg: str) -> None:
        report["checks"][key] = "FAIL"
        report["failures"].append(msg)

    def block(key: str, msg: str) -> None:
        report["checks"][key] = "BLOCKED"
        report["failures"].append(msg)

    def ok(key: str) -> None:
        report["checks"][key] = "PASS"

    # Check 1: CASE_EXPORT_SALT source security check
    export_script = ROOT / "scripts/recommend_hybrid/explainable_v2/export_llm_cases.py"
    if export_script.exists():
        content = export_script.read_text(encoding="utf-8")
        if "recommend_v2_blinded_privacy_salt_2026" in content:
            fail("salt_security", "Hardcoded salt literal found in export_llm_cases.py")
        else:
            ok("salt_security")

    # Check 2: private_case_mapping.json not git-tracked
    p_map_path = PRIVATE_DIR / "private_case_mapping.json"
    res = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(p_map_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if res.returncode == 0:
        fail("private_mapping_git_tracked", "private_case_mapping.json is tracked by Git!")
    else:
        ok("private_mapping_git_tracked")

    # Check 3: Public export files contain ZERO unblinded identifiers
    panel_a_path = EXPORT_DIR / "panel_a_cases.jsonl"
    panel_b_path = EXPORT_DIR / "panel_b_cases.jsonl"
    if panel_a_path.exists() and panel_b_path.exists():
        pa = [json.loads(l) for l in panel_a_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        pb = [json.loads(l) for l in panel_b_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        forbidden_keys = {"query_id", "source_query_id", "id_student", "student_group_id", "module", "presentation", "outer_fold"}
        privacy_leaks = 0
        for c in pa + pb:
            leaked = forbidden_keys & set(c.keys())
            if leaked:
                privacy_leaks += 1
        if privacy_leaks > 0:
            fail("public_privacy", f"{privacy_leaks} public cases contain forbidden unblinded keys")
        else:
            ok("public_privacy")
    else:
        fail("public_privacy", "Public export files missing")

    # Check 4: SAMPLING_AUDIT PASS
    sampling_audit_path = EXPORT_DIR / "SAMPLING_AUDIT.json"
    if sampling_audit_path.exists():
        sa = json.loads(sampling_audit_path.read_text(encoding="utf-8"))
        if sa.get("sampling_method") == "PROPORTIONAL_STRATIFIED_GROUP_ALLOCATION" and sa.get("final_selected_case_count") == 450:
            ok("sampling_audit")
        else:
            fail("sampling_audit", "SAMPLING_AUDIT invalid or missing final selected cases count")
    else:
        fail("sampling_audit", "SAMPLING_AUDIT.json missing")

    # Check 5: Hardened Annotation Provenance via accepted_records.parquet & envelopes
    real_ext_count = 0
    if ACCEPTED_RECORDS_PATH.exists():
        try:
            df_acc = pd.read_parquet(ACCEPTED_RECORDS_PATH)
            if not df_acc.empty:
                real_ext_count = len(df_acc[df_acc["classification"] == "VERIFIED_EXTERNAL_LLM_REVIEW"])
        except Exception:
            pass

    if real_ext_count == 0:
        block("annotation_provenance", "VERIFIED_EXTERNAL_LLM_REVIEW_COUNT=0 — blocked pending real reviews")
    else:
        ok("annotation_provenance")

    report["verified_external_llm_review_count"] = real_ext_count

    # Check 6: Check Five EBM model artifacts (required for exit 0)
    ebm_models_exist = False
    if MODEL_DIR.exists() and len(list(MODEL_DIR.glob("five_ebm_*.pkl"))) >= 5:
        ebm_models_exist = True

    if not ebm_models_exist:
        block("five_ebm_models", "Five EBM models not trained — blocked pending external LLM reviews")

    # Check 7: Check metric recomputation & model selection artifacts
    metrics_exist = (ROOT / "artifacts/recommend_hybrid/explainable_v2/metrics/MODEL_SELECTION_REPORT.json").exists()
    if not metrics_exist:
        block("metric_recomputation", "Model selection & metric recomputation missing — blocked")

    # Check 8: runtime_authorized == False
    state_path = ROOT / "artifacts/recommend_hybrid/explainable_v2/run_state/supervisor.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("runtime_authorized", True) is False:
            ok("runtime_authorized_false")
        else:
            fail("runtime_authorized_false", "runtime_authorized is True — forbidden")

    # Determine overall exit code and scientific status
    statuses = list(report["checks"].values())
    if any(s == "FAIL" for s in statuses):
        report["scientific_status"] = "INVALID_OR_FAILED"
        return 1, report
    elif any(s == "BLOCKED" for s in statuses):
        report["scientific_status"] = "BLOCKED_PENDING_EXTERNAL_LLM_ACCESS"
        return 2, report
    else:
        report["scientific_status"] = "VERIFIED_COMPLETE"
        return 0, report


if __name__ == "__main__":
    code, report = verify()
    print(json.dumps(report, indent=2))
    sys.exit(code)
