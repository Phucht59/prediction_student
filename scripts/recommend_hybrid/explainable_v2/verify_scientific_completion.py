"""Independent scientific completion verifier.

Exit codes:
  0 = VERIFIED_COMPLETE
  2 = BLOCKED_EXTERNAL_DEPENDENCY (waiting for verified external LLM reviews)
  1 = INVALID_OR_FAILED

This verifier does NOT trust status fields. It checks actual file existence,
schemas, counts, lineage, and provenance directly.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Required runtime_authorized = False (always)
VALIDATE_PATH = ROOT / "artifacts/validate_explainable_v2_state.json"
CANDIDATES_PATH = ROOT / "artifacts/recommend_hybrid/explainable_v2/features/action_candidates.parquet"
MANIFEST_PATH = ROOT / "artifacts/recommend_hybrid/explainable_v2/data/FEATURE_TABLE_MANIFEST.json"
EXPORT_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports"
IMPORT_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/imports"
RAW_DIR = IMPORT_DIR / "raw"
LABELS_PATH = ROOT / "artifacts/recommend_hybrid/explainable_v2/labels/probabilistic_relevance_labels.parquet"
MODEL_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/models/five_ebm"
SELECTION_PATH = ROOT / "artifacts/recommend_hybrid/explainable_v2/model_selection/model_selection_manifest.json"

FORBIDDEN_PATTERNS = ["course_alpha", "q_EARLY_20_", "q_EARLY_35_", "q_MIDDLE_50_", "q_LATE_75_", "pseudo_"]
FAKE_MODEL_NAMES = {"Antigravity-LLM-v2-ReviewerA", "Antigravity-LLM-v2-ReviewerB",
                     "Antigravity-LLM-v2-ReviewerC", "ANTIGRAVITY_INTERNAL_RULE_AGENT"}
FORBIDDEN_ANNOTATION_TYPES = {"REAL_LLM_GENERATED_REVIEW"}  # old mislabeled type is forbidden
CANONICAL_ACTIONS = {
    "ASSESSMENT_COMPLETION", "RECOVER_ENGAGEMENT",
    "STUDY_REGULARITY", "TARGETED_CONTENT_REVIEW", "QUIZ_RETRIEVAL_PRACTICE",
}


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

    # ── Check 1: real OULAD feature table exists ──────────────────────────────
    if CANDIDATES_PATH.exists():
        df_cand = pd.read_parquet(CANDIDATES_PATH)
        if len(df_cand) >= 312000:
            ok("feature_table_exists")
        else:
            fail("feature_table_exists", f"candidates rows {len(df_cand)} < 312000")
    else:
        fail("feature_table_exists", "action_candidates.parquet missing")

    # ── Check 2: case exports have OULAD lineage, no synthetic patterns ───────
    panel_a_path = EXPORT_DIR / "panel_a_cases.jsonl"
    panel_b_path = EXPORT_DIR / "panel_b_cases.jsonl"
    if panel_a_path.exists() and panel_b_path.exists():
        def load(p: Path) -> list[dict]:
            return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        pa = load(panel_a_path)
        pb = load(panel_b_path)
        synthetic_found = False
        for case in pa + pb:
            qid = case.get("query_id", "")
            course = case.get("course_pseudonym", "")
            stu = case.get("student_pseudonym", "")
            for pat in FORBIDDEN_PATTERNS:
                if pat in qid or course == "course_alpha" or stu.startswith("pseudo_"):
                    synthetic_found = True
                    break
        if synthetic_found:
            fail("case_export_lineage", "Synthetic patterns detected in case exports")
        elif len(pa) >= 240 and len(pb) >= 120:
            ok("case_export_lineage")
        else:
            fail("case_export_lineage", f"Insufficient cases: A={len(pa)}, B={len(pb)}")
    else:
        fail("case_export_lineage", "Panel export files missing")

    # ── Check 3: No fake annotations in imports/raw/ ─────────────────────────
    real_ext_count = 0
    mislabeled_count = 0
    if RAW_DIR.exists():
        for f in RAW_DIR.glob("*.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    rt = rec.get("reviewer_type", "")
                    mn = rec.get("model_name", "")
                    if rt in FORBIDDEN_ANNOTATION_TYPES:
                        mislabeled_count += 1
                    if mn in FAKE_MODEL_NAMES:
                        mislabeled_count += 1
                    if (rt == "REAL_EXTERNAL_LLM_REVIEW"
                            and rec.get("provider") not in (None, "", "NONE", "NONE_INTERNAL")
                            and rec.get("request_id") not in (None, "")):
                        real_ext_count += 1
                except json.JSONDecodeError:
                    pass

    if mislabeled_count > 0:
        fail("annotation_provenance", f"{mislabeled_count} mislabeled annotations in imports/raw/")
    elif real_ext_count == 0:
        block("annotation_provenance", f"VERIFIED_EXTERNAL_LLM_REVIEW_COUNT=0 — blocked pending real reviews")
    else:
        ok("annotation_provenance")

    report["verified_external_llm_review_count"] = real_ext_count
    report["mislabeled_annotation_count"] = mislabeled_count

    # ── Check 4: runtime_authorized == False ──────────────────────────────────
    static_path = ROOT / "artifacts/recommend_hybrid/explainable_v2/run_state/supervisor.json"
    if static_path.exists():
        state = json.loads(static_path.read_text(encoding="utf-8"))
        if state.get("runtime_authorized", True) is False:
            ok("runtime_authorized_false")
        else:
            fail("runtime_authorized_false", "runtime_authorized is True — forbidden")

    # ── Check 5: No active synthetic artifacts in production paths ────────────
    quarantine_path = ROOT / "artifacts/recommend_hybrid/explainable_v2/audit/INVALID_ARTIFACT_QUARANTINE_MANIFEST.json"
    if quarantine_path.exists():
        qm = json.loads(quarantine_path.read_text(encoding="utf-8"))
        still_active = [
            a["path"] for a in qm.get("invalidated_artifacts", [])
            if Path(ROOT / a["path"]).exists()
            and a["category"] in ("annotation", "model", "label", "calibration", "model_selection", "simulator")
        ]
        if still_active:
            fail("quarantine_complete", f"{len(still_active)} invalid artifacts still in active paths")
        else:
            ok("quarantine_complete")
    else:
        fail("quarantine_complete", "Quarantine manifest missing")

    # ── Determine overall status ──────────────────────────────────────────────
    statuses = list(report["checks"].values())
    if any(s == "FAIL" for s in statuses):
        report["scientific_status"] = "INVALID_OR_FAILED"
        return 1, report
    elif any(s == "BLOCKED" for s in statuses):
        report["scientific_status"] = "BLOCKED_PENDING_VERIFIED_EXTERNAL_LLM_REVIEWS"
        return 2, report
    else:
        report["scientific_status"] = "VERIFIED_COMPLETE"
        return 0, report


if __name__ == "__main__":
    code, report = verify()
    print(json.dumps(report, indent=2))
    sys.exit(code)
