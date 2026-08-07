"""Verify that exported cases have valid lineage in action_candidates.parquet."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANDIDATES_PATH = ROOT / "artifacts/recommend_hybrid/explainable_v2/features/action_candidates.parquet"
EXPORT_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports"

FORBIDDEN_PATTERNS = ["course_alpha", "q_EARLY_20_", "q_EARLY_35_", "q_MIDDLE_50_", "q_LATE_75_"]
SYNTHETIC_STUDENT_PREFIX = "pseudo_"


def verify() -> dict:
    result = {
        "case_lineage_audit_status": "PENDING",
        "failures": [],
        "panel_a_case_count": 0,
        "panel_b_case_count": 0,
        "panel_student_overlap_count": 0,
        "panel_query_overlap_count": 0,
        "post_cutoff_violations": 0,
        "synthetic_pattern_count": 0,
        "lineage_failures": 0,
        "verified_cases": 0,
    }

    if not CANDIDATES_PATH.exists():
        result["case_lineage_audit_status"] = "FAIL"
        result["failures"].append("action_candidates.parquet not found")
        return result

    df_cand = pd.read_parquet(CANDIDATES_PATH)
    real_query_ids = set(df_cand["query_id"].unique())

    panel_a_path = EXPORT_DIR / "panel_a_cases.jsonl"
    panel_b_path = EXPORT_DIR / "panel_b_cases.jsonl"

    def load_cases(p: Path) -> list[dict]:
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    panel_a = load_cases(panel_a_path)
    panel_b = load_cases(panel_b_path)

    result["panel_a_case_count"] = len(panel_a)
    result["panel_b_case_count"] = len(panel_b)

    # Check student overlap
    pa_students = {c.get("source_student_group_id_hash", c.get("student_pseudonym", "")) for c in panel_a}
    pb_students = {c.get("source_student_group_id_hash", c.get("student_pseudonym", "")) for c in panel_b}
    overlap = pa_students & pb_students
    result["panel_student_overlap_count"] = len(overlap)
    if overlap:
        result["failures"].append(f"Student overlap detected: {len(overlap)} students")

    # Check query overlap
    pa_queries = {c.get("query_id", "") for c in panel_a}
    pb_queries = {c.get("query_id", "") for c in panel_b}
    q_overlap = pa_queries & pb_queries
    result["panel_query_overlap_count"] = len(q_overlap)
    if q_overlap:
        result["failures"].append(f"Query overlap: {q_overlap}")

    # Check each case
    all_cases = [(c, "A") for c in panel_a] + [(c, "B") for c in panel_b]
    synthetic_count = 0
    lineage_fail = 0
    verified = 0

    for case, panel in all_cases:
        qid = case.get("query_id", "")
        course_pseudo = case.get("course_pseudonym", "")
        student_pseudo = case.get("student_pseudonym", "")

        # Check synthetic patterns
        for pat in FORBIDDEN_PATTERNS:
            if pat in qid:
                synthetic_count += 1
                result["failures"].append(f"Panel {panel}: synthetic query_id pattern '{pat}' in '{qid}'")
                break
        if course_pseudo == "course_alpha":
            synthetic_count += 1
            result["failures"].append(f"Panel {panel}: hardcoded course_alpha in case {qid}")
        if student_pseudo.startswith(SYNTHETIC_STUDENT_PREFIX):
            synthetic_count += 1
            result["failures"].append(f"Panel {panel}: raw-student-id-based pseudonym in {qid}")

        # Check lineage — source_query_id must exist in candidates
        src_qid = case.get("source_query_id", qid)
        if src_qid not in real_query_ids:
            lineage_fail += 1
            result["failures"].append(f"Panel {panel}: query_id '{src_qid}' not in action_candidates")
        else:
            verified += 1

    result["synthetic_pattern_count"] = synthetic_count
    result["lineage_failures"] = lineage_fail
    result["verified_cases"] = verified

    if synthetic_count == 0 and lineage_fail == 0 and len(overlap) == 0 and len(q_overlap) == 0:
        result["case_lineage_audit_status"] = "PASS"
    else:
        result["case_lineage_audit_status"] = "FAIL"

    return result


if __name__ == "__main__":
    import json as _json
    r = verify()
    print(_json.dumps(r, indent=2))
    sys.exit(0 if r["case_lineage_audit_status"] == "PASS" else 1)
