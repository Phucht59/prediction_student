"""Verify exported case lineage and strict public privacy blinding."""
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
PRIVATE_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/private"

FORBIDDEN_PUBLIC_KEYS = {"source_query_id", "source_student_group_id_hash", "student_pseudonym", "course_pseudonym", "outer_fold"}
FORBIDDEN_PUBLIC_PATTERNS = ["course_alpha", "q_EARLY_20_", "q_EARLY_35_", "q_MIDDLE_50_", "q_LATE_75_", "pseudo_"]


def verify() -> dict:
    result = {
        "case_lineage_audit_status": "PENDING",
        "public_privacy_verified": False,
        "failures": [],
        "panel_a_case_count": 0,
        "panel_b_case_count": 0,
        "panel_student_overlap_count": 0,
        "panel_query_overlap_count": 0,
        "privacy_leak_count": 0,
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
    private_map_path = PRIVATE_DIR / "private_case_mapping.json"

    if not private_map_path.exists():
        result["case_lineage_audit_status"] = "FAIL"
        result["failures"].append("private_case_mapping.json missing")
        return result

    p_map = json.loads(private_map_path.read_text(encoding="utf-8"))

    def load_cases(p: Path) -> list[dict]:
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    panel_a = load_cases(panel_a_path)
    panel_b = load_cases(panel_b_path)

    result["panel_a_case_count"] = len(panel_a)
    result["panel_b_case_count"] = len(panel_b)

    all_public_cases = [(c, "A") for c in panel_a] + [(c, "B") for c in panel_b]

    # ── 1. Check Public Privacy Blinding ──────────────────────────────────────
    privacy_leaks = 0
    for case, panel in all_public_cases:
        leaked = FORBIDDEN_PUBLIC_KEYS & set(case.keys())
        if leaked:
            privacy_leaks += 1
            result["failures"].append(f"Panel {panel} case {case.get('case_id')} leaks private fields: {leaked}")
        
        cid = case.get("case_id", "")
        for pat in FORBIDDEN_PUBLIC_PATTERNS:
            if pat in cid:
                privacy_leaks += 1
                result["failures"].append(f"Panel {panel} case_id contains forbidden pattern '{pat}'")

    result["privacy_leak_count"] = privacy_leaks
    result["public_privacy_verified"] = (privacy_leaks == 0)

    # ── 2. Check Lineage via Private Mapping ──────────────────────────────────
    lineage_fail = 0
    verified = 0

    pa_cids = {c["case_id"] for c in panel_a}
    pb_cids = {c["case_id"] for c in panel_b}

    pa_sids = set()
    pb_sids = set()
    pa_qids = set()
    pb_qids = set()

    for cid in pa_cids | pb_cids:
        if cid not in p_map:
            lineage_fail += 1
            result["failures"].append(f"case_id {cid} missing in private_case_mapping.json")
            continue
        
        m_entry = p_map[cid]
        src_qid = m_entry.get("source_query_id", "")
        src_sid = m_entry.get("source_student_group_id", "")

        if src_qid not in real_query_ids:
            lineage_fail += 1
            result["failures"].append(f"source_query_id '{src_qid}' not found in action_candidates.parquet")
        else:
            verified += 1

        if cid in pa_cids:
            pa_sids.add(src_sid)
            pa_qids.add(src_qid)
        else:
            pb_sids.add(src_sid)
            pb_qids.add(src_qid)

    # ── 3. Check Overlaps ─────────────────────────────────────────────────────
    s_overlap = pa_sids & pb_sids
    q_overlap = pa_qids & pb_qids

    result["panel_student_overlap_count"] = len(s_overlap)
    result["panel_query_overlap_count"] = len(q_overlap)
    result["lineage_failures"] = lineage_fail
    result["verified_cases"] = verified

    if s_overlap:
        result["failures"].append(f"Student overlap detected between panels: {len(s_overlap)}")
    if q_overlap:
        result["failures"].append(f"Query overlap detected between panels: {len(q_overlap)}")

    if (privacy_leaks == 0 and lineage_fail == 0 and len(s_overlap) == 0
            and len(q_overlap) == 0 and len(panel_a) >= 240 and len(panel_b) >= 120):
        result["case_lineage_audit_status"] = "PASS"
    else:
        result["case_lineage_audit_status"] = "FAIL"

    return result


if __name__ == "__main__":
    r = verify()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["case_lineage_audit_status"] == "PASS" else 1)
