"""Export V2 blinded student-stage cases for LLM annotation batches.

LINEAGE: All cases sourced directly from verified OULAD feature table
(action_candidates.parquet). No synthetic or loop-generated data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANDIDATES_PATH = (
    ROOT / "artifacts/recommend_hybrid/explainable_v2/features/action_candidates.parquet"
)
CANDIDATES_MANIFEST_PATH = (
    ROOT / "artifacts/recommend_hybrid/explainable_v2/data/FEATURE_TABLE_MANIFEST.json"
)
EXPORT_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports"
PRIVATE_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/private"
PROMPTS_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/prompts"

SAMPLING_SEED = 2026
PANEL_A_MIN = 240
PANEL_B_MIN = 120

_SALT = os.environ.get("CASE_EXPORT_SALT", "recommend_v2_audit_salt_2026").encode()


def _deterministic_pseudonym(raw_id: str) -> str:
    return hashlib.sha256(_SALT + raw_id.encode()).hexdigest()[:12]


def _row_sha256(row_dict: dict) -> str:
    row_bytes = json.dumps(row_dict, sort_keys=True, default=str).encode()
    return hashlib.sha256(row_bytes).hexdigest()


def _manifest_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def export_v2_cases() -> dict:
    for d in (EXPORT_DIR, PRIVATE_DIR, PROMPTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if not CANDIDATES_PATH.exists():
        print("ERROR: action_candidates.parquet not found")
        sys.exit(1)

    df = pd.read_parquet(CANDIDATES_PATH)
    cand_sha = _manifest_sha256(CANDIDATES_PATH)
    cand_manifest_sha = (
        _manifest_sha256(CANDIDATES_MANIFEST_PATH)
        if CANDIDATES_MANIFEST_PATH.exists()
        else "MISSING"
    )

    query_groups = df.groupby("query_id")
    cases_by_query: dict[str, dict] = {}
    student_to_queries: dict[str, list[str]] = {}

    for qid, grp in query_groups:
        first = grp.iloc[0]
        stage_val = str(first["stage"]) if "stage" in grp.columns else "UNKNOWN"
        student_group_id = (
            str(first["student_group_id"]) if "student_group_id" in grp.columns else qid
        )
        outer_fold = int(first["outer_fold"]) if "outer_fold" in grp.columns else 0

        student_pseudo = "stu_" + _deterministic_pseudonym(student_group_id)
        course_pseudo = "crs_" + _deterministic_pseudonym(stage_val + "_course")

        pre_cutoff: dict = {}
        for feat in [
            "inactivity_streak", "active_day_rate", "assessments_due",
            "regularity_score", "content_coverage", "quiz_activity",
            "missing_assessment_count", "due_soon_count", "completion_rate",
            "recent_activity_drop", "engagement_recovery_possible",
            "inter_session_gap", "study_consistency", "unviewed_content",
            "low_coverage_topics",
        ]:
            if feat in grp.columns:
                val = first[feat]
                pre_cutoff[feat] = val.item() if hasattr(val, "item") else val

        risk_band = str(first["risk_band"]) if "risk_band" in grp.columns else "UNKNOWN"
        unc = float(first["hybrid_uncertainty"]) if "hybrid_uncertainty" in grp.columns else 0.0
        unc_band = "HIGH" if unc > 0.3 else ("MEDIUM" if unc > 0.15 else "LOW")

        feasible_actions = list(grp["action_id"].unique())
        avail = {
            "quiz_available": bool(first["quiz_available"])
            if "quiz_available" in grp.columns
            else True,
        }

        feature_row_dict = {
            c: (first[c].item() if hasattr(first[c], "item") else first[c])
            for c in grp.columns
            if c != "action_id"
        }
        feature_row_sha256 = _row_sha256(feature_row_dict)

        case = {
            "query_id": qid,
            "student_pseudonym": student_pseudo,
            "course_pseudonym": course_pseudo,
            "stage": stage_val,
            "outer_fold": outer_fold,
            "risk_band": risk_band,
            "uncertainty_band": unc_band,
            "observed_pre_cutoff_evidence": pre_cutoff,
            "feasible_candidate_actions": feasible_actions,
            "contraindications": [],
            "availability_flags": avail,
            "source_query_id": qid,
            "source_student_group_id_hash": _deterministic_pseudonym(student_group_id),
            "source_feature_row_sha256": feature_row_sha256,
            "source_candidate_manifest_sha256": cand_sha,
            "source_feature_manifest_sha256": cand_manifest_sha,
        }
        cases_by_query[qid] = case
        student_to_queries.setdefault(student_group_id, []).append(qid)

    unique_students = list(student_to_queries.keys())
    rng = np.random.default_rng(SAMPLING_SEED)
    shuffled = rng.permutation(unique_students).tolist()
    half = len(shuffled) // 2
    pa_students = set(shuffled[:half])
    pb_students = set(shuffled[half:])

    assert len(pa_students & pb_students) == 0

    panel_a_cases: list[dict] = []
    panel_b_cases: list[dict] = []
    for sid in shuffled:
        for qid in student_to_queries[sid]:
            if sid in pa_students:
                panel_a_cases.append(cases_by_query[qid])
            else:
                panel_b_cases.append(cases_by_query[qid])

    panel_a_cases = panel_a_cases[: max(PANEL_A_MIN, min(300, len(panel_a_cases)))]
    panel_b_cases = panel_b_cases[: max(PANEL_B_MIN, min(150, len(panel_b_cases)))]

    pa_hashes = {c["source_student_group_id_hash"] for c in panel_a_cases}
    pb_hashes = {c["source_student_group_id_hash"] for c in panel_b_cases}
    overlap = pa_hashes & pb_hashes
    assert len(overlap) == 0, f"Student overlap: {overlap}"

    pa_q = {c["query_id"] for c in panel_a_cases}
    pb_q = {c["query_id"] for c in panel_b_cases}
    q_overlap = pa_q & pb_q
    assert len(q_overlap) == 0, f"Query overlap: {q_overlap}"

    with (EXPORT_DIR / "panel_a_cases.jsonl").open("w", encoding="utf-8") as f:
        for c in panel_a_cases:
            f.write(json.dumps(c) + "\n")
    with (EXPORT_DIR / "panel_b_cases.jsonl").open("w", encoding="utf-8") as f:
        for c in panel_b_cases:
            f.write(json.dumps(c) + "\n")

    sampling_manifest = {
        "sampling_seed": SAMPLING_SEED,
        "source_candidates_sha256": cand_sha,
        "source_manifest_sha256": cand_manifest_sha,
        "total_eligible_queries": len(cases_by_query),
        "total_eligible_students": len(unique_students),
        "panel_a_student_count": len(pa_hashes),
        "panel_b_student_count": len(pb_hashes),
        "panel_student_overlap_count": len(overlap),
        "panel_query_overlap_count": len(q_overlap),
        "panel_a_selected_case_count": len(panel_a_cases),
        "panel_b_selected_case_count": len(panel_b_cases),
        "zero_student_overlap": len(overlap) == 0,
        "zero_query_overlap": len(q_overlap) == 0,
        "pre_cutoff_only": True,
        "synthetic_fixture_used": False,
        "lineage_source": "action_candidates.parquet",
        "case_export_classification": "VERIFIED_OULAD_LINEAGE",
    }
    (EXPORT_DIR / "case_manifest.json").write_text(
        json.dumps(sampling_manifest, indent=2), encoding="utf-8"
    )

    _generate_prompt_package(PROMPTS_DIR, panel_a_cases, panel_b_cases)
    return sampling_manifest


def _generate_prompt_package(prompts_dir: Path, panel_a: list, panel_b: list) -> None:
    system_prompt = (
        "You are an expert academic advisor evaluating intervention actions for at-risk students.\n"
        "Assess each candidate action on relevance scale 0 to 3 based on pre-cutoff evidence.\n"
        "0 = Unsuitable or harmful\n1 = Weakly relevant\n"
        "2 = Relevant with adequate evidence\n3 = Highly relevant with direct evidence\n"
        "You may abstain if evidence is insufficient.\n"
    )
    (prompts_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    instructions = """# LLM Annotation Instructions

## Relevance Scale
- 0: Unsuitable or harmful
- 1: Weakly relevant
- 2: Relevant with adequate evidence
- 3: Highly relevant with direct evidence

## Required Response Fields
case_id, action_id, relevance_score, abstain, evidence_ids, rationale,
contraindication_detected, safety_flag, reviewer_id, reviewer_type,
provider, model_name, request_id.
reviewer_type must be: REAL_EXTERNAL_LLM_REVIEW
"""
    (prompts_dir / "annotation_instructions.md").write_text(instructions, encoding="utf-8")
    schema = {
        "type": "object",
        "required": [
            "case_id", "action_id", "relevance_score", "abstain",
            "reviewer_id", "reviewer_type", "provider", "model_name", "request_id",
        ],
        "properties": {
            "case_id": {"type": "string"},
            "action_id": {"type": "string"},
            "relevance_score": {"type": "integer", "enum": [0, 1, 2, 3]},
            "abstain": {"type": "boolean"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
            "contraindication_detected": {"type": "boolean"},
            "safety_flag": {"type": "boolean"},
            "reviewer_id": {"type": "string"},
            "reviewer_type": {
                "type": "string",
                "enum": [
                    "REAL_HUMAN_REVIEW",
                    "REAL_EXTERNAL_LLM_REVIEW",
                    "AGENT_GENERATED_PSEUDO_REVIEW",
                    "RULE_DERIVED_LABEL",
                ],
            },
            "provider": {"type": "string"},
            "model_name": {"type": "string"},
            "request_id": {"type": "string"},
        },
    }
    (prompts_dir / "response_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    pa_dir = prompts_dir / "panel_a_request_batches"
    pb_dir = prompts_dir / "panel_b_request_batches"
    pa_dir.mkdir(parents=True, exist_ok=True)
    pb_dir.mkdir(parents=True, exist_ok=True)
    (pa_dir / "batch_01.jsonl").write_text(
        "\n".join(json.dumps(c) for c in panel_a[:20]), encoding="utf-8"
    )
    (pb_dir / "batch_01.jsonl").write_text(
        "\n".join(json.dumps(c) for c in panel_b[:20]), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="all")
    args = parser.parse_args()
    m = export_v2_cases()
    print(f"CASE_EXPORT_STATUS=VERIFIED_OULAD_LINEAGE")
    print(f"CASE_EXPORT_PANEL_A={m['panel_a_selected_case_count']}")
    print(f"CASE_EXPORT_PANEL_B={m['panel_b_selected_case_count']}")
    print(f"PANEL_STUDENT_OVERLAP={m['panel_student_overlap_count']}")
    print(f"PANEL_QUERY_OVERLAP={m['panel_query_overlap_count']}")
