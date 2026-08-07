"""Tests for V2 LLM Case Generation."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.recommend_hybrid.explainable_v2.export_llm_cases import export_v2_cases

ROOT = Path(__file__).resolve().parents[3]


def test_export_v2_cases():
    manifest = export_v2_cases()
    assert manifest["panel_a_case_count"] > 0
    assert manifest["panel_b_case_count"] > 0
    assert manifest["zero_student_overlap"] is True
    assert manifest["zero_query_overlap"] is True
    assert manifest["public_privacy_verified"] is True
    assert manifest["synthetic_fixture_used"] is False
    assert manifest["case_export_classification"] == "VERIFIED_OULAD_LINEAGE"

    # Check case files exist
    pa_path = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports/panel_a_cases.jsonl"
    pb_path = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports/panel_b_cases.jsonl"
    assert pa_path.exists()
    assert pb_path.exists()

    # Check public privacy blinding: ZERO unblinded identifiers in public payload
    with pa_path.open("r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        assert "real_student_id" not in first_line
        assert "source_query_id" not in first_line
        assert "source_student_group_id_hash" not in first_line
        assert "student_pseudonym" not in first_line
        assert "course_pseudonym" not in first_line
        assert "observed_pre_cutoff_evidence" in first_line
        assert "feasible_candidate_actions" in first_line
        assert first_line["case_id"].startswith("case_")

    # Check private mapping exists and holds true lineage
    private_map_path = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/private/private_case_mapping.json"
    assert private_map_path.exists()
    pmap = json.loads(private_map_path.read_text(encoding="utf-8"))
    assert first_line["case_id"] in pmap
    assert "source_query_id" in pmap[first_line["case_id"]]
    assert "source_feature_row_sha256" in pmap[first_line["case_id"]]
