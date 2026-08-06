"""Tests for V2 LLM Case Generation."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.recommend_hybrid.explainable_v2.export_llm_cases import export_v2_cases

ROOT = Path(__file__).resolve().parents[3]


def test_export_v2_cases():
    manifest = export_v2_cases(panel_mode="all")
    assert manifest["panel_a_count"] > 0
    assert manifest["panel_b_count"] > 0
    assert manifest["zero_student_overlap"] is True

    # Check case files exist
    pa_path = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports/panel_a_cases.jsonl"
    pb_path = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports/panel_b_cases.jsonl"
    assert pa_path.exists()
    assert pb_path.exists()

    # Check blinding: no real student ID or final outcome in payload
    with pa_path.open("r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        assert "real_student_id" not in first_line
        assert "final_result" not in first_line
        assert "observed_pre_cutoff_evidence" in first_line
