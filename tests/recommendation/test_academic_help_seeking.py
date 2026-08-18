from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.recommendation.labeling.academic_help_seeking import (
    ACADEMIC_HELP_FUNCTION_NAME,
    ACADEMIC_HELP_STATE_FIELDS,
    academic_help_function_declaration,
    parse_academic_help_function_call,
)
from src.recommendation.labeling.constants import A4_ACADEMIC_HELP_GEMMA_PROMPT_VERSION
from src.recommendation.labeling.parser import LabelParseError
from src.recommendation.weak_supervision.matrix import FINAL_ACTIONS


ROOT = Path(__file__).resolve().parents[2]
JOB_DIR = ROOT / "artifacts/recommendation/labeling/jobs"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _response(case_ref: str = "C01", label: str | int = "2", *, cases: list[dict] | None = None) -> str:
    values = cases if cases is not None else [{"case_ref": case_ref, "label": label}]
    return json.dumps({"candidates": [{"content": {"parts": [{"functionCall": {
        "name": ACADEMIC_HELP_FUNCTION_NAME,
        "args": {"cases": values},
    }}]}}]})


def test_academic_jobs_cover_panel_a_with_single_case_requests():
    panel_a = pd.read_parquet(ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    panel_ids = set(panel_a["case_id"].astype(str))
    pilot = _read_jsonl(JOB_DIR / "academic_help_seeking_gemma_pilot_jobs.jsonl")
    full = _read_jsonl(JOB_DIR / "academic_help_seeking_gemma_single_jobs.jsonl")
    assert len(pilot) == 30
    assert len(full) == 500
    assert set(job["case_ids"][0] for job in full) == panel_ids
    assert len({job["case_ids"][0] for job in pilot}) == 30
    assert all(job["model"] == "gemma-4-31b-it" for job in pilot + full)
    assert all(job["prompt_version"] == A4_ACADEMIC_HELP_GEMMA_PROMPT_VERSION for job in pilot + full)
    assert all(len(job["case_ids"]) == 1 and len(job["payload"]) == 1 for job in pilot + full)
    assert all(set(job["payload"][0]) == {"case_id", *ACADEMIC_HELP_STATE_FIELDS} for job in pilot + full)
    assert all("Progress Monitoring" not in job["prompt"] for job in pilot + full)


def test_academic_function_schema_has_no_stale_required_keys():
    schema = academic_help_function_declaration()

    def visit(node):
        if not isinstance(node, dict):
            return
        if node.get("type") == "OBJECT":
            assert set(node.get("required", [])) <= set(node.get("properties", {}))
            for child in node.get("properties", {}).values():
                visit(child)
        visit(node.get("items"))

    visit(schema["parameters"])
    assert "case_id" not in json.dumps(schema)


def test_academic_function_call_alias_roundtrip_and_numeric_normalization():
    parsed = parse_academic_help_function_call(_response(label="2"), ["real-case"])
    assert parsed["real-case"]["labels"]["A4"]["label"] == 2
    assert parse_academic_help_function_call(_response(label=3), ["real-case"])["real-case"]["labels"]["A4"]["label"] == 3
    assert parse_academic_help_function_call(_response(label="ABSTAIN"), ["real-case"])["real-case"]["labels"]["A4"]["label"] == "ABSTAIN"


@pytest.mark.parametrize("bad_cases", [
    [{"case_ref": "C01", "label": "1"}, {"case_ref": "C01", "label": "2"}],
    [],
    [{"case_ref": "C02", "label": "1"}],
])
def test_academic_function_call_rejects_wrong_case_count_or_alias(bad_cases):
    with pytest.raises(LabelParseError):
        parse_academic_help_function_call(_response(cases=bad_cases), ["real-case"])


def test_academic_help_seeking_is_not_current_a4_matrix_candidate():
    assert FINAL_ACTIONS == (
        "assessment_recovery", "re_engagement", "study_planning", "progress_monitoring", "retrieval_practice",
    )
    assert "academic_help_seeking" not in FINAL_ACTIONS
