from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.recommendation.build_progress_monitoring_gemini31_jobs import build_jobs
from src.recommendation.labeling.constants import A4_PROGRESS_GEMINI31_PROMPT_VERSION
from src.recommendation.labeling.parser import LabelParseError
from src.recommendation.labeling.progress_monitoring import PROGRESS_STATE_FIELDS
from src.recommendation.labeling.progress_monitoring_gemini31 import (
    PROGRESS_GEMINI31_MODEL,
    parse_progress_monitoring_gemini31_response,
)
from src.recommendation.weak_supervision.matrix import SOURCES_BY_ACTION


ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "artifacts/recommendation/labeling/jobs/progress_monitoring_gemini31_jobs.jsonl"


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _response(case_ids, label="2") -> str:
    return json.dumps({"results": [{"case_id": case_id, "labels": {"A4": {"label": label}}} for case_id in case_ids]})


def test_gemini31_jobs_are_50_batches_of_10_panel_a_cases():
    jobs = _read(JOBS)
    panel_a = set(pd.read_parquet(ROOT / "artifacts/recommendation/panels/panel_a.parquet")["case_id"].astype(str))
    panel_b = set(pd.read_parquet(ROOT / "artifacts/recommendation/panels/panel_b.parquet")["case_id"].astype(str))
    assert len(jobs) == 50
    assert all(len(job["case_ids"]) == 10 and len(job["payload"]) == 10 for job in jobs)
    assert set(sum((job["case_ids"] for job in jobs), [])) == panel_a
    assert not panel_b & panel_a
    assert all(job["model"] == PROGRESS_GEMINI31_MODEL for job in jobs)
    assert all(job["prompt_version"] == A4_PROGRESS_GEMINI31_PROMPT_VERSION for job in jobs)
    assert all(set(payload) == {"case_id", *PROGRESS_STATE_FIELDS} for job in jobs for payload in job["payload"])
    assert all("Gemma" not in job["prompt"] and "Academic Help-Seeking" not in job["prompt"] for job in jobs)


def test_gemini31_parser_accepts_batch_and_normalizes_labels():
    parsed = parse_progress_monitoring_gemini31_response(_response(["c1", "c2"], label="3"), ["c1", "c2"])
    assert parsed["c1"]["labels"]["A4"]["label"] == 3
    parsed = parse_progress_monitoring_gemini31_response(_response(["c1"], label="ABSTAIN"), ["c1"])
    assert parsed["c1"]["labels"]["A4"]["label"] == "ABSTAIN"


@pytest.mark.parametrize("bad", [
    _response(["c1"]),
    _response(["c1", "c1"]),
    _response(["c1", "c2", "extra"]),
])
def test_gemini31_parser_rejects_missing_duplicate_or_extra_case(bad):
    with pytest.raises(LabelParseError):
        parse_progress_monitoring_gemini31_response(bad, ["c1", "c2"])


def test_progress_matrix_uses_only_effective_gemini35_and_gemini31_for_a4():
    assert SOURCES_BY_ACTION["progress_monitoring"] == ("LF_GEMINI35", "LF_GEMINI31")
    assert "LF_BEHAVIOR" not in SOURCES_BY_ACTION["progress_monitoring"]
    assert "LF_GEMMA" not in SOURCES_BY_ACTION["progress_monitoring"]


def test_current_gemini35_source_is_b1_only_and_covers_panel_a():
    frame = pd.read_parquet(ROOT / "artifacts/recommendation/labeling/normalized/a4_replacement_gemini_labels.parquet")
    b1 = frame[frame["action_id"] == "B1_PROGRESS_MONITORING"]
    assert len(b1) == 500
    assert frame["action_id"].eq("B2_ACADEMIC_HELP_SEEKING").sum() == 500
