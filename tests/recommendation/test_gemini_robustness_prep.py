from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.recommendation.build_labeling_jobs import ROOT
from src.recommendation.labeling.constants import PROMPT_VERSION, PROMPT_VERSION_B


JOBS = ROOT / "artifacts/recommendation/labeling/jobs"


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_repeat_and_v1b_use_same_150_panel_a_cases_without_panel_b():
    repeat = _read(JOBS / "gemini_repeat_v1_jobs.jsonl")
    v1b = _read(JOBS / "gemini_prompt_v1b_jobs.jsonl")
    panel_a = set(pd.read_parquet(ROOT / "artifacts/recommendation/panels/panel_a.parquet")["case_id"].astype(str))
    panel_b = set(pd.read_parquet(ROOT / "artifacts/recommendation/panels/panel_b.parquet")["case_id"].astype(str))
    repeat_ids = set(sum((job["case_ids"] for job in repeat), []))
    v1b_ids = set(sum((job["case_ids"] for job in v1b), []))
    assert len(repeat) == len(v1b) == 15
    assert len(repeat_ids) == len(v1b_ids) == 150
    assert repeat_ids == v1b_ids
    assert repeat_ids <= panel_a
    assert not repeat_ids & panel_b
    assert repeat[0]["prompt_version"] == PROMPT_VERSION
    assert v1b[0]["prompt_version"] == PROMPT_VERSION_B
    assert all(job["model"] == "gemini-3.5-flash-lite" for job in repeat + v1b)


def test_repeat_and_v1b_cover_required_dimensions_and_keep_main_files_separate():
    repeat = _read(JOBS / "gemini_repeat_v1_jobs.jsonl")
    panel = pd.read_parquet(ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    ids = set(sum((job["case_ids"] for job in repeat), []))
    selected = panel[panel["case_id"].astype(str).isin(ids)]
    assert set(selected["stage"]) == {"20pct", "35pct", "50pct", "75pct"}
    assert set(selected["outer_fold"]) == {0, 1, 2}
    assert set(selected["sampling_risk_band"]) == {"Low", "Borderline", "High"}
    assert not (JOBS / "gemini_repeat_v1_jobs.jsonl").samefile(JOBS / "panel_a_gemini_jobs.jsonl")
    assert not (JOBS / "gemini_prompt_v1b_jobs.jsonl").samefile(JOBS / "panel_a_gemini_jobs.jsonl")
    assert all("GEMINI_API_KEY" not in json.dumps(job) for job in repeat)


def test_v1b_tightens_wording_without_changing_action_rubric_contract():
    repeat = _read(JOBS / "gemini_repeat_v1_jobs.jsonl")[0]["prompt"]
    v1b = _read(JOBS / "gemini_prompt_v1b_jobs.jsonl")[0]["prompt"]
    for phrase in ("A1:", "A2:", "A3:", "A4:", "A5:", "NOT_RELEVANT", "SLIGHTLY_RELEVANT", "RELEVANT", "HIGHLY_RELEVANT / PRIORITY", "ABSTAIN"):
        assert phrase in repeat and phrase in v1b
    for phrase in ("Use only supplied evidence.", "Do not infer unavailable features.",
                   "Evaluate each action independently.", "UNKNOWN is not INFEASIBLE.",
                   "Do not generate recommendation prose."):
        assert phrase in v1b
    assert PROMPT_VERSION_B in v1b and PROMPT_VERSION_B not in repeat
