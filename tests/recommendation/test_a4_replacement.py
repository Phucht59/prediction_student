from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.recommendation.build_a4_replacement_jobs import MODEL
from scripts.recommendation.evaluate_a4_replacement import evaluate
from src.recommendation.labeling.a4_replacement import (
    A4_REPLACEMENT_PROMPT_VERSION,
    REPLACEMENT_ACTIONS,
    REPLACEMENT_STATE_FIELDS,
    parse_replacement_response,
)
from src.recommendation.labeling.parser import LabelParseError
from src.recommendation.labeling.runtime import run_jobs


ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "artifacts/recommendation/labeling/jobs/a4_replacement_gemini_jobs.jsonl"


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _response(case_ids):
    return json.dumps({"results": [{
        "case_id": case_id,
        "labels": {
            "B1_PROGRESS_MONITORING": {"label": "1"},
            "B2_ACADEMIC_HELP_SEEKING": {"label": "ABSTAIN"},
        },
    } for case_id in case_ids]})


def test_replacement_jobs_are_50_batches_of_10_panel_a_cases():
    jobs = _read(JOBS)
    panel_a = set(pd.read_parquet(ROOT / "artifacts/recommendation/panels/panel_a.parquet")["case_id"].astype(str))
    panel_b = set(pd.read_parquet(ROOT / "artifacts/recommendation/panels/panel_b.parquet")["case_id"].astype(str))
    assert len(jobs) == 50
    assert all(len(job["case_ids"]) == 10 for job in jobs)
    assert len(set(sum((job["case_ids"] for job in jobs), []))) == 500
    assert set(sum((job["case_ids"] for job in jobs), [])) == panel_a
    assert not panel_b & panel_a
    assert all(job["model"] == MODEL for job in jobs)
    assert all(job["prompt_version"] == A4_REPLACEMENT_PROMPT_VERSION for job in jobs)
    assert all(set(job["payload"][0]) == {"case_id", *REPLACEMENT_STATE_FIELDS} for job in jobs)
    assert all("feasibility" not in payload for job in jobs for payload in job["payload"])


def test_replacement_prompt_locks_two_candidates_and_existing_main_is_untouched():
    job = _read(JOBS)[0]
    prompt = job["prompt"]
    assert "B1_PROGRESS_MONITORING" in prompt
    assert "B2_ACADEMIC_HELP_SEEKING" in prompt
    assert "This is NOT Study Planning" in prompt or "not A3 Study Planning" in prompt
    assert "High risk alone must not imply label 3." in prompt
    assert not (JOBS).samefile(ROOT / "artifacts/recommendation/labeling/jobs/panel_a_gemini_jobs.jsonl")


def test_replacement_parser_normalizes_and_rejects_wrong_contract():
    parsed = parse_replacement_response(_response(["c1", "c2"]), ["c1", "c2"])
    assert parsed["c1"]["labels"]["B1_PROGRESS_MONITORING"]["label"] == 1
    assert parsed["c1"]["labels"]["B2_ACADEMIC_HELP_SEEKING"]["label"] == "ABSTAIN"
    with pytest.raises(LabelParseError):
        parse_replacement_response(_response(["c1"]), ["c1", "c2"])
    bad = json.loads(_response(["c1"]))
    bad["results"][0]["labels"]["B1_PROGRESS_MONITORING"]["label"] = 4
    with pytest.raises(LabelParseError):
        parse_replacement_response(json.dumps(bad), ["c1"])
    extra = json.loads(_response(["c1"]))
    extra["results"][0]["labels"]["A4"] = {"label": 1}
    with pytest.raises(LabelParseError):
        parse_replacement_response(json.dumps(extra), ["c1"])


def test_existing_runtime_can_execute_replacement_job_with_custom_parser(tmp_path):
    job = _read(JOBS)[0]
    output = tmp_path / "raw.jsonl"

    def fake_request(current):
        return _response(current["case_ids"]), None

    run_jobs([job], output, fake_request, max_retries=0, retry_delay=0, batch_size=10,
             response_parser=parse_replacement_response)
    records = _read(output)
    assert len(records) == 10
    assert all(record["status"] == "completed" for record in records)
    assert all(record["parsed_labels"]["labels"].keys() == set(REPLACEMENT_ACTIONS) for record in records)


def test_evaluation_reports_state_variation_without_mean_based_selection():
    rows = []
    state = []
    for index in range(20):
        case_id = f"c{index}"
        state.append({"case_id": case_id, "stage": "20pct" if index < 10 else "75pct", "risk_band": "low", "risk_probability": index / 20})
        rows.extend([
            {"case_id": case_id, "action_id": "B1_PROGRESS_MONITORING", "label": "0" if index < 10 else "3"},
            {"case_id": case_id, "action_id": "B2_ACADEMIC_HELP_SEEKING", "label": "ABSTAIN" if index % 2 else "1"},
        ])
    metrics = evaluate(pd.DataFrame(rows), pd.DataFrame(state))
    assert metrics["by_action"]["B1_PROGRESS_MONITORING"]["degeneracy_flag"] is False
    assert metrics["by_action"]["B1_PROGRESS_MONITORING"]["state_variation_features"]
    assert metrics["by_action"]["B2_ACADEMIC_HELP_SEEKING"]["abstain_rate"] == 0.5
