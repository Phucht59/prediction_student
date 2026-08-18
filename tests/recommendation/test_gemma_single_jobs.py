from __future__ import annotations

import json
from pathlib import Path

from scripts.recommendation.build_labeling_jobs import ROOT


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_gemma_single_job_files_have_one_case_per_request():
    pilot = _read(ROOT / "artifacts/recommendation/labeling/jobs/pilot_gemma_single_jobs.jsonl")
    panel = _read(ROOT / "artifacts/recommendation/labeling/jobs/panel_a_gemma_single_jobs.jsonl")
    assert len(pilot) == 30
    assert len(panel) == 500
    for jobs in (pilot, panel):
        assert all(len(job["case_ids"]) == 1 for job in jobs)
        assert all(len(job["payload"]) == 1 for job in jobs)
        assert len({job["case_ids"][0] for job in jobs}) == len(jobs)
        assert all(job["model"] == "gemma-4-31b-it" for job in jobs)
