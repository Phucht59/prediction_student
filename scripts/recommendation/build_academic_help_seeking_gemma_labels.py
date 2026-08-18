"""Offline parser for completed Academic Help-Seeking Gemma jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.labeling.academic_help_seeking import parse_academic_help_function_call  # noqa: E402
from src.recommendation.labeling.constants import A4_ACADEMIC_HELP_GEMMA_PROMPT_VERSION  # noqa: E402
from src.recommendation.labeling.runtime import load_jsonl  # noqa: E402


def build(raw_path: Path, jobs_path: Path, output_path: Path) -> pd.DataFrame:
    jobs = load_jsonl(jobs_path)
    records = load_jsonl(raw_path)
    if len(records) != len(jobs):
        raise ValueError(f"raw records {len(records)} do not match jobs {len(jobs)}")
    rows = []
    seen = set()
    for job in jobs:
        job_id = str(job["job_id"])
        matching = [record for record in records if str(record.get("job_id")) == job_id]
        if len(matching) != 1 or matching[0].get("status") != "completed":
            raise ValueError(f"invalid or incomplete Academic Help-Seeking record: {job_id}")
        case_id = str(job["case_ids"][0])
        parsed = parse_academic_help_function_call(matching[0]["raw_response"], [case_id])
        if case_id in seen:
            raise ValueError(f"duplicate candidate case: {case_id}")
        seen.add(case_id)
        label = parsed[case_id]["labels"]["A4"]["label"]
        rows.append({"case_id": case_id, "action_id": "academic_help_seeking", "lf_name": "LF_GEMMA",
                     "label": str(label), "abstain": label == "ABSTAIN", "provider": "gemma",
                     "model": job["model"], "prompt_version": A4_ACADEMIC_HELP_GEMMA_PROMPT_VERSION})
    frame = pd.DataFrame(rows).sort_values("case_id")
    if len(frame) != len(jobs) or frame["case_id"].nunique() != len(jobs):
        raise ValueError("candidate normalized grain is invalid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "artifacts/recommendation/labeling/raw/academic_help_seeking_gemma_pilot.jsonl")
    parser.add_argument("--jobs", type=Path, default=ROOT / "artifacts/recommendation/labeling/jobs/academic_help_seeking_gemma_pilot_jobs.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/academic_help_seeking_gemma_pilot_labels.parquet")
    args = parser.parse_args()
    frame = build(args.raw, args.jobs, args.output)
    print(json.dumps({"rows": len(frame), "cases": frame["case_id"].nunique()}, indent=2))


if __name__ == "__main__":
    main()
