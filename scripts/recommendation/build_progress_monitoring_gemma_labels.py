"""Offline parser for completed final A4 Gemma single-case responses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.labeling.constants import A4_PROGRESS_GEMMA_PROMPT_VERSION  # noqa: E402
from src.recommendation.labeling.progress_monitoring import parse_progress_function_call  # noqa: E402
from src.recommendation.labeling.runtime import load_jsonl  # noqa: E402


def build(raw_path: Path, jobs_path: Path, output_path: Path) -> pd.DataFrame:
    jobs = load_jsonl(jobs_path)
    records = load_jsonl(raw_path)
    if len(jobs) != 500 or len(records) != 500:
        raise ValueError("A4 Gemma requires exactly 500 jobs and 500 raw records")
    rows = []
    seen = set()
    for job, record in zip(jobs, records):
        if record.get("job_id") != job.get("job_id") or record.get("status") != "completed":
            raise ValueError(f"invalid or incomplete A4 Gemma record: {job.get('job_id')}")
        case_id = str(job["case_ids"][0])
        parsed = parse_progress_function_call(record["raw_response"], [case_id])
        if case_id in seen:
            raise ValueError(f"duplicate A4 Gemma case: {case_id}")
        seen.add(case_id)
        label = parsed[case_id]["labels"]["A4"]["label"]
        rows.append({
            "case_id": case_id,
            "action_id": "progress_monitoring",
            "lf_name": "LF_GEMMA",
            "label": str(label),
            "abstain": label == "ABSTAIN",
            "provider": "gemma",
            "model": job["model"],
            "prompt_version": A4_PROGRESS_GEMMA_PROMPT_VERSION,
        })
    frame = pd.DataFrame(rows).sort_values("case_id")
    if len(frame) != 500 or frame["case_id"].nunique() != 500:
        raise ValueError("A4 Gemma normalized output must contain 500 unique cases")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "artifacts/recommendation/labeling/raw/progress_monitoring_gemma.jsonl")
    parser.add_argument("--jobs", type=Path, default=ROOT / "artifacts/recommendation/labeling/jobs/progress_monitoring_gemma_single_jobs.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/progress_monitoring_gemma_labels.parquet")
    args = parser.parse_args()
    frame = build(args.raw, args.jobs, args.output)
    print(json.dumps({"rows": len(frame), "cases": frame["case_id"].nunique(), "action": "progress_monitoring"}, indent=2))


if __name__ == "__main__":
    main()
