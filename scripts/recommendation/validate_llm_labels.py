"""Offline validation for user-produced raw weak-label JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.labeling.constants import ACTION_IDS  # noqa: E402
from src.recommendation.labeling.parser import LabelParseError, parse_llm_response  # noqa: E402
from src.recommendation.labeling.runtime import load_jsonl  # noqa: E402


def validate(raw_path: Path, jobs_path: Path, panel_a_path: Path, panel_b_path: Path) -> dict:
    jobs = load_jsonl(jobs_path)
    prompt_versions = {str(job.get("prompt_version")) for job in jobs}
    if len(prompt_versions) != 1:
        raise ValueError("jobs must contain exactly one prompt version")
    expected_prompt_version = next(iter(prompt_versions))
    expected: dict[str, tuple[dict, str]] = {}
    for job in jobs:
        for payload in job["payload"]:
            case_id = str(payload["case_id"])
            if case_id in expected:
                raise ValueError(f"duplicate expected case_id in jobs: {case_id}")
            expected[case_id] = (payload["feasibility"], job["job_id"])
    panel_a = set(pd.read_parquet(panel_a_path)["case_id"].astype(str))
    panel_b = set(pd.read_parquet(panel_b_path)["case_id"].astype(str))
    if not set(expected).issubset(panel_a):
        raise ValueError("jobs contain a case outside Panel A")
    if set(expected) & panel_b:
        raise ValueError("jobs contain Panel B case")
    records = load_jsonl(raw_path)
    seen = set()
    failures = []
    for record in records:
        case_id = str(record.get("case_id", ""))
        if case_id not in expected:
            raise ValueError(f"unexpected case_id: {case_id}")
        key = (str(record.get("job_id")), case_id)
        if key in seen:
            raise ValueError(f"duplicate raw job/case record: {key}")
        seen.add(key)
        if record.get("prompt_version") != expected_prompt_version:
            raise ValueError(f"invalid prompt version for {case_id}")
        if not record.get("provider") or not record.get("model"):
            raise ValueError(f"provider/model missing for {case_id}")
        if record.get("status") != "completed":
            failures.append(case_id)
            continue
        parsed = record.get("parsed_labels")
        try:
            parsed_result = parse_llm_response(json.dumps(parsed), [case_id], {case_id: expected[case_id][0]}, prompt_version=expected_prompt_version)
        except (LabelParseError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid labels for {case_id}: {exc}") from exc
        if set(parsed_result[case_id]["labels"]) != set(ACTION_IDS):
            raise ValueError(f"A1-A5 missing for {case_id}")
    expected_jobs = {job["job_id"] for job in jobs}
    completed_cases = {str(record["case_id"]) for record in records if record.get("status") == "completed"}
    expected_cases = set(expected)
    if completed_cases != expected_cases or failures:
        raise ValueError(f"incomplete labels: missing={len(expected_cases - completed_cases)}, failures={len(failures)}")
    return {"expected_cases": len(expected_cases), "raw_records": len(records), "failures": len(failures), "jobs": len(expected_jobs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--panel-a", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    parser.add_argument("--panel-b", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_b.parquet")
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    summary = validate(args.input, args.jobs, args.panel_a, args.panel_b)
    if args.output_report:
        args.output_report.write_text("# LLM label validation\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
