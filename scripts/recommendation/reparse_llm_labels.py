"""Re-parse existing raw responses offline; never calls a provider API."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.labeling.parser import parse_llm_response  # noqa: E402
from src.recommendation.labeling.runtime import load_jsonl  # noqa: E402


def reparse(input_path: Path, jobs_path: Path, output_path: Path) -> dict:
    records = load_jsonl(input_path)
    jobs = load_jsonl(jobs_path)
    by_job: dict[str, list[dict]] = {}
    for record in records:
        by_job.setdefault(str(record["job_id"]), []).append(record)
    parsed_by_job: dict[str, dict] = {}
    for job in jobs:
        job_id = str(job["job_id"])
        job_records = by_job.get(job_id, [])
        raw_responses = [record.get("raw_response") for record in job_records if record.get("raw_response")]
        if not raw_responses:
            raise ValueError(f"no raw response available for {job_id}")
        if len(set(raw_responses)) != 1:
            raise ValueError(f"raw response differs within {job_id}")
        case_ids = [str(case_id) for case_id in job["case_ids"]]
        feasibility = {str(payload["case_id"]): payload["feasibility"] for payload in job["payload"]}
        parsed_by_job[job_id] = parse_llm_response(raw_responses[0], case_ids, feasibility, prompt_version=job["prompt_version"])
    reparsed_at = datetime.now(timezone.utc).isoformat()
    output_records = []
    for record in records:
        job_id = str(record["job_id"])
        case_id = str(record["case_id"])
        parsed = parsed_by_job[job_id][case_id]
        updated = dict(record)
        updated["parsed_labels"] = parsed
        updated["status"] = "completed"
        updated["error"] = None
        updated["reparsed_at"] = reparsed_at
        output_records.append(updated)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {"records": len(output_records), "cases": len({record["case_id"] for record in output_records}), "jobs": len(parsed_by_job)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reparse(args.input, args.jobs, args.output), indent=2))


if __name__ == "__main__":
    main()
