"""Build 50 Gemini 3.1 Flash-Lite jobs for A4 Progress Monitoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.labeling.constants import A4_PROGRESS_GEMINI31_PROMPT_VERSION  # noqa: E402
from src.recommendation.labeling.progress_monitoring import PROGRESS_STATE_FIELDS  # noqa: E402
from src.recommendation.labeling.progress_monitoring_gemini31 import (  # noqa: E402
    PROGRESS_GEMINI31_MODEL, PROGRESS_GEMINI31_SCHEMA_VERSION,
    build_progress_monitoring_gemini31_prompt,
)

BATCH_SIZE = 10


def _payload(row: pd.Series) -> dict:
    payload = {"case_id": str(row["case_id"])}
    for field in PROGRESS_STATE_FIELDS:
        if field not in row:
            raise ValueError(f"Panel A is missing required Student State field: {field}")
        value = row[field]
        payload[field] = value.item() if hasattr(value, "item") else value
    return payload


def build_jobs(panel_a_path: Path, output_path: Path, *, batch_size: int = BATCH_SIZE) -> list[dict]:
    if batch_size != BATCH_SIZE:
        raise ValueError("Progress Monitoring Gemini 3.1 jobs are locked to 10 cases/request")
    panel = pd.read_parquet(panel_a_path).copy()
    panel["case_id"] = panel["case_id"].astype(str)
    if len(panel) != 500 or panel["case_id"].duplicated().any():
        raise ValueError("Panel A must contain exactly 500 unique cases")
    if set(panel["stage"]) - {"20pct", "35pct", "50pct", "75pct"}:
        raise ValueError("Panel A contains FINAL or invalid stages")
    payloads = [_payload(row) for _, row in panel.sort_values("case_id").iterrows()]
    jobs = []
    for index, start in enumerate(range(0, len(payloads), batch_size), start=1):
        batch = payloads[start:start + batch_size]
        jobs.append({
            "job_id": f"progress_monitoring_gemini31_batch_{index:04d}",
            "provider": "gemini",
            "model": PROGRESS_GEMINI31_MODEL,
            "prompt_version": A4_PROGRESS_GEMINI31_PROMPT_VERSION,
            "batch_index": index,
            "case_ids": [item["case_id"] for item in batch],
            "payload": batch,
            "prompt": build_progress_monitoring_gemini31_prompt(batch),
            "schema_version": PROGRESS_GEMINI31_SCHEMA_VERSION,
        })
    if len(jobs) != 50 or sum(len(job["case_ids"]) for job in jobs) != 500:
        raise ValueError("Progress Monitoring Gemini 3.1 jobs must contain 50 jobs and 500 cases")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False, sort_keys=True) + "\n")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-a", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/recommendation/labeling/jobs/progress_monitoring_gemini31_jobs.jsonl")
    args = parser.parse_args()
    jobs = build_jobs(args.panel_a, args.output)
    print(json.dumps({"jobs": len(jobs), "cases": sum(len(job["case_ids"]) for job in jobs), "model": PROGRESS_GEMINI31_MODEL, "rpm_limit": 25}, indent=2))


if __name__ == "__main__":
    main()
