"""Build 500 single-case Gemma jobs for final A4 Progress Monitoring."""

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
from src.recommendation.labeling.progress_monitoring import PROGRESS_STATE_FIELDS, build_progress_monitoring_prompt  # noqa: E402

MODEL = "gemma-4-31b-it"
SCHEMA_VERSION = "recommendation.progress_monitoring_gemma.v1"


def build_jobs(panel_a_path: Path, output_path: Path) -> list[dict]:
    panel = pd.read_parquet(panel_a_path).copy()
    panel["case_id"] = panel["case_id"].astype(str)
    if len(panel) != 500 or panel["case_id"].duplicated().any():
        raise ValueError("Panel A must contain exactly 500 unique cases")
    if set(panel["stage"]) - {"20pct", "35pct", "50pct", "75pct"}:
        raise ValueError("Panel A contains FINAL or invalid stages")
    jobs = []
    for index, (_, row) in enumerate(panel.sort_values("case_id").iterrows(), start=1):
        payload = {"case_id": str(row["case_id"])}
        for field in PROGRESS_STATE_FIELDS:
            value = row[field]
            payload[field] = value.item() if hasattr(value, "item") else value
        jobs.append({
            "job_id": f"progress_monitoring_gemma_single_{index:04d}",
            "provider": "gemma",
            "model": MODEL,
            "prompt_version": A4_PROGRESS_GEMMA_PROMPT_VERSION,
            "batch_index": index,
            "case_ids": [payload["case_id"]],
            "payload": [payload],
            "prompt": build_progress_monitoring_prompt(payload),
            "schema_version": SCHEMA_VERSION,
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False, sort_keys=True) + "\n")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-a", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/recommendation/labeling/jobs/progress_monitoring_gemma_single_jobs.jsonl")
    args = parser.parse_args()
    jobs = build_jobs(args.panel_a, args.output)
    print(json.dumps({"jobs": len(jobs), "cases": sum(len(job["case_ids"]) for job in jobs), "model": MODEL}, indent=2))


if __name__ == "__main__":
    main()
