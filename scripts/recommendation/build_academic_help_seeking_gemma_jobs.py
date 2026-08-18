"""Build pilot and full single-case Gemma jobs for A4 Academic Help-Seeking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.recommendation.build_labeling_jobs import choose_pilot  # noqa: E402
from src.recommendation.labeling.academic_help_seeking import ACADEMIC_HELP_STATE_FIELDS, build_academic_help_prompt  # noqa: E402
from src.recommendation.labeling.constants import A4_ACADEMIC_HELP_GEMMA_PROMPT_VERSION  # noqa: E402

MODEL = "gemma-4-31b-it"
SCHEMA_VERSION = "recommendation.academic_help_seeking_gemma.v1"


def _jobs(frame: pd.DataFrame, scope: str) -> list[dict]:
    jobs = []
    for index, (_, row) in enumerate(frame.sort_values("case_id").iterrows(), start=1):
        payload = {"case_id": str(row["case_id"])}
        for field in ACADEMIC_HELP_STATE_FIELDS:
            value = row[field]
            payload[field] = value.item() if hasattr(value, "item") else value
        jobs.append({
            "job_id": f"academic_help_seeking_gemma_{scope}_single_{index:04d}",
            "provider": "gemma",
            "model": MODEL,
            "prompt_version": A4_ACADEMIC_HELP_GEMMA_PROMPT_VERSION,
            "batch_index": index,
            "case_ids": [payload["case_id"]],
            "payload": [payload],
            "prompt": build_academic_help_prompt(payload),
            "schema_version": SCHEMA_VERSION,
        })
    return jobs


def _write(path: Path, jobs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False, sort_keys=True) + "\n")


def build(panel_a_path: Path, pilot_output: Path, full_output: Path, *, seed: int = 2026, pilot_size: int = 30) -> tuple[list[dict], list[dict]]:
    panel = pd.read_parquet(panel_a_path).copy()
    panel["case_id"] = panel["case_id"].astype(str)
    if len(panel) != 500 or panel["case_id"].duplicated().any():
        raise ValueError("Panel A must contain exactly 500 unique cases")
    if set(panel["stage"]) - {"20pct", "35pct", "50pct", "75pct"}:
        raise ValueError("Panel A contains FINAL or invalid stages")
    pilot = choose_pilot(panel, pilot_size, seed)
    pilot_jobs = _jobs(pilot, "pilot")
    full_jobs = _jobs(panel, "full")
    if len(pilot_jobs) != pilot_size or len(full_jobs) != 500:
        raise ValueError("Academic Help-Seeking jobs must be 30 pilot and 500 full single-case jobs")
    _write(pilot_output, pilot_jobs)
    _write(full_output, full_jobs)
    return pilot_jobs, full_jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-a", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    parser.add_argument("--pilot-output", type=Path, default=ROOT / "artifacts/recommendation/labeling/jobs/academic_help_seeking_gemma_pilot_jobs.jsonl")
    parser.add_argument("--full-output", type=Path, default=ROOT / "artifacts/recommendation/labeling/jobs/academic_help_seeking_gemma_single_jobs.jsonl")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    pilot, full = build(args.panel_a, args.pilot_output, args.full_output, seed=args.seed)
    print(json.dumps({"pilot_jobs": len(pilot), "full_jobs": len(full), "model": MODEL}, indent=2))


if __name__ == "__main__":
    main()
