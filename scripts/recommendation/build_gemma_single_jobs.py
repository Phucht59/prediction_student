"""Create Gemma-only single-case jobs without touching existing batched jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.recommendation.build_labeling_jobs import _load_feasibility, choose_pilot  # noqa: E402
from src.recommendation.labeling.constants import PROMPT_VERSION, SCHEMA_VERSION  # noqa: E402
from src.recommendation.labeling.payload import build_label_payload  # noqa: E402
from src.recommendation.labeling.prompt import build_prompt  # noqa: E402


def _build_jobs(frame: pd.DataFrame, feasibility: dict[str, dict[str, str]], model: str, scope: str) -> list[dict]:
    jobs = []
    for index, (_, row) in enumerate(frame.sort_values("case_id").iterrows(), start=1):
        payload = build_label_payload(row, panel="Panel A", feasibility_statuses=feasibility[str(row["case_id"])])
        jobs.append({
            "job_id": f"{scope}_gemma_single_{index:04d}",
            "provider": "gemma",
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "batch_index": index,
            "case_ids": [payload["case_id"]],
            "payload": [payload],
            "prompt": build_prompt([payload]),
            "schema_version": SCHEMA_VERSION,
        })
    return jobs


def _write(path: Path, jobs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-a", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    parser.add_argument("--panel-b", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_b.parquet")
    parser.add_argument("--feasibility", type=Path, default=ROOT / "artifacts/recommendation/feasibility/oulad_action_feasibility.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/recommendation/labeling/jobs")
    parser.add_argument("--pilot-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gemma-model", default="gemma-4-31b-it")
    args = parser.parse_args()
    panel_a = pd.read_parquet(args.panel_a)
    panel_b = pd.read_parquet(args.panel_b)
    if len(panel_a) != 500 or panel_a["case_id"].duplicated().any():
        raise ValueError("Panel A must contain exactly 500 unique cases")
    if set(panel_a["case_id"]) & set(panel_b["case_id"]):
        raise ValueError("Panel A and Panel B overlap")
    feasibility = _load_feasibility(args.feasibility)
    pilot = choose_pilot(panel_a, args.pilot_size, args.seed)
    pilot_jobs = _build_jobs(pilot, feasibility, args.gemma_model, "pilot")
    panel_jobs = _build_jobs(panel_a, feasibility, args.gemma_model, "panel_a")
    _write(args.output_dir / "pilot_gemma_single_jobs.jsonl", pilot_jobs)
    _write(args.output_dir / "panel_a_gemma_single_jobs.jsonl", panel_jobs)
    print(json.dumps({"pilot_jobs": len(pilot_jobs), "panel_a_jobs": len(panel_jobs), "cases_per_job": 1}, indent=2))


if __name__ == "__main__":
    main()
