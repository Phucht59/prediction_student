"""Prepare API-free Gemini repeatability and prompt-robustness jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.recommendation.build_labeling_jobs import _load_feasibility, choose_sample  # noqa: E402
from src.recommendation.labeling.constants import PROMPT_VERSION, PROMPT_VERSION_B, SCHEMA_VERSION  # noqa: E402
from src.recommendation.labeling.payload import build_label_payload  # noqa: E402
from src.recommendation.labeling.prompt import build_prompt, build_prompt_v1b  # noqa: E402


def _build_jobs(frame: pd.DataFrame, feasibility: dict[str, dict[str, str]], *, prompt_version: str,
                prompt_builder, experiment: str, model: str, batch_size: int) -> list[dict]:
    payloads = [build_label_payload(row, panel="Panel A", feasibility_statuses=feasibility[str(row["case_id"])])
                for _, row in frame.sort_values("case_id").iterrows()]
    jobs = []
    for batch_index, start in enumerate(range(0, len(payloads), batch_size), start=1):
        batch = payloads[start:start + batch_size]
        jobs.append({
            "job_id": f"{experiment}_batch_{batch_index:04d}",
            "experiment": experiment,
            "provider": "gemini",
            "model": model,
            "prompt_version": prompt_version,
            "batch_index": batch_index,
            "case_ids": [payload["case_id"] for payload in batch],
            "payload": batch,
            "prompt": prompt_builder(batch),
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
    parser.add_argument("--size", type=int, default=150)
    parser.add_argument("--batch-size", type=int, choices=(1, 5, 10), default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gemini-model", default="gemini-3.5-flash-lite")
    args = parser.parse_args()
    if args.size <= 0 or args.size > 500:
        raise ValueError("robustness sample size must be between 1 and 500")
    panel_a = pd.read_parquet(args.panel_a)
    panel_b = pd.read_parquet(args.panel_b)
    if set(panel_a["case_id"]) & set(panel_b["case_id"]):
        raise ValueError("Panel A and Panel B overlap")
    if len(panel_a) != 500 or panel_a["case_id"].duplicated().any():
        raise ValueError("Panel A must contain exactly 500 unique cases")
    feasibility = _load_feasibility(args.feasibility)
    sample = choose_sample(panel_a, args.size, args.seed)
    if len(sample) != args.size or set(sample["case_id"]) & set(panel_b["case_id"]):
        raise ValueError("robustness sample is invalid")
    if set(sample["stage"]) != {"20pct", "35pct", "50pct", "75pct"}:
        raise ValueError("robustness sample does not cover all recommendation stages")
    if set(sample["outer_fold"]) != {0, 1, 2}:
        raise ValueError("robustness sample does not cover all outer folds")
    if set(sample["sampling_risk_band"]) != {"Low", "Borderline", "High"}:
        raise ValueError("robustness sample does not cover all sampling risk bands")
    repeat_jobs = _build_jobs(sample, feasibility, prompt_version=PROMPT_VERSION, prompt_builder=build_prompt,
                              experiment="gemini_repeat_v1", model=args.gemini_model, batch_size=args.batch_size)
    v1b_jobs = _build_jobs(sample, feasibility, prompt_version=PROMPT_VERSION_B, prompt_builder=build_prompt_v1b,
                           experiment="gemini_prompt_v1b", model=args.gemini_model, batch_size=args.batch_size)
    _write(args.output_dir / "gemini_repeat_v1_jobs.jsonl", repeat_jobs)
    _write(args.output_dir / "gemini_prompt_v1b_jobs.jsonl", v1b_jobs)
    report = ROOT / "reports/recommendation/PHASE5_ROBUSTNESS_PREP.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f"""# Phase 5 Gemini robustness preparation

- Main run remains `LF_GEMINI_MAIN`, with its existing jobs/raw artifacts untouched.
- Repeatability experiment: `gemini_repeat_v1`, {len(sample)} shared Panel A cases, {len(repeat_jobs)} jobs, seed `{args.seed}`, prompt `{PROMPT_VERSION}`.
- Prompt robustness experiment: `gemini_prompt_v1b`, {len(sample)} identical cases, {len(v1b_jobs)} jobs, prompt `{PROMPT_VERSION_B}`.
- Model: `{args.gemini_model}`; batch size: `{args.batch_size}`.
- Sample coverage: stages 20/35/50/75, folds 0/1/2, risk bands Low/Borderline/High.
- Panel B cases: excluded. API calls made: none.

The comparison script reports LLM self-consistency for main versus repeat and prompt robustness
for main versus v1b. Neither experiment is an independent labeling function and neither is sent
to Snorkel. A4 diagnostics report numeric, ABSTAIN, and available abstain-reason rates; it flags
`A4 lacks observable evidence in current Student State` when all runs are nearly all ABSTAIN.
""", encoding="utf-8")
    print(json.dumps({"cases": len(sample), "repeat_jobs": len(repeat_jobs), "v1b_jobs": len(v1b_jobs)}, indent=2))


if __name__ == "__main__":
    main()
