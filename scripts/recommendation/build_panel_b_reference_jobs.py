"""Prepare Panel B automated-reference jobs. Does not call an API or invent labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.feasibility.rules_v2 import build_feasibility_frame_v2  # noqa: E402
from src.recommendation.labeling.constants import SCHEMA_VERSION  # noqa: E402
from src.recommendation.labeling.panel_b_reference import (  # noqa: E402
    PANEL_B_ACTIONS,
    PANEL_B_REFERENCE_PROMPT_VERSION,
    build_panel_b_payload,
    build_panel_b_prompt,
)


PROVIDERS = {
    "REF_GEMINI35": ("gemini", "gemini-3.5-flash-lite", "panel_b_reference_gemini35_jobs.jsonl"),
    "REF_GEMINI31": ("gemini", "gemini-3.1-flash-lite", "panel_b_reference_gemini31_jobs.jsonl"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-b", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_b.parquet")
    parser.add_argument("--panel-a", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_a.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/recommendation/labeling/jobs")
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    panel_b = pd.read_parquet(args.panel_b).copy()
    panel_a = pd.read_parquet(args.panel_a, columns=["case_id", "student_id", "enrollment_identity"])
    panel_b["case_id"] = panel_b["case_id"].astype(str)
    if len(panel_b) != 150 or panel_b["case_id"].duplicated().any():
        raise ValueError("Panel B must contain 150 unique cases")
    if set(panel_b["case_id"]) & set(panel_a["case_id"].astype(str)):
        raise ValueError("Panel B jobs would leak Panel A cases")
    if set(panel_b["student_id"].astype(str)) & set(panel_a["student_id"].astype(str)):
        raise ValueError("Panel B jobs would leak Panel A students")
    feas = build_feasibility_frame_v2(panel_b)
    feas_map = {str(case_id): dict(zip(group["action_id"].astype(str), group["feasibility_status"].astype(str))) for case_id, group in feas.groupby("case_id")}
    payloads = [build_panel_b_payload(row, feas_map[str(row["case_id"])]) for _, row in panel_b.sort_values("case_id").iterrows()]
    if any("raw_score" in json.dumps(payload) or "relevance_score" in json.dumps(payload) for payload in payloads):
        raise ValueError("reference jobs must not contain model predictions")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for source, (provider, model, filename) in PROVIDERS.items():
        jobs = []
        for index, start in enumerate(range(0, len(payloads), args.batch_size), start=1):
            batch = payloads[start:start + args.batch_size]
            jobs.append({
                "job_id": f"panel_b_reference_{source.lower()}_batch_{index:04d}",
                "provider": provider,
                "model": model,
                "prompt_version": PANEL_B_REFERENCE_PROMPT_VERSION,
                "batch_index": index,
                "case_ids": [item["case_id"] for item in batch],
                "payload": batch,
                "prompt": build_panel_b_prompt(batch),
                "schema_version": SCHEMA_VERSION,
                "actions": list(PANEL_B_ACTIONS),
                "reference_source": source,
            })
        path = args.output_dir / filename
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for job in jobs:
                handle.write(json.dumps(job, ensure_ascii=False, sort_keys=True) + "\n")
        counts[source] = len(jobs)
    print(json.dumps({"cases": 150, "jobs": counts, "api_calls": 0}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
