"""Create reproducible, API-free Gemma/Gemini labeling jobs from Panel A."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.labeling.constants import PROMPT_VERSION, SCHEMA_VERSION  # noqa: E402
from src.recommendation.labeling.payload import build_label_payload  # noqa: E402
from src.recommendation.labeling.prompt import build_prompt  # noqa: E402


def _hash_key(case_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{case_id}".encode()).hexdigest()


def choose_pilot(panel: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    ordered = panel.assign(_order=panel["case_id"].map(lambda value: _hash_key(str(value), seed))).sort_values("_order")
    chosen: list[str] = []
    unseen = {
        "stage": {"20pct", "35pct", "50pct", "75pct"},
        "outer_fold": {0, 1, 2},
        "sampling_risk_band": {"Low", "Borderline", "High"},
    }
    remaining = ordered.copy()
    while any(unseen.values()) and len(chosen) < size:
        scored = []
        for index, row in remaining.iterrows():
            score = sum(row[column] in values for column, values in unseen.items())
            scored.append((score, str(row["_order"]), index))
        _, _, index = max(scored)
        row = remaining.loc[index]
        chosen.append(str(row["case_id"]))
        for column in unseen:
            unseen[column].discard(row[column])
        remaining = remaining.drop(index)
    if any(unseen.values()):
        raise ValueError(f"pilot cannot cover marginal strata: {unseen}")
    chosen_set = set(chosen)
    for case_id in ordered[~ordered["case_id"].astype(str).isin(chosen_set)]["case_id"].astype(str):
        if len(chosen) >= size:
            break
        chosen.append(case_id)
    if len(chosen) != size:
        raise ValueError(f"requested pilot size {size}, only selected {len(chosen)}")
    return panel[panel["case_id"].astype(str).isin(chosen)].copy().sort_values("case_id")


def choose_sample(panel: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Deterministic marginal coverage sampler for later robustness experiments."""
    return choose_pilot(panel, size, seed)


def _load_feasibility(path: Path) -> dict[str, dict[str, str]]:
    frame = pd.read_parquet(path)
    grouped = frame.groupby("case_id", sort=False)
    result = {}
    for case_id, rows in grouped:
        mapping = dict(zip(rows["action_id"].astype(str), rows["feasibility_status"].astype(str)))
        if set(mapping) != {"A1", "A2", "A3", "A4", "A5"}:
            raise ValueError(f"feasibility must have exactly five actions for {case_id}")
        result[str(case_id)] = mapping
    return result


def _jobs(frame: pd.DataFrame, feasibility: dict[str, dict[str, str]], provider: str, model: str, scope: str, batch_size: int) -> list[dict]:
    payloads = [build_label_payload(row, panel="Panel A", feasibility_statuses=feasibility[str(row["case_id"])])
                for _, row in frame.sort_values("case_id").iterrows()]
    jobs = []
    for batch_index, start in enumerate(range(0, len(payloads), batch_size), start=1):
        batch = payloads[start:start + batch_size]
        jobs.append({
            "job_id": f"{scope.lower().replace(' ', '_')}_{provider}_batch_{batch_index:04d}",
            "provider": provider,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "batch_index": batch_index,
            "case_ids": [payload["case_id"] for payload in batch],
            "payload": batch,
            "prompt": build_prompt(batch),
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
    parser.add_argument("--batch-size", type=int, choices=(1, 5, 10), default=10)
    parser.add_argument("--pilot-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gemma-model", default="gemma-4-31b-it")
    parser.add_argument("--gemini-model", default="gemini-3.5-flash-lite")
    args = parser.parse_args()
    if args.pilot_size <= 0 or args.pilot_size > 500:
        raise ValueError("pilot size must be between 1 and 500")
    panel_a = pd.read_parquet(args.panel_a)
    panel_b = pd.read_parquet(args.panel_b)
    if len(panel_a) != 500 or panel_a["case_id"].duplicated().any():
        raise ValueError("Panel A must contain exactly 500 unique cases")
    if set(panel_a["case_id"]) & set(panel_b["case_id"]):
        raise ValueError("Panel A and Panel B overlap")
    if set(panel_a["stage"]) - {"20pct", "35pct", "50pct", "75pct"}:
        raise ValueError("Panel A contains an invalid recommendation stage")
    feasibility = _load_feasibility(args.feasibility)
    if set(panel_a["case_id"].astype(str)) - set(feasibility):
        raise ValueError("missing feasibility rows for Panel A")
    pilot = choose_pilot(panel_a, args.pilot_size, args.seed)
    specs = (("gemma", args.gemma_model), ("gemini", args.gemini_model))
    counts = {}
    for provider, model in specs:
        full = _jobs(panel_a, feasibility, provider, model, "panel_a", args.batch_size)
        pilot_jobs = _jobs(pilot, feasibility, provider, model, "pilot", args.batch_size)
        _write(args.output_dir / f"panel_a_{provider}_jobs.jsonl", full)
        _write(args.output_dir / f"pilot_{provider}_jobs.jsonl", pilot_jobs)
        counts[provider] = {"panel_a_jobs": len(full), "pilot_jobs": len(pilot_jobs)}
    report = ROOT / "reports/recommendation/PHASE5_LABELING_INFRASTRUCTURE.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f"""# Phase 5 labeling infrastructure

- Prompt/rubric: `{PROMPT_VERSION}`; one provider-neutral rubric for Gemma and Gemini.
- Action taxonomy: A1 Assessment Recovery, A2 Re-engagement, A3 Study Planning, A4 Content Review, A5 Retrieval Practice.
- Label domain: `0/1/2/3/ABSTAIN`; infeasible action is `ABSTAIN` with reason `INFEASIBLE`, never numeric zero.
- Panel A coverage: {len(panel_a)} cases; Panel B cases are excluded from all jobs.
- Pilot coverage: {len(pilot)} cases; pilot is deterministically selected with seed `{args.seed}` across stages, folds, and risk bands.
- Default batch size: `{args.batch_size}`; supported sizes: 1, 5, 10.
- Models: Gemma `{args.gemma_model}`, Gemini `{args.gemini_model}`.
- Jobs: Gemma {counts['gemma']['panel_a_jobs']} full / {counts['gemma']['pilot_jobs']} pilot; Gemini {counts['gemini']['panel_a_jobs']} full / {counts['gemini']['pilot_jobs']} pilot.
- API calls made during generation: none. API keys are not stored in jobs.

Runners write raw response records only after the user runs them locally. Snorkel, label modeling,
Panel B labeling, recommendation prose, and human evaluation are out of scope for this phase.
""", encoding="utf-8")
    print(json.dumps({"panel_a_cases": len(panel_a), "pilot_cases": len(pilot), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
