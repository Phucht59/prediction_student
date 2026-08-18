"""Offline pilot gate comparing Gemma candidate labels to existing Gemini B2."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.labeling.academic_help_seeking import parse_academic_help_function_call  # noqa: E402
from src.recommendation.labeling.runtime import load_jsonl  # noqa: E402


def _distribution(series: pd.Series) -> dict[str, int]:
    return {label: int(series.eq(label).sum()) for label in ("0", "1", "2", "3", "ABSTAIN")}


def _kappa(left: pd.Series, right: pd.Series) -> float | None:
    mask = ~left.eq("ABSTAIN") & ~right.eq("ABSTAIN")
    if not mask.any():
        return None
    try:
        from sklearn.metrics import cohen_kappa_score
        value = float(cohen_kappa_score(pd.to_numeric(left[mask]), pd.to_numeric(right[mask]), labels=[0, 1, 2, 3], weights="quadratic"))
        return None if math.isnan(value) else value
    except Exception:
        return None


def _load_gemma(raw_path: Path, jobs_path: Path) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError("PILOT_NOT_AVAILABLE")
    jobs = load_jsonl(jobs_path)
    records = load_jsonl(raw_path)
    rows = []
    for job in jobs:
        matching = [record for record in records if str(record.get("job_id")) == str(job["job_id"])]
        if len(matching) != 1 or matching[0].get("status") != "completed":
            raise ValueError(f"pilot job not completed: {job['job_id']}")
        case_id = str(job["case_ids"][0])
        parsed = parse_academic_help_function_call(matching[0]["raw_response"], [case_id])
        rows.append({"case_id": case_id, "label_gemma": str(parsed[case_id]["labels"]["A4"]["label"])})
    return pd.DataFrame(rows)


def evaluate(gemma: pd.DataFrame, gemini_path: Path, pilot_jobs_path: Path, config_path: Path) -> dict:
    jobs = load_jsonl(pilot_jobs_path)
    pilot_ids = {str(job["case_ids"][0]) for job in jobs}
    gemini = pd.read_parquet(gemini_path)[["case_id", "action_id", "label"]].copy()
    gemini = gemini[(gemini["action_id"] == "B2_ACADEMIC_HELP_SEEKING") & gemini["case_id"].astype(str).isin(pilot_ids)]
    gemini = gemini.rename(columns={"label": "label_gemini"})
    gemini["case_id"] = gemini["case_id"].astype(str)
    merged = gemma.merge(gemini, on="case_id", how="inner", validate="one_to_one")
    if len(merged) != len(pilot_ids):
        raise ValueError("Gemma/Gemini pilot case overlap is incomplete")
    exact = merged["label_gemma"].eq(merged["label_gemini"])
    distribution_gemma = _distribution(merged["label_gemma"])
    distribution_gemini = _distribution(merged["label_gemini"])
    max_share = max(distribution_gemma.values()) / len(merged)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gate = config["gate"]
    kappa = _kappa(merged["label_gemma"], merged["label_gemini"])
    non_degenerate = max_share < float(gate["degeneracy_max_share"])
    agreement_ok = bool(exact.mean() >= float(gate["minimum_exact_agreement"]) or (kappa is not None and kappa >= float(gate["minimum_weighted_kappa"])))
    status = "ALLOW_FULL_RUN" if non_degenerate and agreement_ok else "STOP_A4_REPLACEMENT"
    return {"status": status, "n": len(merged), "exact_agreement": float(exact.mean()), "weighted_kappa": kappa,
            "gemma_abstain_rate": float(merged["label_gemma"].eq("ABSTAIN").mean()),
            "gemini_abstain_rate": float(merged["label_gemini"].eq("ABSTAIN").mean()),
            "gemma_distribution": distribution_gemma, "gemini_distribution": distribution_gemini,
            "non_degenerate": non_degenerate, "agreement_ok": agreement_ok, "max_gemma_class_share": max_share}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "artifacts/recommendation/labeling/raw/academic_help_seeking_gemma_pilot.jsonl")
    parser.add_argument("--jobs", type=Path, default=ROOT / "artifacts/recommendation/labeling/jobs/academic_help_seeking_gemma_pilot_jobs.jsonl")
    parser.add_argument("--gemini", type=Path, default=ROOT / "artifacts/recommendation/labeling/normalized/a4_replacement_gemini_labels.parquet")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/recommendation/academic_help_seeking.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/recommendation/A4_ACADEMIC_HELP_SEEKING_PILOT.md")
    args = parser.parse_args()
    try:
        gemma = _load_gemma(args.raw, args.jobs)
    except FileNotFoundError:
        args.output.write_text("# A4 Academic Help-Seeking pilot\n\n`PILOT_NOT_AVAILABLE`\n", encoding="utf-8")
        print("PILOT_NOT_AVAILABLE")
        return 2
    result = evaluate(gemma, args.gemini, args.jobs, args.config)
    lines = ["# A4 Academic Help-Seeking pilot", "", "Comparison uses existing Gemini B2 labels only; no Gemini API call is made.", "", f"- Gate: `{result['status']}`", f"- Exact agreement: `{result['exact_agreement']:.4f}`", f"- Weighted kappa: `{result['weighted_kappa'] if result['weighted_kappa'] is not None else 'UNAVAILABLE'}`", f"- Gemma ABSTAIN: `{result['gemma_abstain_rate']:.4f}`; distribution `{result['gemma_distribution']}`", f"- Gemini ABSTAIN: `{result['gemini_abstain_rate']:.4f}`; distribution `{result['gemini_distribution']}`", f"- Gemma non-degenerate: `{result['non_degenerate']}`; agreement gate: `{result['agreement_ok']}`"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ALLOW_FULL_RUN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
