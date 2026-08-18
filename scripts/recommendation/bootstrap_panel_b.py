"""Case-level bootstrap for frozen Panel B automated-reference metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.evaluation.metrics import mean_optional  # noqa: E402
from src.recommendation.evaluation.ranking_diagnostic import aggregate_case_metrics  # noqa: E402


def bootstrap_case_metrics(case_frame: pd.DataFrame, *, iterations: int = 2000, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    case_ids = case_frame["case_id"].astype(str).to_numpy()
    by_case = {str(row["case_id"]): dict(row) for _, row in case_frame.iterrows()}
    rows = []
    for iteration in range(iterations):
        sample = rng.choice(case_ids, size=len(case_ids), replace=True)
        summary = aggregate_case_metrics([by_case[case_id] for case_id in sample])
        summary["iteration"] = iteration
        rows.append(summary)
    return pd.DataFrame(rows)


def percentile_ci(series: pd.Series) -> dict:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"mean": None, "low": None, "high": None}
    return {"mean": float(clean.mean()), "low": float(np.percentile(clean, 2.5)), "high": float(np.percentile(clean, 97.5))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-metrics", type=Path, default=ROOT / "artifacts/recommendation/evaluation/panel_b_case_metrics.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/recommendation/evaluation/panel_b_bootstrap.parquet")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not args.case_metrics.exists():
        print(json.dumps({"status": "BLOCKED_PANEL_B_REFERENCE_API"}))
        return 2
    frame = pd.read_parquet(args.case_metrics)
    boot = bootstrap_case_metrics(frame, iterations=args.iterations, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    boot.to_parquet(args.output, index=False)
    print(json.dumps({"ndcg@3": percentile_ci(boot["ndcg@3"]), "iterations": args.iterations}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
