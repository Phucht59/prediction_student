"""Learner-cluster bootstrap for hybrid-only OOF recommendation evidence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/hybrid_only_final"
PROTOCOL = yaml.safe_load(
    (ROOT / "configs/recommend_hybrid/hybrid_only_final_protocol.yaml").read_text(
        encoding="utf-8"
    )
)
SEED = int(PROTOCOL["evaluation"]["bootstrap"]["seed"])
REPLICATES = int(PROTOCOL["evaluation"]["bootstrap"]["replicates"])


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metric(frame: pd.DataFrame) -> tuple[float, float]:
    issued = frame[frame["issued"] == 1]
    precision = float(issued["silver_positive"].mean()) if len(issued) else 0.0
    positive = int(frame["group_has_positive"].sum())
    covered = int(
        ((frame["issued"] == 1) & (frame["group_has_positive"] == 1)).sum()
    )
    coverage = float(covered / positive) if positive else 0.0
    return precision, coverage


def main() -> None:
    path = OUT / "evaluation/OOF_PREDICTIONS.parquet"
    frame = pd.read_parquet(path)
    required = {
        "base_record_id",
        "issued",
        "silver_positive",
        "group_has_positive",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"OOF predictions missing bootstrap columns: {sorted(missing)}")

    learner_rows = {
        str(learner): group.copy()
        for learner, group in frame.groupby("base_record_id", sort=False)
    }
    learners = np.asarray(sorted(learner_rows), dtype=object)
    if len(learners) < 2:
        raise RuntimeError("learner-cluster bootstrap requires at least two learners")

    point_precision, point_coverage = _metric(frame)
    rng = np.random.default_rng(SEED)
    precision_values = np.empty(REPLICATES, dtype=np.float64)
    coverage_values = np.empty(REPLICATES, dtype=np.float64)

    # Aggregate sufficient statistics by learner so each replicate is cheap and
    # all learner-stage groups remain clustered together.
    sufficient = frame.assign(
        issued_correct=(
            (frame["issued"] == 1) & (frame["silver_positive"] == 1)
        ).astype(int),
        issued_count=(frame["issued"] == 1).astype(int),
        positive_count=(frame["group_has_positive"] == 1).astype(int),
        covered_positive=(
            (frame["issued"] == 1) & (frame["group_has_positive"] == 1)
        ).astype(int),
    ).groupby("base_record_id", sort=False)[
        ["issued_correct", "issued_count", "positive_count", "covered_positive"]
    ].sum()
    stats = sufficient.reindex(learners).to_numpy(dtype=np.int64)

    for replicate in range(REPLICATES):
        sampled = rng.integers(0, len(learners), size=len(learners))
        totals = stats[sampled].sum(axis=0)
        precision_values[replicate] = (
            float(totals[0] / totals[1]) if totals[1] else 0.0
        )
        coverage_values[replicate] = (
            float(totals[3] / totals[2]) if totals[2] else 0.0
        )

    result = {
        "status": "COMPLETE",
        "cluster": "base_record_id",
        "replicates": REPLICATES,
        "seed": SEED,
        "learner_count": int(len(learners)),
        "group_count": int(len(frame)),
        "precision_at_1": {
            "point_estimate": point_precision,
            "lower_95": float(np.quantile(precision_values, 0.025)),
            "upper_95": float(np.quantile(precision_values, 0.975)),
            "bootstrap_mean": float(np.mean(precision_values)),
        },
        "actionable_coverage": {
            "point_estimate": point_coverage,
            "lower_95": float(np.quantile(coverage_values, 0.025)),
            "upper_95": float(np.quantile(coverage_values, 0.975)),
            "bootstrap_mean": float(np.mean(coverage_values)),
        },
    }
    _write(OUT / "evaluation/BOOTSTRAP.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
