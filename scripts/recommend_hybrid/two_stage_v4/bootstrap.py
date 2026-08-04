"""Learner-cluster bootstrap for action-aware V4 held-out predictions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/two_stage_v4"
PROTOCOL = yaml.safe_load(
    (ROOT / "configs/recommend_hybrid/two_stage_v4_protocol.yaml").read_text(
        encoding="utf-8"
    )
)
REPLICATES = int(PROTOCOL["evaluation"]["bootstrap"]["replicates"])
SEED = int(PROTOCOL["evaluation"]["bootstrap"]["seed"])


def _summary(values: np.ndarray, point: float) -> dict[str, float]:
    return {
        "point_estimate": float(point),
        "bootstrap_mean": float(values.mean()),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def main() -> None:
    path = OUT / "final_oof/OOF_PREDICTIONS.parquet"
    frame = pd.read_parquet(path)
    required = {
        "base_record_id",
        "issued",
        "correct_top1",
        "group_has_positive",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"V4 OOF predictions missing columns: {sorted(missing)}")
    sufficient = frame.assign(
        issued_count=(frame["issued"] == 1).astype(int),
        correct_count=(frame["correct_top1"] == 1).astype(int),
        positive_count=(frame["group_has_positive"] == 1).astype(int),
        issued_positive=(
            (frame["issued"] == 1) & (frame["group_has_positive"] == 1)
        ).astype(int),
    ).groupby("base_record_id", sort=True)[
        ["issued_count", "correct_count", "positive_count", "issued_positive"]
    ].sum()
    learners = sufficient.index.to_numpy(dtype=object)
    stats = sufficient.to_numpy(dtype=np.int64)
    if len(learners) < 2:
        raise RuntimeError("V4 learner bootstrap requires at least two clusters")
    point = stats.sum(axis=0)
    point_end_to_end = float(point[1] / point[0]) if point[0] else 0.0
    point_coverage = float(point[3] / point[2]) if point[2] else 0.0
    point_gate_precision = float(point[3] / point[0]) if point[0] else 0.0
    point_conditional = float(point[1] / point[3]) if point[3] else 0.0

    rng = np.random.default_rng(SEED)
    end_to_end = np.zeros(REPLICATES, dtype=np.float64)
    coverage = np.zeros(REPLICATES, dtype=np.float64)
    gate_precision = np.zeros(REPLICATES, dtype=np.float64)
    conditional = np.zeros(REPLICATES, dtype=np.float64)
    for replicate in range(REPLICATES):
        sampled = rng.integers(0, len(learners), size=len(learners))
        totals = stats[sampled].sum(axis=0)
        end_to_end[replicate] = totals[1] / totals[0] if totals[0] else 0.0
        coverage[replicate] = totals[3] / totals[2] if totals[2] else 0.0
        gate_precision[replicate] = totals[3] / totals[0] if totals[0] else 0.0
        conditional[replicate] = totals[1] / totals[3] if totals[3] else 0.0
    result = {
        "schema_version": "two_stage_v4_learner_bootstrap_v1",
        "status": "COMPLETE",
        "cluster": "base_record_id",
        "replicates": REPLICATES,
        "seed": SEED,
        "learner_count": int(len(learners)),
        "group_count": int(len(frame)),
        "end_to_end_precision_at_1": _summary(end_to_end, point_end_to_end),
        "positive_group_coverage": _summary(coverage, point_coverage),
        "stage_a_precision": _summary(gate_precision, point_gate_precision),
        "stage_a_recall": _summary(coverage, point_coverage),
        "stage_b_conditional_precision_at_1": _summary(
            conditional,
            point_conditional,
        ),
        "claim_boundary": PROTOCOL["claim_boundary"],
    }
    output = OUT / "final_oof/BOOTSTRAP.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
