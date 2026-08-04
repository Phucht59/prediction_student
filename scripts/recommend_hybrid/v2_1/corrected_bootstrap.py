"""Corrected learner-cluster bootstrap for V2.1 ranking metrics.

Two estimands are reported separately:
- group weighted: every learner-stage ranking group has equal weight;
- learner weighted: every learner has equal weight after averaging their groups.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from scientific_core import group_metric_rows

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/outcome_grounded_v2_1"
FINAL = OUT / "final_oof"
CONFIG = ROOT / "configs/recommend_hybrid/outcome_grounded_v2_1.yaml"
SEED = 20260804


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def update_progress(status: str, **details: Any) -> None:
    path = OUT / "PROGRESS.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("stages", {})["CORRECTED_BOOTSTRAP"] = {"status": status, **details}
    atomic_json(path, payload)


def metric_table(predictions: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    merged = None
    identity = ["group_id", "base_record_id", "stage", "outer_fold"]
    for method in methods:
        rows = group_metric_rows(predictions, method)[identity + ["ndcg_at_3"]]
        rows = rows.rename(columns={"ndcg_at_3": method})
        merged = rows if merged is None else merged.merge(rows, on=identity, how="inner", validate="one_to_one")
    if merged is None or merged.empty:
        raise RuntimeError("No group-level metrics available for bootstrap")
    return merged


def bootstrap_comparison(
    table: pd.DataFrame,
    baseline: str,
    replicates: int,
    seed: int,
    batch_size: int = 100,
) -> tuple[dict[str, Any], dict[str, Any]]:
    table = table.copy()
    table["difference"] = table["model_score"] - table[baseline]
    learner = (
        table.groupby("base_record_id", sort=True)
        .agg(group_sum=("difference", "sum"), group_count=("difference", "size"), learner_mean=("difference", "mean"))
        .reset_index()
    )
    sums = learner["group_sum"].to_numpy(dtype=float)
    counts = learner["group_count"].to_numpy(dtype=float)
    learner_means = learner["learner_mean"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    group_values = np.empty(replicates, dtype=float)
    learner_values = np.empty(replicates, dtype=float)

    cursor = 0
    while cursor < replicates:
        current = min(batch_size, replicates - cursor)
        sampled = rng.integers(0, len(learner), size=(current, len(learner)))
        sampled_sums = sums[sampled].sum(axis=1)
        sampled_counts = counts[sampled].sum(axis=1)
        group_values[cursor : cursor + current] = sampled_sums / sampled_counts
        learner_values[cursor : cursor + current] = learner_means[sampled].mean(axis=1)
        cursor += current

    group_point = float(table["difference"].mean())
    learner_point = float(learner_means.mean())
    common = {
        "baseline": baseline,
        "replicates": replicates,
        "learners": int(len(learner)),
        "groups": int(len(table)),
        "cluster": "base_record_id",
    }
    group_result = {
        **common,
        "estimand": "group_weighted_equal_learner_stage_groups",
        "estimate": group_point,
        "ci95_low": float(np.quantile(group_values, 0.025)),
        "ci95_high": float(np.quantile(group_values, 0.975)),
        "probability_difference_le_zero": float(np.mean(group_values <= 0)),
    }
    learner_result = {
        **common,
        "estimand": "learner_weighted_equal_learners",
        "estimate": learner_point,
        "ci95_low": float(np.quantile(learner_values, 0.025)),
        "ci95_high": float(np.quantile(learner_values, 0.975)),
        "probability_difference_le_zero": float(np.mean(learner_values <= 0)),
    }
    return group_result, learner_result


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    predictions = pd.read_parquet(FINAL / "OOF_RANKING_PREDICTIONS.parquet")
    methods = [
        "model_score",
        "popular_score",
        "workload_score",
        "policy_score",
        "counterfactual_score",
    ]
    table = metric_table(predictions, methods)
    group_results = []
    learner_results = []
    for offset, baseline in enumerate(methods[1:]):
        group_result, learner_result = bootstrap_comparison(
            table,
            baseline,
            replicates,
            SEED + offset * 100,
        )
        group_results.append(group_result)
        learner_results.append(learner_result)

    atomic_json(
        FINAL / "BOOTSTRAP_GROUP_WEIGHTED.json",
        {"status": "COMPLETE", "comparisons": group_results},
    )
    atomic_json(
        FINAL / "BOOTSTRAP_LEARNER_WEIGHTED.json",
        {"status": "COMPLETE", "comparisons": learner_results},
    )
    update_progress("COMPLETE", replicates=replicates)
    print(json.dumps({"group_weighted": group_results, "learner_weighted": learner_results}, indent=2))


if __name__ == "__main__":
    main()
