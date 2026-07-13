"""CLI for paired comparison of protocol-V2 outer-fold prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

def _score(frame: pd.DataFrame, metric: str) -> float:
    truth, predicted = frame.true_label, frame.predicted_label
    if metric == "macro_f1":
        return float(f1_score(truth, predicted, average="macro", zero_division=0))
    if metric == "accuracy":
        return float(accuracy_score(truth, predicted))
    if metric == "weighted_f1":
        return float(f1_score(truth, predicted, average="weighted", zero_division=0))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(truth, predicted))
    if metric == "quadratic_weighted_kappa":
        return float(cohen_kappa_score(truth, predicted, weights="quadratic"))
    if metric == "ordinal_mae":
        return float(np.mean(np.abs(truth.to_numpy() - predicted.to_numpy())))
    raise ValueError(f"Unsupported comparison metric: {metric}")


def compare_predictions(run_a: pd.DataFrame, run_b: pd.DataFrame, *, metric: str, n_bootstrap: int = 2000, seed: int = 42) -> dict:
    required = {"record_id", "outer_fold", "true_label", "predicted_label"}
    if required - set(run_a) or required - set(run_b):
        raise ValueError("Both prediction artifacts need record_id, outer_fold, true_label, predicted_label.")
    merged = run_a.merge(run_b, on=["record_id", "outer_fold"], suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) != len(run_a) or len(merged) != len(run_b) or not (merged.true_label_a == merged.true_label_b).all():
        raise ValueError("Runs must predict the same outer-validation records with identical labels.")
    fold_differences = []
    for fold, group in merged.groupby("outer_fold", sort=True):
        a = group.rename(columns={"true_label_a": "true_label", "predicted_label_a": "predicted_label"})
        b = group.rename(columns={"true_label_b": "true_label", "predicted_label_b": "predicted_label"})
        fold_differences.append({"outer_fold": int(fold), "run_a": _score(a, metric), "run_b": _score(b, metric)})
    differences = np.asarray([row["run_a"] - row["run_b"] for row in fold_differences], dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        sampled = merged.iloc[rng.integers(0, len(merged), len(merged))]
        a = sampled.rename(columns={"true_label_a": "true_label", "predicted_label_a": "predicted_label"})
        b = sampled.rename(columns={"true_label_b": "true_label", "predicted_label_b": "predicted_label"})
        draws.append(_score(a, metric) - _score(b, metric))
    truth = merged.true_label_a.to_numpy()
    return {
        "metric": metric,
        "n_records": int(len(merged)),
        "fold_wise": fold_differences,
        "mean_difference": float(differences.mean()),
        "standard_deviation": float(differences.std(ddof=1)) if len(differences) > 1 else 0.0,
        "run_a_fold_wins": int(np.sum(differences > 0)),
        "run_b_fold_wins": int(np.sum(differences < 0)),
        "ties": int(np.sum(differences == 0)),
        "paired_bootstrap_95_ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "classwise_correctness_difference": {str(label): float(((merged.loc[truth == label, "predicted_label_a"] == label).mean()) - ((merged.loc[truth == label, "predicted_label_b"] == label).mean())) for label in range(3)},
        "one_step_error_difference": int(np.sum(np.abs(merged.true_label_a - merged.predicted_label_a) == 1) - np.sum(np.abs(merged.true_label_b - merged.predicted_label_b) == 1)),
        "two_step_error_difference": int(np.sum(np.abs(merged.true_label_a - merged.predicted_label_a) >= 2) - np.sum(np.abs(merged.true_label_b - merged.predicted_label_b) >= 2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True, help="Record-level CSV prediction artifact for model A.")
    parser.add_argument("--run-b", type=Path, required=True, help="Record-level CSV prediction artifact for model B.")
    parser.add_argument("--metric", default="macro_f1", choices=["macro_f1", "accuracy", "weighted_f1", "balanced_accuracy", "quadratic_weighted_kappa", "ordinal_mae"])
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(compare_predictions(pd.read_csv(args.run_a), pd.read_csv(args.run_b), metric=args.metric, n_bootstrap=args.n_bootstrap), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
