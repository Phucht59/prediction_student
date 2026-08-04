"""Run all V2.1 ablations with the exact selected outer-fold models.

The initial ablation runner reduced selected LambdaMART models from 100 trees to
10 trees.  This wrapper preserves any partial reduced-budget outputs, installs an
exact-hyperparameter evaluator, and delegates to the resumable ablation ledger.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

import corrected_ablation as ablation
from scientific_core import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FeaturePreprocessor,
    RelevanceTransformer,
    aggregate_metrics,
    fit_ranker,
    predict_ranker,
)

OUT = ablation.OUT
ABLATION_OUT = ablation.ABLATION_OUT
ARCHIVE = OUT / "ablations_reduced_budget_archive"
MARKER = OUT / "EXACT_ABLATION_EXECUTION.json"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def archive_reduced_budget_ablations_once() -> None:
    if MARKER.exists():
        return
    if ABLATION_OUT.exists():
        if ARCHIVE.exists():
            raise RuntimeError(f"Ablation archive already exists: {ARCHIVE}")
        shutil.move(str(ABLATION_OUT), str(ARCHIVE))
    atomic_json(
        MARKER,
        {
            "status": "RUNNING",
            "selected_hyperparameters_required": True,
            "reduced_tree_budget_allowed": False,
            "archived_reduced_budget_outputs": ARCHIVE.exists(),
        },
    )


def exact_evaluate_ablation(
    eligible_raw: pd.DataFrame,
    all_raw: pd.DataFrame,
    name: str,
    spec: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    source = all_raw if spec.get("all_candidates") else eligible_raw

    for outer_fold in [0, 1, 2]:
        raw_train = ablation.ordered(source[source["outer_fold"] != outer_fold].copy())
        raw_test = ablation.ordered(source[source["outer_fold"] == outer_fold].copy())
        relevance = RelevanceTransformer(seed=ablation.SEED + outer_fold)
        train = relevance.fit_transform(raw_train)
        test = relevance.transform(raw_test)

        if spec.get("prior_only"):
            prior = train.groupby("action_family", observed=True)["continuous_relevance"].mean()
            test["ablation_score"] = test["action_family"].map(prior).fillna(0.0)
        else:
            numeric = [
                column
                for column in NUMERIC_FEATURES
                if column not in spec.get("remove", [])
            ]
            preprocessor = FeaturePreprocessor(
                numeric_features=numeric,
                categorical_features=CATEGORICAL_FEATURES,
                include_interactions=bool(spec.get("interactions", True)),
            )
            train_matrix = preprocessor.fit_transform(train)
            test_matrix = preprocessor.transform(test)
            family, parameters = ablation.selected_model(outer_fold)
            ranker = fit_ranker(
                family,
                train_matrix,
                train,
                dict(parameters),
                ablation.SEED + outer_fold,
            )
            test["ablation_score"] = predict_ranker(ranker, test_matrix)

        metrics = aggregate_metrics(test, "ablation_score")
        metrics["unavailable_top_action_rate"] = ablation.unavailable_top_rate(
            test, "ablation_score"
        )
        fold_records.append({"outer_fold": outer_fold, **metrics})
        test["ablation"] = name
        predictions.append(test)

    oof = pd.concat(predictions, ignore_index=True)
    metrics = aggregate_metrics(oof, "ablation_score")
    metrics["unavailable_top_action_rate"] = ablation.unavailable_top_rate(
        oof, "ablation_score"
    )
    metrics["folds"] = fold_records
    return oof, metrics


def finalize_marker() -> None:
    summary_path = ABLATION_OUT / "SUMMARY.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    expected = set(ablation.ABLATIONS)
    completed = set(summary["ablation"].astype(str))
    payload = json.loads(MARKER.read_text(encoding="utf-8"))
    payload.update(
        {
            "status": "COMPLETE" if expected.issubset(completed) else "PARTIAL",
            "expected_ablations": sorted(expected),
            "completed_ablations": sorted(completed),
        }
    )
    atomic_json(MARKER, payload)


def main() -> None:
    archive_reduced_budget_ablations_once()
    ablation.evaluate_ablation = exact_evaluate_ablation
    ablation.main()
    finalize_marker()


if __name__ == "__main__":
    main()
