"""Trial- and inner-fold-resumable full-grid scientific evaluation for V2.1.

The previous full-grid wrapper delegated an entire outer fold to one process.
An interruption therefore discarded every unfinished trial in that fold.  This
runner writes one JSON checkpoint per (outer fold, trial, inner fold), promotes
only a fully completed search into the official ``final_oof`` namespace, and
never overwrites historical evidence without archiving it first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import corrected_nested_evaluation as evaluator
from run_full_registered_search import full_candidate_grid
from scientific_core import (
    FeaturePreprocessor,
    RelevanceTransformer,
    add_baseline_scores,
    aggregate_metrics,
    fit_ranker,
    model_selection_key,
    predict_ranker,
)

OUT = evaluator.OUT
DATA = evaluator.DATA
OFFICIAL_FINAL = evaluator.FINAL
OFFICIAL_SELECTION = evaluator.MODEL_SELECTION
WORK = OUT / "full_grid_resumable"
WORK_FINAL = WORK / "final_oof"
WORK_SELECTION = WORK / "model_selection"
MARKER = OUT / "FULL_REGISTERED_SEARCH.json"
SEED = evaluator.SEED


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["group_id", "action_family"], kind="stable").reset_index(drop=True)


def trial_directory(outer_fold: int, trial_number: int) -> Path:
    return WORK_SELECTION / f"fold_{outer_fold}" / "trials" / f"trial_{trial_number:03d}"


def load_or_run_inner_fold(
    raw_outer_train: pd.DataFrame,
    family: str,
    parameters: dict[str, Any],
    outer_fold: int,
    trial_number: int,
    inner_fold: int,
    train_index: np.ndarray,
    validation_index: np.ndarray,
) -> dict[str, Any]:
    directory = trial_directory(outer_fold, trial_number)
    path = directory / f"inner_{inner_fold}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    inner_train_raw = ordered(raw_outer_train.iloc[train_index].copy())
    inner_validation_raw = ordered(raw_outer_train.iloc[validation_index].copy())
    seed = SEED + outer_fold * 100_000 + trial_number * 100 + inner_fold
    try:
        relevance = RelevanceTransformer(seed=seed)
        inner_train = relevance.fit_transform(inner_train_raw)
        inner_validation = relevance.transform(inner_validation_raw)
        preprocessor = FeaturePreprocessor(include_interactions=True)
        train_matrix = preprocessor.fit_transform(inner_train)
        validation_matrix = preprocessor.transform(inner_validation)
        ranker = fit_ranker(family, train_matrix, inner_train, parameters, seed)
        inner_validation["model_score"] = predict_ranker(ranker, validation_matrix)
        payload: dict[str, Any] = {
            "status": "COMPLETE",
            "outer_fold": outer_fold,
            "trial_number": trial_number,
            "inner_fold": inner_fold,
            "model": family,
            "parameters": parameters,
            **aggregate_metrics(inner_validation, "model_score"),
        }
    except Exception as exc:  # keep a durable diagnostic rather than losing work
        payload = {
            "status": "ERROR",
            "outer_fold": outer_fold,
            "trial_number": trial_number,
            "inner_fold": inner_fold,
            "model": family,
            "parameters": parameters,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    atomic_json(path, payload)
    return payload


def run_trial(
    raw_outer_train: pd.DataFrame,
    family: str,
    parameters: dict[str, Any],
    outer_fold: int,
    trial_number: int,
    inner_splits: int,
) -> dict[str, Any]:
    directory = trial_directory(outer_fold, trial_number)
    summary_path = directory / "trial.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    groups = raw_outer_train["base_record_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    folds = min(inner_splits, len(unique_groups))
    if folds < 2:
        payload = {
            "status": "INSUFFICIENT_INNER_GROUPS",
            "outer_fold": outer_fold,
            "trial_number": trial_number,
            "model": family,
            "parameters": parameters,
        }
        atomic_json(summary_path, payload)
        return payload

    splitter = GroupKFold(n_splits=folds)
    inner_results = []
    for inner_fold, (train_index, validation_index) in enumerate(
        splitter.split(raw_outer_train, groups=groups)
    ):
        inner_results.append(
            load_or_run_inner_fold(
                raw_outer_train,
                family,
                parameters,
                outer_fold,
                trial_number,
                inner_fold,
                train_index,
                validation_index,
            )
        )

    errors = [item for item in inner_results if item.get("status") != "COMPLETE"]
    if errors:
        payload = {
            "status": "ERROR",
            "outer_fold": outer_fold,
            "trial_number": trial_number,
            "model": family,
            "parameters": parameters,
            "inner_folds": inner_results,
            "error": "At least one inner fold failed",
        }
    else:
        ndcg = [float(item["ndcg_at_3"]) for item in inner_results]
        precision = [float(item["precision_at_1"]) for item in inner_results]
        diversity = [float(item["action_diversity"]) for item in inner_results]
        payload = {
            "status": "COMPLETE",
            "outer_fold": outer_fold,
            "trial_number": trial_number,
            "model": family,
            "parameters": parameters,
            "inner_folds": inner_results,
            "mean_ndcg_at_3": float(np.mean(ndcg)),
            "std_ndcg_at_3": float(np.std(ndcg)),
            "worst_ndcg_at_3": float(np.min(ndcg)),
            "mean_precision_at_1": float(np.mean(precision)),
            "mean_action_diversity": float(np.mean(diversity)),
        }
    atomic_json(summary_path, payload)
    return payload


def collect_trials(outer_fold: int, expected: int) -> list[dict[str, Any]]:
    output = []
    for trial_number in range(expected):
        path = trial_directory(outer_fold, trial_number) / "trial.json"
        if path.exists():
            output.append(json.loads(path.read_text(encoding="utf-8")))
    return output


def write_selection(outer_fold: int, trials: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    expected = len(full_candidate_grid(config))
    if len(trials) != expected:
        raise RuntimeError(f"Outer fold {outer_fold} has {len(trials)}/{expected} durable trial summaries")
    completed = [item for item in trials if item.get("status") == "COMPLETE"]
    required_families = set(config["models"]["candidates"])
    evaluated_families = {str(item["model"]) for item in completed}
    missing = sorted(required_families.difference(evaluated_families))
    if missing:
        raise RuntimeError(f"Outer fold {outer_fold} has no completed trial for: {missing}")
    selected = max(completed, key=model_selection_key)
    selected = {
        **selected,
        "selection_rule": [
            "mean_ndcg_at_3",
            "mean_precision_at_1",
            "worst_ndcg_at_3",
            "mean_action_diversity",
        ],
        "expected_trials": expected,
        "completed_trials": len(completed),
    }
    WORK_SELECTION.mkdir(parents=True, exist_ok=True)
    pd.json_normalize(trials, sep=".").to_csv(
        WORK_SELECTION / f"fold_{outer_fold}_trials.csv", index=False
    )
    atomic_json(WORK_SELECTION / f"fold_{outer_fold}_selected.json", selected)
    return selected


def fit_outer_fold(
    raw: pd.DataFrame,
    outer_fold: int,
    selected: dict[str, Any],
) -> pd.DataFrame:
    fold_directory = WORK_FINAL / f"fold_{outer_fold}"
    predictions_path = fold_directory / "predictions.parquet"
    if predictions_path.exists() and (fold_directory / "selected_model.json").exists():
        return pd.read_parquet(predictions_path)

    raw_train = ordered(raw[raw["outer_fold"] != outer_fold].copy())
    raw_test = ordered(raw[raw["outer_fold"] == outer_fold].copy())
    relevance = RelevanceTransformer(seed=SEED + outer_fold)
    train = relevance.fit_transform(raw_train)
    test = relevance.transform(raw_test)
    preprocessor = FeaturePreprocessor(include_interactions=True)
    train_matrix = preprocessor.fit_transform(train)
    test_matrix = preprocessor.transform(test)
    ranker = fit_ranker(
        str(selected["model"]),
        train_matrix,
        train,
        dict(selected["parameters"]),
        SEED + outer_fold,
    )
    test["model_score"] = predict_ranker(ranker, test_matrix)
    test = add_baseline_scores(train, test, seed=SEED + outer_fold)
    methods = [
        "model_score",
        "random_debug_score",
        "popular_score",
        "workload_score",
        "policy_score",
        "counterfactual_score",
    ]
    metrics = {method: aggregate_metrics(test, method) for method in methods}
    fold_directory.mkdir(parents=True, exist_ok=True)
    atomic_parquet(predictions_path, test)
    atomic_json(fold_directory / "metrics.json", metrics)
    atomic_json(
        fold_directory / "selected_model.json",
        {
            "outer_fold": outer_fold,
            "selected_model": selected,
            "all_registered_configurations_evaluated": True,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_learners": int(train["base_record_id"].nunique()),
            "test_learners": int(test["base_record_id"].nunique()),
        },
    )
    joblib.dump(
        {
            "relevance_transformer": relevance,
            "feature_preprocessor": preprocessor,
            "ranker": ranker,
        },
        fold_directory / "scientific_model.joblib",
    )
    atomic_json(
        fold_directory / "CHECKSUMS.json",
        {
            path.name: sha256(path)
            for path in fold_directory.iterdir()
            if path.is_file() and path.name != "CHECKSUMS.json"
        },
    )
    return test


def aggregate_work(config: dict[str, Any]) -> None:
    predictions = [
        pd.read_parquet(WORK_FINAL / f"fold_{int(fold)}/predictions.parquet")
        for fold in config["evaluation"]["outer_folds"]
    ]
    oof = pd.concat(predictions, ignore_index=True)
    atomic_parquet(WORK_FINAL / "OOF_RANKING_PREDICTIONS.parquet", oof)
    methods = [
        "model_score",
        "random_debug_score",
        "popular_score",
        "workload_score",
        "policy_score",
        "counterfactual_score",
    ]
    metrics = {method: aggregate_metrics(oof, method) for method in methods}
    repetitions = int(config["evaluation"]["random_baseline_repetitions"])
    null = evaluator.random_null_fast(oof, repetitions, SEED)
    random_summary = {
        "repetitions": repetitions,
        "mean": float(np.mean(null)),
        "std": float(np.std(null)),
        "ci95_low": float(np.quantile(null, 0.025)),
        "ci95_high": float(np.quantile(null, 0.975)),
        "p95": float(np.quantile(null, 0.95)),
        "p99": float(np.quantile(null, 0.99)),
    }
    fold_metrics = {
        str(fold): json.loads(
            (WORK_FINAL / f"fold_{int(fold)}/metrics.json").read_text(encoding="utf-8")
        )
        for fold in config["evaluation"]["outer_folds"]
    }
    selections = [
        json.loads(
            (WORK_SELECTION / f"fold_{int(fold)}_selected.json").read_text(encoding="utf-8")
        )
        for fold in config["evaluation"]["outer_folds"]
    ]
    result = {
        "status": "COMPLETE",
        "execution": "FULL_REGISTERED_GRID_TRIAL_RESUMABLE",
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "selected_models": selections,
        "models_actually_evaluated": sorted(
            {str(item["model"]) for fold in config["evaluation"]["outer_folds"] for item in collect_trials(int(fold), len(full_candidate_grid(config))) if item.get("status") == "COMPLETE"}
        ),
        "expected_trials_per_outer_fold": len(full_candidate_grid(config)),
        "random_null": random_summary,
        "learners": int(oof["base_record_id"].nunique()),
        "groups": int(oof["group_id"].nunique()),
        "candidate_rows": int(len(oof)),
        "claim_boundary": config["claim_boundary"],
    }
    atomic_json(WORK_FINAL / "NESTED_OOF_RESULTS.json", result)
    rows = [{"method": method, **values} for method, values in metrics.items()]
    rows.append({"method": "random_null_distribution", **random_summary})
    pd.DataFrame(rows).to_csv(WORK_FINAL / "BASELINE_COMPARISON.csv", index=False)
    atomic_json(
        WORK_FINAL / "CHECKSUMS.json",
        {
            str(path.relative_to(WORK_FINAL)).replace("\\", "/"): sha256(path)
            for path in sorted(WORK_FINAL.rglob("*"))
            if path.is_file() and path.name != "CHECKSUMS.json"
        },
    )


def next_archive_path(name: str) -> Path:
    base = OUT / name
    if not base.exists():
        return base
    counter = 2
    while (OUT / f"{name}_{counter}").exists():
        counter += 1
    return OUT / f"{name}_{counter}"


def promote_completed_search(config: dict[str, Any]) -> None:
    expected = len(full_candidate_grid(config))
    for fold in config["evaluation"]["outer_folds"]:
        if len(collect_trials(int(fold), expected)) != expected:
            raise RuntimeError(f"Cannot promote: outer fold {fold} is incomplete")
        if not (WORK_FINAL / f"fold_{int(fold)}/predictions.parquet").exists():
            raise RuntimeError(f"Cannot promote: outer fold {fold} has no predictions")
    aggregate_work(config)

    if OFFICIAL_FINAL.exists():
        shutil.move(str(OFFICIAL_FINAL), str(next_archive_path("interrupted_full_grid_final_oof_archive")))
    if OFFICIAL_SELECTION.exists():
        shutil.move(str(OFFICIAL_SELECTION), str(next_archive_path("interrupted_full_grid_model_selection_archive")))
    shutil.copytree(WORK_FINAL, OFFICIAL_FINAL)
    shutil.copytree(WORK_SELECTION, OFFICIAL_SELECTION)

    payload = json.loads(MARKER.read_text(encoding="utf-8")) if MARKER.exists() else {}
    payload.update(
        {
            "status": "COMPLETE",
            "execution": "TRIAL_AND_INNER_FOLD_RESUMABLE",
            "expected_trials_per_outer_fold": expected,
            "official_final_oof_sha256": sha256(OFFICIAL_FINAL / "NESTED_OOF_RESULTS.json"),
            "folds": [
                {
                    "outer_fold": int(fold),
                    "trial_count": expected,
                    "selected": json.loads(
                        (OFFICIAL_SELECTION / f"fold_{int(fold)}_selected.json").read_text(encoding="utf-8")
                    ),
                }
                for fold in config["evaluation"]["outer_folds"]
            ],
        }
    )
    atomic_json(MARKER, payload)


def update_marker(config: dict[str, Any], requested_folds: list[int]) -> None:
    expected = len(full_candidate_grid(config))
    payload = json.loads(MARKER.read_text(encoding="utf-8")) if MARKER.exists() else {}
    payload.update(
        {
            "status": "RUNNING",
            "execution": "TRIAL_AND_INNER_FOLD_RESUMABLE",
            "expected_trials_per_outer_fold": expected,
            "requested_folds": requested_folds,
            "progress": {
                str(fold): len(collect_trials(fold, expected))
                for fold in config["evaluation"]["outer_folds"]
            },
        }
    )
    atomic_json(MARKER, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", choices=["all", "0", "1", "2"], default="all")
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--trial-stop", type=int, default=None)
    args = parser.parse_args()

    config = evaluator.load_config()
    candidates = full_candidate_grid(config)
    folds = [int(value) for value in config["evaluation"]["outer_folds"]]
    requested_folds = folds if args.outer_fold == "all" else [int(args.outer_fold)]
    stop = len(candidates) if args.trial_stop is None else min(args.trial_stop, len(candidates))
    if args.trial_start < 0 or args.trial_start >= stop:
        raise ValueError("Invalid trial range")

    WORK_FINAL.mkdir(parents=True, exist_ok=True)
    WORK_SELECTION.mkdir(parents=True, exist_ok=True)
    raw = ordered(pd.read_parquet(DATA / "candidate_rows.parquet"))
    update_marker(config, requested_folds)

    for outer_fold in requested_folds:
        raw_outer_train = ordered(raw[raw["outer_fold"] != outer_fold].copy())
        for trial_number in range(args.trial_start, stop):
            family, parameters = candidates[trial_number]
            run_trial(
                raw_outer_train,
                family,
                parameters,
                outer_fold,
                trial_number,
                int(config["evaluation"]["inner_group_folds"]),
            )
            update_marker(config, requested_folds)

        trials = collect_trials(outer_fold, len(candidates))
        if len(trials) == len(candidates):
            selected = write_selection(outer_fold, trials, config)
            fit_outer_fold(raw, outer_fold, selected)
            evaluator.update_progress(
                f"FULL_GRID_RESUMABLE_FOLD_{outer_fold}",
                "COMPLETE",
                trials=len(trials),
            )

    all_complete = all(
        len(collect_trials(fold, len(candidates))) == len(candidates)
        and (WORK_FINAL / f"fold_{fold}/predictions.parquet").exists()
        for fold in folds
    )
    if all_complete:
        promote_completed_search(config)
        evaluator.update_progress(
            "FULL_REGISTERED_SEARCH", "COMPLETE", trials_per_fold=len(candidates)
        )
        print(MARKER.read_text(encoding="utf-8"))
    else:
        update_marker(config, requested_folds)
        print(MARKER.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
