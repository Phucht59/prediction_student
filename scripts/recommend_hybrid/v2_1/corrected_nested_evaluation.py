"""Corrected nested grouped OOF evaluation for outcome-grounded V2.1.

This evaluator never fits labels, preprocessing, or hyperparameters on an outer
or inner validation partition. Preliminary single-model artifacts remain intact;
all corrected outputs are written below ``final_oof``.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

from scientific_core import (
    FeaturePreprocessor,
    RelevanceTransformer,
    add_baseline_scores,
    aggregate_metrics,
    fit_ranker,
    hyperparameter_configs,
    model_selection_key,
    predict_ranker,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/outcome_grounded_v2_1"
DATA = OUT / "dataset"
FINAL = OUT / "final_oof"
MODEL_SELECTION = OUT / "model_selection"
CONFIG_PATH = ROOT / "configs/recommend_hybrid/outcome_grounded_v2_1.yaml"
SEED = 20260804


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def update_progress(stage: str, status: str, **details: Any) -> None:
    path = OUT / "PROGRESS.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stages": {}}
    payload.setdefault("stages", {})[stage] = {
        "status": status,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **details,
    }
    atomic_json(path, payload)


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["group_id", "action_family"], kind="stable").reset_index(drop=True)


def build_authority(config: dict[str, Any]) -> None:
    dataset_files = sorted(path for path in DATA.rglob("*") if path.is_file())
    payload = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_path": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": sha256(CONFIG_PATH),
        "dataset_files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in dataset_files
        },
        "outer_folds": config["evaluation"]["outer_folds"],
        "inner_group_folds": config["evaluation"]["inner_group_folds"],
        "models": config["models"],
        "baselines": config["baselines"],
        "negative_controls": config["negative_controls"],
        "claim_boundary": config["claim_boundary"],
        "status": "FROZEN_FOR_CORRECTED_SCIENTIFIC_EXECUTION",
    }
    atomic_json(OUT / "SCIENTIFIC_EXECUTION_AUTHORITY.json", payload)


def candidate_grid(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    output: list[tuple[str, dict[str, Any]]] = []
    for family in config["models"]["candidates"]:
        family_config = config["models"].get("hyperparameters", {}).get(family, {})
        for parameters in hyperparameter_configs(family_config):
            parameters = dict(parameters)
            parameters.setdefault("n_jobs", 4)
            output.append((family, parameters))
    return output


def evaluate_inner_candidate(
    raw_outer_train: pd.DataFrame,
    family: str,
    parameters: dict[str, Any],
    inner_splits: int,
    seed: int,
) -> dict[str, Any]:
    groups = raw_outer_train["base_record_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    folds = min(inner_splits, len(unique_groups))
    if folds < 2:
        return {
            "model": family,
            "parameters": parameters,
            "status": "INSUFFICIENT_INNER_GROUPS",
        }

    splitter = GroupKFold(n_splits=folds)
    fold_metrics: list[dict[str, Any]] = []
    try:
        for inner_fold, (train_index, validation_index) in enumerate(
            splitter.split(raw_outer_train, groups=groups)
        ):
            inner_train_raw = ordered(raw_outer_train.iloc[train_index].copy())
            inner_validation_raw = ordered(raw_outer_train.iloc[validation_index].copy())

            relevance = RelevanceTransformer(seed=seed + inner_fold)
            inner_train = relevance.fit_transform(inner_train_raw)
            inner_validation = relevance.transform(inner_validation_raw)

            preprocessor = FeaturePreprocessor(include_interactions=True)
            train_matrix = preprocessor.fit_transform(inner_train)
            validation_matrix = preprocessor.transform(inner_validation)
            ranker = fit_ranker(
                family,
                train_matrix,
                inner_train,
                parameters,
                seed + inner_fold,
            )
            inner_validation["model_score"] = predict_ranker(ranker, validation_matrix)
            metrics = aggregate_metrics(inner_validation, "model_score")
            fold_metrics.append({"inner_fold": inner_fold, **metrics})
    except Exception as exc:
        return {
            "model": family,
            "parameters": parameters,
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    ndcg_values = [float(item["ndcg_at_3"]) for item in fold_metrics]
    precision_values = [float(item["precision_at_1"]) for item in fold_metrics]
    diversity_values = [float(item["action_diversity"]) for item in fold_metrics]
    return {
        "model": family,
        "parameters": parameters,
        "status": "COMPLETE",
        "inner_folds": fold_metrics,
        "mean_ndcg_at_3": float(np.mean(ndcg_values)),
        "std_ndcg_at_3": float(np.std(ndcg_values)),
        "worst_ndcg_at_3": float(np.min(ndcg_values)),
        "mean_precision_at_1": float(np.mean(precision_values)),
        "mean_action_diversity": float(np.mean(diversity_values)),
    }


def select_model(
    raw_outer_train: pd.DataFrame,
    config: dict[str, Any],
    outer_fold: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trials = []
    for trial_number, (family, parameters) in enumerate(candidate_grid(config)):
        result = evaluate_inner_candidate(
            raw_outer_train,
            family,
            parameters,
            int(config["evaluation"]["inner_group_folds"]),
            SEED + outer_fold * 1000 + trial_number * 10,
        )
        result["trial_number"] = trial_number
        trials.append(result)

    completed = [trial for trial in trials if trial.get("status") == "COMPLETE"]
    evaluated_families = {str(trial["model"]) for trial in completed}
    required_families = set(config["models"]["candidates"])
    missing_families = sorted(required_families.difference(evaluated_families))
    if missing_families:
        errors = [trial for trial in trials if trial.get("model") in missing_families]
        raise RuntimeError(
            "Registered model families were not evaluated: "
            f"{missing_families}. Errors: {errors[:4]}"
        )
    if not completed:
        raise RuntimeError("No candidate model completed inner evaluation")

    selected = max(completed, key=model_selection_key)
    selected = {
        **selected,
        "selection_rule": [
            "mean_ndcg_at_3",
            "mean_precision_at_1",
            "worst_ndcg_at_3",
            "mean_action_diversity",
        ],
    }
    return selected, trials


def random_null_fast(
    predictions: pd.DataFrame,
    repetitions: int,
    seed: int,
    batch_size: int = 25,
) -> np.ndarray:
    groups = list(predictions.groupby("group_id", sort=False))
    max_actions = max(len(group) for _, group in groups)
    relevance = np.zeros((len(groups), max_actions), dtype=np.float32)
    mask = np.zeros((len(groups), max_actions), dtype=bool)
    for row, (_, group) in enumerate(groups):
        values = group["graded_relevance"].to_numpy(dtype=np.float32)
        values = values - float(values.min())
        relevance[row, : len(values)] = values
        mask[row, : len(values)] = True

    k = min(3, max_actions)
    discount = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float32))
    ideal = np.sort(relevance, axis=1)[:, ::-1][:, :k]
    ideal_dcg = np.sum(ideal * discount, axis=1)
    ideal_dcg[ideal_dcg == 0] = 1.0

    rng = np.random.default_rng(seed)
    output = np.empty(repetitions, dtype=float)
    cursor = 0
    while cursor < repetitions:
        current = min(batch_size, repetitions - cursor)
        scores = rng.random((current, len(groups), max_actions), dtype=np.float32)
        scores[:, ~mask] = -1.0
        order = np.argsort(-scores, axis=2)[:, :, :k]
        expanded_relevance = np.broadcast_to(
            relevance[None, :, :], (current, len(groups), max_actions)
        )
        gains = np.take_along_axis(expanded_relevance, order, axis=2)
        dcg = np.sum(gains * discount, axis=2)
        output[cursor : cursor + current] = np.mean(
            dcg / ideal_dcg[None, :], axis=1
        )
        cursor += current
    return output


def run_outer_fold(
    raw: pd.DataFrame,
    config: dict[str, Any],
    outer_fold: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_outer_train = ordered(raw[raw["outer_fold"] != outer_fold].copy())
    raw_outer_test = ordered(raw[raw["outer_fold"] == outer_fold].copy())
    if raw_outer_train.empty or raw_outer_test.empty:
        raise RuntimeError(f"Outer fold {outer_fold} has an empty train or test partition")

    selected, trials = select_model(raw_outer_train, config, outer_fold)
    trial_frame = pd.json_normalize(trials, sep=".")
    MODEL_SELECTION.mkdir(parents=True, exist_ok=True)
    trial_frame.to_csv(MODEL_SELECTION / f"fold_{outer_fold}_trials.csv", index=False)
    atomic_json(MODEL_SELECTION / f"fold_{outer_fold}_selected.json", selected)

    relevance = RelevanceTransformer(seed=SEED + outer_fold)
    outer_train = relevance.fit_transform(raw_outer_train)
    outer_test = relevance.transform(raw_outer_test)
    preprocessor = FeaturePreprocessor(include_interactions=True)
    train_matrix = preprocessor.fit_transform(outer_train)
    test_matrix = preprocessor.transform(outer_test)
    ranker = fit_ranker(
        str(selected["model"]),
        train_matrix,
        outer_train,
        dict(selected["parameters"]),
        SEED + outer_fold,
    )
    outer_test["model_score"] = predict_ranker(ranker, test_matrix)
    outer_test = add_baseline_scores(outer_train, outer_test, seed=SEED + outer_fold)

    methods = [
        "model_score",
        "random_debug_score",
        "popular_score",
        "workload_score",
        "policy_score",
        "counterfactual_score",
    ]
    metrics = {method: aggregate_metrics(outer_test, method) for method in methods}
    fold_directory = FINAL / f"fold_{outer_fold}"
    fold_directory.mkdir(parents=True, exist_ok=True)
    atomic_parquet(fold_directory / "predictions.parquet", outer_test)
    atomic_json(fold_directory / "metrics.json", metrics)
    atomic_json(
        fold_directory / "selected_model.json",
        {
            "outer_fold": outer_fold,
            "selected_model": selected,
            "all_registered_families_evaluated": True,
            "train_rows": int(len(outer_train)),
            "test_rows": int(len(outer_test)),
            "train_learners": int(outer_train["base_record_id"].nunique()),
            "test_learners": int(outer_test["base_record_id"].nunique()),
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
    checksums = {
        path.name: sha256(path)
        for path in fold_directory.iterdir()
        if path.is_file() and path.name != "CHECKSUMS.json"
    }
    atomic_json(fold_directory / "CHECKSUMS.json", checksums)
    return outer_test, selected


def main() -> None:
    config = load_config()
    FINAL.mkdir(parents=True, exist_ok=True)
    MODEL_SELECTION.mkdir(parents=True, exist_ok=True)
    build_authority(config)
    update_progress("MODEL_SELECTION", "RUNNING")
    update_progress("FINAL_OOF", "RUNNING")

    raw = pd.read_parquet(DATA / "candidate_rows.parquet")
    raw = ordered(raw)
    all_predictions = []
    selections = []
    for outer_fold in config["evaluation"]["outer_folds"]:
        predictions, selected = run_outer_fold(raw, config, int(outer_fold))
        all_predictions.append(predictions)
        selections.append({"outer_fold": int(outer_fold), **selected})
        update_progress(f"FINAL_FOLD_{outer_fold}", "COMPLETE", rows=len(predictions))

    oof = pd.concat(all_predictions, ignore_index=True)
    atomic_parquet(FINAL / "OOF_RANKING_PREDICTIONS.parquet", oof)
    methods = [
        "model_score",
        "random_debug_score",
        "popular_score",
        "workload_score",
        "policy_score",
        "counterfactual_score",
    ]
    metrics = {method: aggregate_metrics(oof, method) for method in methods}

    random_repetitions = int(config["evaluation"]["random_baseline_repetitions"])
    random_null = random_null_fast(oof, random_repetitions, SEED)
    random_summary = {
        "repetitions": random_repetitions,
        "mean": float(np.mean(random_null)),
        "std": float(np.std(random_null)),
        "ci95_low": float(np.quantile(random_null, 0.025)),
        "ci95_high": float(np.quantile(random_null, 0.975)),
        "p95": float(np.quantile(random_null, 0.95)),
        "p99": float(np.quantile(random_null, 0.99)),
    }

    fold_metrics = {
        str(fold): {
            method: aggregate_metrics(oof[oof["outer_fold"] == fold], method)
            for method in methods
        }
        for fold in config["evaluation"]["outer_folds"]
    }
    stage_metrics = {
        str(stage): {
            method: aggregate_metrics(group, method) for method in methods
        }
        for stage, group in oof.groupby("stage", sort=True)
    }

    result = {
        "status": "CORRECTED_NESTED_OOF_COMPLETE",
        "claim_boundary": config["claim_boundary"],
        "selections": selections,
        "models_actually_evaluated": sorted(config["models"]["candidates"]),
        "metrics": metrics,
        "random_null": random_summary,
        "fold_metrics": fold_metrics,
        "stage_metrics": stage_metrics,
        "learners": int(oof["base_record_id"].nunique()),
        "groups": int(oof["group_id"].nunique()),
        "candidate_rows": int(len(oof)),
    }
    atomic_json(FINAL / "NESTED_OOF_RESULTS.json", result)
    rows = [{"method": method, **value} for method, value in metrics.items()]
    rows.append({"method": "random_null_distribution", **random_summary})
    pd.DataFrame(rows).to_csv(FINAL / "BASELINE_COMPARISON.csv", index=False)
    atomic_json(
        MODEL_SELECTION / "search_registry.json",
        {
            "status": "COMPLETE",
            "outer_folds": config["evaluation"]["outer_folds"],
            "inner_group_folds": config["evaluation"]["inner_group_folds"],
            "models_actually_evaluated": sorted(config["models"]["candidates"]),
            "selection_rule": [
                "mean_ndcg_at_3",
                "mean_precision_at_1",
                "worst_ndcg_at_3",
                "mean_action_diversity",
            ],
        },
    )
    final_checksums = {
        str(path.relative_to(FINAL)).replace("\\", "/"): sha256(path)
        for path in sorted(FINAL.rglob("*"))
        if path.is_file() and path.name != "CHECKSUMS.json"
    }
    atomic_json(FINAL / "CHECKSUMS.json", final_checksums)
    update_progress("MODEL_SELECTION", "COMPLETE")
    update_progress("FINAL_OOF", "COMPLETE")
    update_progress("RELEASE", "PENDING")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
