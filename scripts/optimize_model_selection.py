"""Validation-only model selection for the DB-first student-mat pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASETS, DEFAULT_SEED, FIXED_SEEDS, REPORTS_DIR, ensure_dirs
from src.data_pipeline import (
    SOURCE_ROW_NUMBER_COLUMN,
    get_context_excluded_columns,
    get_sequence_columns,
    process_target_and_stratify,
)
from src.evaluation.evaluation import _connect, canonical_json, sha256_json
from src.model_selection import (
    collect_oof_by_seed,
    evaluate_ensemble_strategies,
    make_folds,
    metric_summary,
    run_optuna_cv_search,
    write_json,
)
from src.postgres_data_source import load_dataset_version_from_postgres, reconstruct_splits_from_run


DEFAULT_REFERENCE_RUN_ID = "647158f5-c055-468d-a1c6-47dd6a580028"
OUTPUT_DIR = REPORTS_DIR / "model_selection"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="student-mat", choices=sorted(DATASETS))
    parser.add_argument("--target-mode", default="3class", choices=["3class"])
    parser.add_argument("--dataset-version-id", type=int, default=1)
    parser.add_argument("--reference-run-id", default=DEFAULT_REFERENCE_RUN_ID)
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def target_frame(raw_frame: pd.DataFrame, dataset: str, target_mode: str) -> pd.DataFrame:
    spec = DATASETS[dataset]
    frame = process_target_and_stratify(raw_frame.copy(), spec.target_col, spec.kind, target_mode)
    return frame.dropna(subset=["_strat_target"]).drop(columns=["_strat_target"])


def runtime_connection() -> tuple[str, str]:
    connection = _connect()
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT current_database(), current_user")
        return cursor.fetchone()
    finally:
        connection.close()


def split_hash(frame: pd.DataFrame) -> str:
    rows = sorted(int(row) for row in frame[SOURCE_ROW_NUMBER_COLUMN].tolist())
    return sha256_json(rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def strategy_rows_for_csv(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        output.append(
            {
                "strategy_name": row["strategy_name"],
                "ensemble_method": row["ensemble_method"],
                "seed_count": row["seed_count"],
                "seed_list": json.dumps(row["seed_list"]),
                "threshold_policy": json.dumps(row["threshold_policy"], sort_keys=True),
                "cv_f1_macro_mean": row["cv_f1_macro_mean"],
                "cv_f1_macro_std": row["cv_f1_macro_std"],
                "cv_f1_macro_min": row["cv_f1_macro_min"],
                "cv_f1_macro_max": row["cv_f1_macro_max"],
                "accuracy": row["accuracy"],
                "recall_macro": row["recall_macro"],
                "class_f1_low": row["class_report"]["0"]["f1"],
                "class_f1_medium": row["class_report"]["1"]["f1"],
                "class_f1_high": row["class_report"]["2"]["f1"],
            }
        )
    return output


def ablation_rows_for_csv(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        output.append(
            {
                "name": row.get("name", row.get("strategy_name")),
                "ablation_mode": row.get("ablation_mode", "hybrid"),
                "seed_count": row["seed_count"],
                "cv_f1_macro_mean": row["cv_f1_macro_mean"],
                "cv_f1_macro_std": row["cv_f1_macro_std"],
                "cv_f1_macro_min": row["cv_f1_macro_min"],
                "cv_f1_macro_max": row["cv_f1_macro_max"],
                "accuracy": row["accuracy"],
                "recall_macro": row["recall_macro"],
                "class_f1_low": row["class_report"]["0"]["f1"],
                "class_f1_medium": row["class_report"]["1"]["f1"],
                "class_f1_high": row["class_report"]["2"]["f1"],
            }
        )
    return output


def oof_prediction_rows(train_pool: pd.DataFrame, y_true: np.ndarray, fold_ids: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> list[dict]:
    rows = []
    source_rows = train_pool[SOURCE_ROW_NUMBER_COLUMN].astype(int).to_numpy()
    for index in range(len(train_pool)):
        rows.append(
            {
                "source_row_number": int(source_rows[index]),
                "fold": int(fold_ids[index]),
                "true_label": int(y_true[index]),
                "predicted_label": int(predictions[index]),
                "prob_low": float(probabilities[index, 0]),
                "prob_medium": float(probabilities[index, 1]),
                "prob_high": float(probabilities[index, 2]),
            }
        )
    return rows


def write_audit(
    *,
    path: Path,
    dataset: str,
    train_pool: pd.DataFrame,
    selected_config: dict,
    oof: dict,
    selected_strategy: dict,
) -> None:
    spec = DATASETS[dataset]
    seq_cols = get_sequence_columns(spec.kind)
    context_exclusions = get_context_excluded_columns(spec.kind)
    seed_key = str(selected_strategy["seed_list"][0])
    selected_features = oof["selected_features_by_seed"].get(seed_key, [[]])[0]
    context_cols = [col for col in selected_features if col not in seq_cols and col not in context_exclusions]
    lines = [
        "# Model Optimization Audit",
        "",
        f"- Dataset: `{dataset}`.",
        f"- Train split records used for model selection: `{len(train_pool)}`.",
        f"- Sequence input columns: `{seq_cols}`.",
        f"- Context MLP candidate columns after fold-0 feature selection for seed {seed_key}: `{context_cols}`.",
        f"- Target column: `{spec.target_col}`. It is removed from feature matrix before preprocessing inputs are built.",
        "- `G3_raw` is retained only as regression-diagnostic metadata in `StudentDataset.reg_label`; it is excluded from sequence/context inputs.",
        "- Protected metadata `__source_row_number` is dropped by `DataPreprocessor`; `record_id` and `dataset_version_id` are never added to the training DataFrame.",
        "- Oversampling location: `DataPreprocessor.fit_transform(..., apply_oversampling=True)` is called only for each CV train fold. Validation/test use `transform()` and are never oversampled.",
        f"- Fixed ensemble seeds considered: `{FIXED_SEEDS}`.",
        "- Checkpoint behavior: final pipeline saves one checkpoint per selected seed; CV model-selection script does not persist fold checkpoints.",
        "- Loss: weighted cross entropy unless Optuna selects focal loss. Class weights are computed from the current training fold labels only.",
        "- Optimizer: Adam. Scheduler: ReduceLROnPlateau on validation F1. Early stopping tracks validation F1.",
        f"- Selected strategy: `{selected_strategy['strategy_name']}`.",
        f"- Selected seeds: `{selected_strategy['seed_list']}`.",
        f"- Selected threshold policy: `{selected_strategy['threshold_policy']}`.",
        "",
        "## Why a single seed can beat the current average ensemble",
        "",
        "Mean-probability ensembling is not guaranteed to improve macro F1. If weaker seeds make correlated mistakes or smooth away a strong seed's class-specific confidence, argmax after averaging can reduce Low/Medium/High balance. The selected policy is therefore chosen from OOF validation only, not from the locked test.",
        "",
        "## Selected Config Hash",
        "",
        f"`{sha256_json(selected_config)}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = DATASETS[args.dataset]
    connection = runtime_connection()
    raw_frame, dataset_version = load_dataset_version_from_postgres(args.dataset, args.dataset_version_id)
    target = target_frame(raw_frame, args.dataset, args.target_mode)
    train_pool, locked_test = reconstruct_splits_from_run(target, args.reference_run_id)
    folds = make_folds(train_pool, spec.target_col, n_splits=args.folds, seed=DEFAULT_SEED)

    best, trials, folds = run_optuna_cv_search(train_pool, spec, n_trials=args.n_trials, n_splits=args.folds, seed=DEFAULT_SEED)
    best_params = best["best_params"]
    oof = collect_oof_by_seed(train_pool, spec, best_params, folds, FIXED_SEEDS, ablation_mode="hybrid")
    strategies, selected_strategy = evaluate_ensemble_strategies(oof, FIXED_SEEDS)

    selected_probabilities = None
    from src.model_selection import apply_threshold_policy, combine_seed_probabilities

    selected_probabilities = combine_seed_probabilities(
        oof["seed_probabilities"],
        method=selected_strategy["ensemble_method"],
        seed_list=selected_strategy["seed_list"],
        weights=selected_strategy.get("seed_weights"),
    )
    selected_predictions = apply_threshold_policy(selected_probabilities, selected_strategy["threshold_policy"])

    ablation_rows = []
    for name, ablation_mode, seeds in [
        ("context_mlp_baseline", "context_only", [DEFAULT_SEED]),
        ("cnn_bilstm_sequence_only", "sequence_only", [DEFAULT_SEED]),
        ("cnn_bilstm_context_mlp", "hybrid", [DEFAULT_SEED]),
    ]:
        ablation_oof = collect_oof_by_seed(train_pool, spec, best_params, folds, seeds, ablation_mode=ablation_mode)
        probabilities = ablation_oof["seed_probabilities"][DEFAULT_SEED]
        summary = metric_summary(ablation_oof["y_true"], probabilities, {"type": "argmax"}, ablation_oof["fold_ids"])
        ablation_rows.append({"name": name, "ablation_mode": ablation_mode, "seed_count": len(seeds), **summary})
    ablation_rows.append({"name": "selected_hybrid_strategy", **selected_strategy})

    selected_config = {
        "dataset": args.dataset,
        "target_mode": args.target_mode,
        "dataset_version_id": args.dataset_version_id,
        "reference_run_id": args.reference_run_id,
        "split_hash": {
            "train": split_hash(train_pool),
            "locked_test": split_hash(locked_test),
        },
        "selection_protocol": {
            "source": "train_split_only",
            "folds": args.folds,
            "fold_seed": DEFAULT_SEED,
            "optuna_trials": args.n_trials,
            "optuna_seed": DEFAULT_SEED,
            "objective": "mean_5fold_oof_f1_macro",
            "locked_test_used_for_selection": False,
        },
        "best_cv_f1_macro": best["best_cv_f1_macro"],
        "best_params": best_params,
        "selected_strategy": selected_strategy,
    }

    write_json(OUTPUT_DIR / "optuna_trials.json", trials)
    write_json(OUTPUT_DIR / "selected_config.json", selected_config)
    write_json(
        OUTPUT_DIR / "model_selection_summary.json",
        {
            "runtime_connection": connection,
            "dataset_version": dataset_version,
            "best": best,
            "selected_strategy": selected_strategy,
            "strategies": strategies,
            "ablation": ablation_rows,
        },
    )
    write_csv(OUTPUT_DIR / "ensemble_strategy_cv.csv", strategy_rows_for_csv(strategies))
    write_csv(OUTPUT_DIR / "ablation_cv.csv", ablation_rows_for_csv(ablation_rows))
    write_csv(
        OUTPUT_DIR / "oof_predictions_selected_strategy.csv",
        oof_prediction_rows(train_pool, oof["y_true"], oof["fold_ids"], selected_probabilities, selected_predictions),
    )
    write_audit(
        path=REPORTS_DIR / "model_optimization_audit.md",
        dataset=args.dataset,
        train_pool=train_pool,
        selected_config=selected_config,
        oof=oof,
        selected_strategy=selected_strategy,
    )

    print(json.dumps({
        "runtime_connection": connection,
        "best_cv_f1_macro": best["best_cv_f1_macro"],
        "best_params": best_params,
        "selected_strategy": selected_strategy,
        "train_split_hash": selected_config["split_hash"]["train"],
        "locked_test_split_hash": selected_config["split_hash"]["locked_test"],
        "outputs": {
            "selected_config": str((OUTPUT_DIR / "selected_config.json").resolve()),
            "audit": str((REPORTS_DIR / "model_optimization_audit.md").resolve()),
        },
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
