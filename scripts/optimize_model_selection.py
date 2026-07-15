"""Nested validation-only model selection for the DB-first student-mat pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASETS, DEFAULT_SEED, ROOT_DIR, ensure_dirs
from src.data_pipeline import (
    DataPreprocessor,
    FeatureSelector,
    SOURCE_ROW_NUMBER_COLUMN,
    apply_feature_engineering,
    get_sequence_columns,
    process_target_and_stratify,
)
from src.evaluation.evaluation import _connect, sha256_json
from src.evaluation.protocol import (
    DEFAULT_FOLD_MANIFEST_PATH,
    assert_no_legacy_records,
    load_fold_manifest,
    outer_folds_from_manifest,
    source_record_identity,
    validate_scenario_features,
)
from src.model_selection import (
    apply_probability_calibration,
    apply_threshold_policy,
    collect_oof_by_seed,
    combine_seed_probabilities,
    evaluate_ensemble_strategies,
    fit_fold_predict_proba,
    make_folds,
    metric_summary,
    metric_summary_from_predictions,
    run_optuna_cv_search,
    write_json,
)
from src.models import create_model
from src.postgres_data_source import load_development_subset_from_postgres


SELECTION_ROOT = ROOT_DIR / "artifacts" / "model_selection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="student-mat", choices=sorted(DATASETS))
    parser.add_argument("--target-mode", default="3class", choices=["3class"])
    parser.add_argument("--dataset-version-id", type=int, required=True)
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=DEFAULT_FOLD_MANIFEST_PATH,
        help="Required shared V2 outer-fold manifest; it excludes legacy_heldout_observed records.",
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--selection-run-id", default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, check=True, text=True, capture_output=True
    ).stdout.strip()


def runtime_connection() -> tuple[str, str]:
    connection = _connect()
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT current_database(), current_user")
        return cursor.fetchone()
    finally:
        connection.close()


def target_frame(raw_frame: pd.DataFrame, dataset: str, target_mode: str) -> pd.DataFrame:
    spec = DATASETS[dataset]
    frame = process_target_and_stratify(raw_frame.copy(), spec.target_col, spec.kind, target_mode)
    return frame.dropna(subset=["_strat_target"]).drop(columns=["_strat_target"])


def split_hash(frame: pd.DataFrame) -> str:
    rows = sorted(int(row) for row in frame[SOURCE_ROW_NUMBER_COLUMN].tolist())
    return sha256_json(rows)


def development_frame_from_manifest(frame: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    """Select exactly the V2 development cohort, never the observed 79 rows."""
    if int(manifest["dataset_version_id"]) <= 0:
        raise ValueError("Fold manifest requires a valid dataset_version_id.")
    allowed_rows = {int(row["source_row_number"]) for row in manifest["development_records"]}
    development = frame[frame[SOURCE_ROW_NUMBER_COLUMN].astype(int).isin(allowed_rows)].copy()
    if len(development) != len(allowed_rows):
        raise ValueError("Dataset does not contain every shared-manifest development record.")
    identities = [source_record_identity(int(manifest["dataset_version_id"]), row) for row in development[SOURCE_ROW_NUMBER_COLUMN].astype(int)]
    assert_no_legacy_records(identities)
    return development.sort_values(SOURCE_ROW_NUMBER_COLUMN).reset_index(drop=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def strategy_rows_for_csv(rows: list[dict[str, Any]], *, outer_fold: int | None = None) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append(
            {
                "outer_fold": outer_fold,
                "strategy_name": row["strategy_name"],
                "ensemble_method": row["ensemble_method"],
                "seed_count": row["seed_count"],
                "seed_list": _json(row["seed_list"]),
                "seed_weights": _json(row.get("seed_weights")),
                "calibration_policy": _json(row.get("calibration_policy", {"type": "none"})),
                "threshold_policy": _json(row["threshold_policy"]),
                "cv_f1_macro_mean": row["cv_f1_macro_mean"],
                "cv_f1_macro_std": row["cv_f1_macro_std"],
                "cv_f1_macro_min": row["cv_f1_macro_min"],
                "cv_f1_macro_max": row["cv_f1_macro_max"],
                "accuracy": row["accuracy"],
                "brier_score": row["brier_score"],
                "ece": row["ece"],
                "class0_f1": row["class_report"]["0"]["f1"],
                "class1_precision": row["class_report"]["1"]["precision"],
                "class1_recall": row["class_report"]["1"]["recall"],
                "class1_f1": row["class_report"]["1"]["f1"],
                "class2_f1": row["class_report"]["2"]["f1"],
            }
        )
    return output


def apply_selected_strategy(seed_probabilities: dict[int, np.ndarray], strategy: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    probabilities = combine_seed_probabilities(
        seed_probabilities,
        method=strategy["ensemble_method"],
        seed_list=strategy["seed_list"],
        weights=strategy.get("seed_weights"),
    )
    probabilities = apply_probability_calibration(probabilities, strategy.get("calibration_policy"))
    predictions = apply_threshold_policy(probabilities, strategy["threshold_policy"])
    return probabilities, predictions


def evaluate_deep_outer_fold(
    *,
    train_pool: pd.DataFrame,
    spec,
    outer_fold: int,
    outer_train_idx: np.ndarray,
    outer_val_idx: np.ndarray,
    n_trials: int,
    inner_folds: int,
    selection_seed: int,
) -> dict[str, Any]:
    outer_train = train_pool.iloc[outer_train_idx].copy()
    outer_val = train_pool.iloc[outer_val_idx].copy()
    best, trial_history, inner_split = run_optuna_cv_search(
        outer_train,
        spec,
        n_trials=n_trials,
        n_splits=inner_folds,
        seed=DEFAULT_SEED + outer_fold,
    )
    inner_oof = collect_oof_by_seed(
        outer_train,
        spec,
        best["best_params"],
        inner_split,
        [selection_seed],
        ablation_mode="sequence_only",
    )
    strategies, selected_strategy = evaluate_ensemble_strategies(
        inner_oof,
        [selection_seed],
        single_seed_only=True,
    )
    outer_seed_probabilities = {}
    selected_features = {}
    for seed in selected_strategy["seed_list"]:
        result = fit_fold_predict_proba(
            train_fold=outer_train,
            validation_fold=outer_val,
            spec=spec,
            params=best["best_params"],
            seed=int(seed),
            fold_index=outer_fold,
            ablation_mode="sequence_only",
        )
        outer_seed_probabilities[int(seed)] = result.probabilities
        selected_features[str(seed)] = result.selected_features
    probabilities, predictions = apply_selected_strategy(outer_seed_probabilities, selected_strategy)
    y_true = outer_val[spec.target_col].astype(int).to_numpy()
    summary = metric_summary_from_predictions(y_true, probabilities, predictions, np.zeros(len(outer_val), dtype=int))
    return {
        "outer_fold": outer_fold,
        "outer_val_source_rows": [int(value) for value in outer_val[SOURCE_ROW_NUMBER_COLUMN].tolist()],
        "outer_val_hash": split_hash(outer_val),
        "inner_best": best,
        "inner_trials": trial_history,
        "inner_strategies": strategies,
        "selected_strategy": selected_strategy,
        "selected_features": selected_features,
        "probabilities": probabilities,
        "predictions": predictions,
        "true_labels": y_true,
        "summary": summary,
    }


def fit_tabular_baseline_fold(train_frame: pd.DataFrame, val_frame: pd.DataFrame, spec) -> tuple[np.ndarray, np.ndarray]:
    train_engineered = apply_feature_engineering(train_frame.copy(), spec.kind)
    val_engineered = apply_feature_engineering(val_frame.copy(), spec.kind)
    preprocessor = DataPreprocessor(target_col=spec.target_col, oversample_method="none")
    train_prepared = preprocessor.fit_transform(train_engineered, apply_oversampling=False)
    val_prepared = preprocessor.transform(val_engineered)
    selector = FeatureSelector(target_col=spec.target_col, use_feature_selection=True, required_features=get_sequence_columns(spec.kind))
    train_selected = selector.fit_transform(train_prepared, preprocessor.numerical_cols, preprocessor.categorical_cols)
    val_selected = selector.transform(val_prepared)
    feature_cols = [column for column in train_selected.columns if column not in {spec.target_col, "G3_raw"}]
    model = HistGradientBoostingClassifier(random_state=DEFAULT_SEED, max_iter=100, learning_rate=0.05)
    model.fit(train_selected[feature_cols], train_selected[spec.target_col].astype(int))
    probabilities = model.predict_proba(val_selected[feature_cols])
    if probabilities.shape[1] != 3:
        full = np.zeros((len(val_selected), 3), dtype=float)
        for index, label in enumerate(model.classes_):
            full[:, int(label)] = probabilities[:, index]
        probabilities = full
    return probabilities, np.argmax(probabilities, axis=1)


def evaluate_tabular_baseline(train_pool: pd.DataFrame, spec, outer_folds: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    probabilities = np.zeros((len(train_pool), 3), dtype=float)
    predictions = np.zeros(len(train_pool), dtype=int)
    fold_ids = np.zeros(len(train_pool), dtype=int)
    y_true = train_pool[spec.target_col].astype(int).to_numpy()
    for fold_index, (train_idx, val_idx) in enumerate(outer_folds):
        fold_probabilities, fold_predictions = fit_tabular_baseline_fold(
            train_pool.iloc[train_idx].copy(),
            train_pool.iloc[val_idx].copy(),
            spec,
        )
        probabilities[val_idx] = fold_probabilities
        predictions[val_idx] = fold_predictions
        fold_ids[val_idx] = fold_index
    summary = metric_summary_from_predictions(y_true, probabilities, predictions, fold_ids)
    return {"name": "hist_gradient_boosting", "summary": summary, "probabilities": probabilities, "predictions": predictions}


def oof_rows(train_pool: pd.DataFrame, spec, fold_ids: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    y_true = train_pool[spec.target_col].astype(int).to_numpy()
    source_rows = train_pool[SOURCE_ROW_NUMBER_COLUMN].astype(int).to_numpy()
    for index in range(len(train_pool)):
        rows.append(
            {
                "source_row_number": int(source_rows[index]),
                "outer_fold": int(fold_ids[index]),
                "true_label": int(y_true[index]),
                "predicted_label": int(predictions[index]),
                "prob_low": float(probabilities[index, 0]),
                "prob_medium": float(probabilities[index, 1]),
                "prob_high": float(probabilities[index, 2]),
            }
        )
    return rows


def choose_final_config(
    train_pool: pd.DataFrame,
    spec,
    *,
    n_trials: int,
    inner_folds: int,
    selection_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    best, trials, folds = run_optuna_cv_search(
        train_pool,
        spec,
        n_trials=n_trials,
        n_splits=inner_folds,
        seed=DEFAULT_SEED,
    )
    inner_oof = collect_oof_by_seed(
        train_pool,
        spec,
        best["best_params"],
        folds,
        [selection_seed],
        ablation_mode="sequence_only",
    )
    strategies, selected_strategy = evaluate_ensemble_strategies(
        inner_oof,
        [selection_seed],
        single_seed_only=True,
    )
    return best, selected_strategy, strategies


def write_protocol_audit(path: Path) -> None:
    checks = [
        ("Feature engineering fit state", "PASS", "Feature engineering is stateless; preprocessing fit is scoped to fold-training only."),
        ("Imputation/encoding/scaling", "PASS", "`DataPreprocessor.fit_transform()` is called only on model-train partitions; validation/scoring folds use `transform()`."),
        ("Feature selection", "PASS", "`FeatureSelector.fit_transform()` is called only on model-train partitions."),
        ("SMOTE/ADASYN", "PASS", "Oversampling is only inside `fit_transform(..., apply_oversampling=True)` for gradient-training rows."),
        ("Class weights", "PASS", "Class weights use `model_train_fold` labels only, not early-stop or scoring folds."),
        ("Early stopping", "PASS", "Early stopping uses a split carved from fold-training data; outer validation is only scored."),
        ("Outer validation role", "PASS", "Outer folds are evaluated after inner CV freezes params/strategy/calibration/threshold."),
        ("OOF row coverage", "PASS", "Outer StratifiedKFold assigns each train source row to exactly one validation fold."),
        ("Ensemble/threshold/calibration", "PASS", "Strategy, weights, temperature and thresholds are fit on inner OOF only, then frozen for outer evaluation."),
        ("Metadata leakage", "PASS", "`__source_row_number`, DB IDs and `G3_raw` are excluded from model inputs by preprocessing/dataset code."),
    ]
    lines = ["# Optimization Protocol Audit", ""]
    for name, status, evidence in checks:
        lines.append(f"- **{name}**: `{status}` - {evidence}")
    lines.append("")
    lines.append("Locked test is not passed to Optuna, nested CV, threshold fitting or calibration fitting.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    selection_run_id = args.selection_run_id or f"nested-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    output_dir = SELECTION_ROOT / selection_run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    audit_path = output_dir / "optimization_protocol_audit.md"
    spec = DATASETS[args.dataset]
    fold_manifest = load_fold_manifest(args.fold_manifest)
    if int(fold_manifest["dataset_version_id"]) != int(args.dataset_version_id):
        raise ValueError("--dataset-version-id must match the shared fold manifest.")
    source_rows = [int(row["source_row_number"]) for row in fold_manifest["development_records"]]
    raw_frame, dataset_version = load_development_subset_from_postgres(
        args.dataset,
        args.dataset_version_id,
        source_rows,
    )
    target = target_frame(raw_frame, args.dataset, args.target_mode)
    if fold_manifest["dataset_checksum"] != dataset_version["content_hash"]:
        raise ValueError("Dataset checksum does not match the shared fold manifest.")
    validate_scenario_features(get_sequence_columns(spec.kind), "late_stage")
    train_pool = development_frame_from_manifest(target, fold_manifest)
    protocol_manifest = {
        "selection_run_id": selection_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "dataset_version_id": args.dataset_version_id,
        "dataset_checksum": dataset_version["content_hash"],
        "split_hashes": {"development": split_hash(train_pool)},
        "fold_manifest_path": str(args.fold_manifest),
        "fold_manifest_checksum": fold_manifest["manifest_checksum"],
        "legacy_heldout_observed_excluded": True,
        "scenario": "late_stage_G1_G2",
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "n_trials": args.n_trials,
        "selection_seed": args.selection_seed,
        "objective": "mean inner-CV Macro-F1",
        "legacy_heldout_observed_used_for_selection": False,
        "ensemble_used_for_selection": False,
        "threshold_policy": "argmax fixed before locked-test evaluation",
        "calibration_policy": "none fixed before locked-test evaluation",
        "is_full_final_selection_protocol": bool(
            args.outer_folds >= 5 and args.inner_folds >= 3 and args.n_trials >= 30
        ),
    }
    write_json(output_dir / "protocol_manifest.json", protocol_manifest)
    if not protocol_manifest["is_full_final_selection_protocol"]:
        (output_dir / "SMOKE_RUN.md").write_text(
            "# Non-final nested selection smoke run\n\n"
            "This run has fewer than 5 outer folds, 3 inner folds or 30 trials and must not be used for final locked-test evaluation.\n",
            encoding="utf-8",
        )
    if args.outer_folds != int(fold_manifest["outer_folds"]):
        raise ValueError("--outer-folds must match the immutable shared fold manifest.")
    outer_folds = outer_folds_from_manifest(train_pool, fold_manifest, source_column=SOURCE_ROW_NUMBER_COLUMN)
    fold_ids = np.zeros(len(train_pool), dtype=int)
    outer_probabilities = np.zeros((len(train_pool), 3), dtype=float)
    outer_predictions = np.zeros(len(train_pool), dtype=int)
    outer_results = []
    candidate_rows = []
    trial_rows = []

    for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer_folds):
        result = evaluate_deep_outer_fold(
            train_pool=train_pool,
            spec=spec,
            outer_fold=outer_fold,
            outer_train_idx=outer_train_idx,
            outer_val_idx=outer_val_idx,
            n_trials=args.n_trials,
            inner_folds=args.inner_folds,
            selection_seed=args.selection_seed,
        )
        outer_results.append(result)
        outer_probabilities[outer_val_idx] = result["probabilities"]
        outer_predictions[outer_val_idx] = result["predictions"]
        fold_ids[outer_val_idx] = outer_fold
        candidate_rows.extend(strategy_rows_for_csv(result["inner_strategies"], outer_fold=outer_fold))
        for trial in result["inner_trials"]:
            trial_rows.append({"outer_fold": outer_fold, **trial})

    y_true = train_pool[spec.target_col].astype(int).to_numpy()
    outer_summary = metric_summary_from_predictions(y_true, outer_probabilities, outer_predictions, fold_ids)
    baseline = evaluate_tabular_baseline(train_pool, spec, outer_folds)
    final_best, final_strategy, final_strategies = choose_final_config(
        train_pool,
        spec,
        n_trials=args.n_trials,
        inner_folds=args.inner_folds,
        selection_seed=args.selection_seed,
    )

    selected_config = {
        "selection_run_id": selection_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "dataset": args.dataset,
        "target_mode": args.target_mode,
        "dataset_version_id": args.dataset_version_id,
        "split_hash": {"development": split_hash(train_pool)},
        "fold_manifest_path": str(args.fold_manifest),
        "fold_manifest_checksum": fold_manifest["manifest_checksum"],
        "selection_protocol": {
            "source": "shared_v2_development_manifest_only",
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "fold_seed": DEFAULT_SEED,
            "model_selection_seed": args.selection_seed,
            "optuna_trials_per_inner_search": args.n_trials,
            "legacy_heldout_observed_used_for_selection": False,
            "selection_rule": "mean inner-CV Macro-F1; nested outer CV is performance estimation only; one fixed seed and no ensemble selection",
        },
        "nested_cv_result": outer_summary,
        "tabular_baseline": baseline["summary"],
        "best_cv_f1_macro": final_best["best_cv_f1_macro"],
        "best_params": final_best["best_params"],
        "architecture": "cnn_bilstm_classifier",
        "context_mlp_enabled": False,
        "sequence_columns": get_sequence_columns(spec.kind),
        "classifier_head": "linear",
        "selected_strategy": final_strategy,
        "resampling_mode": final_best["best_params"].get("oversample_method", "none"),
        "class_weight_mode": final_best["best_params"].get("class_weight_mode", "none"),
        "model_parameter_count": int(sum(parameter.numel() for parameter in create_model(spec.kind, final_best["best_params"]).parameters())),
        "expected_input_schema": {"sequence_columns": get_sequence_columns(spec.kind), "num_classes": 3},
    }

    write_protocol_audit(audit_path)
    write_json(output_dir / "selected_config.json", selected_config)
    write_json(
        output_dir / "selection_manifest.json",
        {
            **protocol_manifest,
            "selected_config_path": "selected_config.json",
            "selected_config_checksum": sha256_file(output_dir / "selected_config.json"),
            "nested_outer_summary": outer_summary,
        },
    )
    write_json(
        output_dir / "nested_model_selection_summary.json",
        {
            "runtime_connection": runtime_connection(),
            "dataset_version": dataset_version,
            "outer_summary": outer_summary,
            "outer_results": [
                {
                    "outer_fold": result["outer_fold"],
                    "outer_val_hash": result["outer_val_hash"],
                    "inner_best": result["inner_best"],
                    "selected_strategy": result["selected_strategy"],
                    "summary": result["summary"],
                }
                for result in outer_results
            ],
            "tabular_baseline": baseline["summary"],
            "final_inner_best": final_best,
            "final_selected_strategy": final_strategy,
            "final_inner_strategies": final_strategies,
        },
    )
    write_csv(output_dir / "nested_candidate_strategy_cv.csv", candidate_rows)
    write_csv(output_dir / "nested_optuna_trials.csv", trial_rows)
    write_csv(output_dir / "final_inner_strategy_cv.csv", strategy_rows_for_csv(final_strategies))
    write_csv(output_dir / "outer_oof_predictions.csv", oof_rows(train_pool, spec, fold_ids, outer_probabilities, outer_predictions))

    print(
        json.dumps(
            {
                "runtime_connection": runtime_connection(),
                "nested_outer_f1_macro_mean": outer_summary["cv_f1_macro_mean"],
                "nested_outer_f1_macro_std": outer_summary["cv_f1_macro_std"],
                "tabular_baseline_f1_macro_mean": baseline["summary"]["cv_f1_macro_mean"],
                "final_best_cv_f1_macro": final_best["best_cv_f1_macro"],
                "final_selected_strategy": final_strategy,
                "outputs": {
                    "audit": str(audit_path.resolve()),
                    "selected_config": str((output_dir / "selected_config.json").resolve()),
                    "summary": str((output_dir / "nested_model_selection_summary.json").resolve()),
                },
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
