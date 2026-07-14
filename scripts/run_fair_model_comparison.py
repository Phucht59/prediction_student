"""Nested-CV comparison of the approved ML and deep-learning candidates.

This runner evaluates only the immutable development cohort.  The locked test
split is excluded from model selection and from this comparison; it may be
used later once, only for a pre-specified final model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import DATASETS, DEFAULT_SEED, ROOT_DIR
from src.data_pipeline import SOURCE_ROW_NUMBER_COLUMN, process_target_and_stratify
from src.evaluation.protocol import DEFAULT_FOLD_MANIFEST_PATH, load_fold_manifest, outer_folds_from_manifest, source_record_identity
from src.model_selection import fit_fold_predict_proba, metric_summary_from_predictions, run_optuna_cv_search, write_json
from src.models import create_model
from src.postgres_data_source import load_dataset_version_from_postgres


CLASSICAL_MODELS = ("decision_tree", "random_forest", "svm_rbf", "xgboost", "gradient_boosting")
DEEP_MODELS = ("cnn_lstm", "cnn_bilstm")
ALL_MODELS = CLASSICAL_MODELS + DEEP_MODELS
REFERENCE_MODEL = "cnn_bilstm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="student-mat", choices=sorted(DATASETS))
    parser.add_argument("--dataset-version-id", required=True, type=int)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST_PATH)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, check=True, text=True, capture_output=True).stdout.strip()


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _development_frame(frame: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    expected = {int(row["source_row_number"]) for row in manifest["development_records"]}
    selected = frame[frame[SOURCE_ROW_NUMBER_COLUMN].astype(int).isin(expected)].copy()
    if len(selected) != len(expected):
        raise ValueError("Dataset does not contain every record in the immutable development manifest.")
    return selected.sort_values(SOURCE_ROW_NUMBER_COLUMN).reset_index(drop=True)


def _settings(args: argparse.Namespace) -> tuple[int, int]:
    if args.smoke:
        return 1, 1
    if (args.outer_folds, args.inner_folds, args.n_trials) != (5, 3, 30):
        raise ValueError("Official comparison requires exactly 5 outer folds, 3 inner folds and 30 trials.")
    return 5, 30


def _classical_params(trial: optuna.Trial, name: str) -> dict[str, Any]:
    if name == "decision_tree":
        return {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_categorical("max_depth", [2, 3, 4, 5, None]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 4, 8]),
            "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 4, 8]),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300]),
            "max_depth": trial.suggest_categorical("max_depth", [3, 5, 8, None]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 4]),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 1.0]),
        }
    if name == "svm_rbf":
        return {
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 1.0, log=True),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [75, 150, 250]),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if name == "gradient_boosting":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [75, 150, 250]),
            "max_depth": trial.suggest_int("max_depth", 1, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 4]),
        }
    raise ValueError(f"Unsupported classical model: {name}")


def _classical_estimator(name: str, params: dict[str, Any], seed: int) -> Pipeline:
    if name == "decision_tree":
        estimator = DecisionTreeClassifier(random_state=seed, **params)
    elif name == "random_forest":
        estimator = RandomForestClassifier(random_state=seed, n_jobs=1, class_weight=None, **params)
    elif name == "svm_rbf":
        estimator = SVC(kernel="rbf", probability=True, class_weight=None, random_state=seed, **params)
    elif name == "xgboost":
        estimator = XGBClassifier(
            random_state=seed,
            n_jobs=1,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            **params,
        )
    elif name == "gradient_boosting":
        estimator = GradientBoostingClassifier(random_state=seed, **params)
    else:
        raise ValueError(f"Unsupported classical model: {name}")
    # Every model receives the same train-fold-only feature transform. Scaling
    # is neutral for trees and required for an RBF SVM.
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def _predict_proba(estimator: Pipeline, x: np.ndarray) -> np.ndarray:
    probabilities = estimator.predict_proba(x)
    labels = estimator.classes_.astype(int)
    if np.array_equal(labels, np.arange(3)):
        return probabilities
    full = np.zeros((len(x), 3), dtype=float)
    full[:, labels] = probabilities
    return full


def _select_classical(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_trials: int,
    n_splits: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True))

    def objective(trial: optuna.Trial) -> float:
        params = _classical_params(trial, name)
        scores = []
        for fold_index, (train_idx, validation_idx) in enumerate(folds.split(x, y)):
            estimator = _classical_estimator(name, params, seed + trial.number * n_splits + fold_index)
            estimator.fit(x[train_idx], y[train_idx])
            predictions = np.argmax(_predict_proba(estimator, x[validation_idx]), axis=1)
            scores.append(float(f1_score(y[validation_idx], predictions, average="macro", zero_division=0)))
        trial.set_user_attr("fold_macro_f1", scores)
        return float(np.mean(scores))

    study.optimize(objective, n_trials=n_trials)
    history = [
        {"number": trial.number, "value": float(trial.value), "params": trial.params, "fold_macro_f1": trial.user_attrs["fold_macro_f1"]}
        for trial in study.trials
        if trial.value is not None
    ]
    return {"best_cv_f1_macro": float(study.best_value), "best_params": dict(study.best_params)}, history


def run(args: argparse.Namespace) -> Path:
    outer_count, n_trials = _settings(args)
    spec = DATASETS[args.dataset]
    raw, dataset_version = load_dataset_version_from_postgres(args.dataset, args.dataset_version_id)
    manifest = load_fold_manifest(args.fold_manifest)
    if int(manifest["dataset_version_id"]) != args.dataset_version_id:
        raise ValueError("Dataset version must match the shared fold manifest.")
    processed = process_target_and_stratify(raw.copy(), spec.target_col, spec.kind, "3class")
    development = _development_frame(processed.dropna(subset=["_strat_target"]).drop(columns=["_strat_target"]), manifest)
    outer_folds = outer_folds_from_manifest(development, manifest, source_column=SOURCE_ROW_NUMBER_COLUMN)
    outer_folds = outer_folds[:outer_count]

    run_id = args.run_id or f"fair-model-comparison-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output = ROOT_DIR / "artifacts" / "baseline_comparison" / run_id
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    protocol = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_checksum": dataset_version["content_hash"],
        "dataset_version_id": args.dataset_version_id,
        "development_records": len(development),
        "locked_test_used_for_selection": False,
        "locked_test_used_for_evaluation": False,
        "features": ["G1", "G2"],
        "models": list(ALL_MODELS),
        "reference_model": REFERENCE_MODEL,
        "outer_folds": len(outer_folds),
        "inner_folds": args.inner_folds,
        "trials_per_model_per_outer_fold": n_trials,
        "selection_metric": "mean inner-CV Macro-F1",
        "imbalance_policy": "none for every model",
        "preprocessing": "StandardScaler fit on each classical training fold only; deep sequence preprocessing fit on training fold only",
        "decision_rule": "argmax",
        "smoke": args.smoke,
    }
    write_json(output / "protocol.json", protocol)

    x = development[["G1", "G2"]].to_numpy(dtype=float)
    y = development[spec.target_col].astype(int).to_numpy()
    all_rows: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for name in ALL_MODELS:
        probabilities = np.zeros((len(development), 3), dtype=float)
        predictions = np.zeros(len(development), dtype=int)
        fold_ids = np.full(len(development), -1, dtype=int)
        parameters: list[float] = []
        for fold_index, (train_idx, validation_idx) in enumerate(outer_folds):
            if name in CLASSICAL_MODELS:
                selected, history = _select_classical(name, x[train_idx], y[train_idx], n_trials=n_trials, n_splits=args.inner_folds, seed=args.seed + fold_index)
                estimator = _classical_estimator(name, selected["best_params"], args.seed)
                estimator.fit(x[train_idx], y[train_idx])
                fold_probabilities = _predict_proba(estimator, x[validation_idx])
                parameters.append(float("nan"))
            else:
                selected, history, _ = run_optuna_cv_search(
                    development.iloc[train_idx].copy(), spec, n_trials=n_trials, n_splits=args.inner_folds,
                    seed=args.seed + fold_index, architecture_variant=name, fair_comparison=True,
                )
                result = fit_fold_predict_proba(
                    train_fold=development.iloc[train_idx].copy(), validation_fold=development.iloc[validation_idx].copy(),
                    spec=spec, params=selected["best_params"], seed=args.seed, fold_index=fold_index,
                )
                fold_probabilities = result.probabilities
                parameters.append(float(sum(parameter.numel() for parameter in create_model(spec.kind, selected["best_params"]).parameters())))
            probabilities[validation_idx] = fold_probabilities
            predictions[validation_idx] = np.argmax(fold_probabilities, axis=1)
            fold_ids[validation_idx] = fold_index
            configs.append({
                "model": name, "outer_fold": fold_index, "inner_best_macro_f1": selected["best_cv_f1_macro"],
                "parameter_count": parameters[-1], "best_params": json.dumps(selected["best_params"], sort_keys=True),
                "trial_history": json.dumps(history, sort_keys=True),
            })
        evaluated = fold_ids >= 0
        summary = metric_summary_from_predictions(y[evaluated], probabilities[evaluated], predictions[evaluated], fold_ids[evaluated])
        summaries.append({
            "model": name,
            "mean_parameter_count": None if name in CLASSICAL_MODELS else float(np.mean(parameters)),
            "outer_macro_f1_mean": summary["cv_f1_macro_mean"], "outer_macro_f1_std": summary["cv_f1_macro_std"],
            "oof_macro_f1": summary["f1_macro"], "oof_accuracy": summary["accuracy"],
            "oof_brier": summary["brier_score"], "oof_ece": summary["ece"],
        })
        for index in np.flatnonzero(evaluated):
            all_rows.append({
                "model": name, "source_record_id": source_record_identity(args.dataset_version_id, int(development.iloc[index][SOURCE_ROW_NUMBER_COLUMN])),
                "outer_fold": int(fold_ids[index]), "true_label": int(y[index]), "predicted_label": int(predictions[index]),
                "probability_low": float(probabilities[index, 0]), "probability_medium": float(probabilities[index, 1]), "probability_high": float(probabilities[index, 2]),
            })

    summary_frame = pd.DataFrame(summaries).sort_values("outer_macro_f1_mean", ascending=False)
    oof = pd.DataFrame(all_rows)
    paired = []
    reference = oof[oof.model == REFERENCE_MODEL]
    for name in ALL_MODELS:
        if name == REFERENCE_MODEL:
            continue
        candidate = oof[oof.model == name]
        for fold in sorted(reference.outer_fold.unique()):
            ref_fold, candidate_fold = reference[reference.outer_fold == fold], candidate[candidate.outer_fold == fold]
            if ref_fold.source_record_id.tolist() != candidate_fold.source_record_id.tolist():
                raise RuntimeError("Outer-fold membership differs across models.")
            paired.append({
                "model": name, "reference_model": REFERENCE_MODEL, "outer_fold": int(fold),
                "model_macro_f1": float(f1_score(candidate_fold.true_label, candidate_fold.predicted_label, average="macro", zero_division=0)),
                "reference_macro_f1": float(f1_score(ref_fold.true_label, ref_fold.predicted_label, average="macro", zero_division=0)),
            })
    paired_frame = pd.DataFrame(paired)
    paired_frame["model_minus_reference"] = paired_frame["model_macro_f1"] - paired_frame["reference_macro_f1"]
    summary_frame.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(configs).to_csv(output / "selected_configs.csv", index=False)
    oof.to_csv(output / "outer_oof_predictions.csv", index=False)
    paired_frame.to_csv(output / "paired_macro_f1_deltas.csv", index=False)
    write_json(output / "checksums.json", {path.name: _checksum(path) for path in output.glob("*.csv")})
    return output


if __name__ == "__main__":
    print(run(parse_args()))
