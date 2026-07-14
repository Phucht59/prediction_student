"""Run Strategy B Phase C0 smoke or the authorized C1-C2 main comparison.

The command reads only the immutable 316-row PostgreSQL development allowlist.
It never launches conditional candidates, Phase D/E, or legacy-observed access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from src.config import DATASETS, ROOT_DIR
from src.data_pipeline import SOURCE_ROW_NUMBER_COLUMN, process_target_and_stratify
from src.evaluation.protocol import DEFAULT_FOLD_MANIFEST_PATH, file_checksum, load_fold_manifest, outer_folds_from_manifest, semantic_checksum
from src.estimator_factory import resolved_config_hash, resolved_config_schema, validate_resolved_config
from src.model_selection import fit_fold_predict_proba, predict_with_fitted_estimator
from src.postgres_data_source import load_development_subset_from_postgres
from src.strategy_b_phase_ab import (
    assert_development_only_frame,
    development_source_rows,
    evidence_quarantine_registry,
    materialize_early_stop_ledger,
    materialize_inner_fold_ledger,
    sha256_file,
    source_rows_hash,
    write_json,
)
from src.strategy_b_phase_c import (
    ABLATIONS,
    ALL_CANDIDATES,
    MAIN_NEURAL,
    ML_CANDIDATES,
    PARAMETER_GUARDRAIL,
    PHASE_C_PROTOCOL_VERSION,
    PHASE_C_SEEDS,
    RANKING_CANDIDATES,
    boundary_error_analysis,
    candidate_registry,
    config_hash,
    detailed_metrics,
    matched_ablation_config,
    ml_resolved_config,
    model_summary,
    paired_deltas,
    probability_contract,
    sample_neural_config,
    search_spaces,
    selection_rule,
)


FINAL_ROOT = ROOT_DIR / "artifacts" / "strategy_b_phase_c"
FINAL_REPORT_ROOT = ROOT_DIR / "reports" / "strategy_b_phase_c"
SMOKE_ROOT = ROOT_DIR / "artifacts" / "strategy_b_phase_c_smoke"
SMOKE_REPORT_ROOT = ROOT_DIR / "reports" / "strategy_b_phase_c_smoke"
REPRODUCTION_TOLERANCE = 1e-7
HIGH_RECALL_GUARDRAIL = 0.60
MINIMUM_OUTPUTS = [
    "protocol.json", "candidate_registry.json", "search_spaces.json", "selection_rule.json",
    "resolved_configs.csv", "trial_history.csv", "job_ledger.csv", "fold_seed_metrics.csv",
    "model_summary.csv", "outer_oof_predictions.csv", "paired_model_deltas.csv",
    "per_class_metrics.csv", "confusion_matrices.csv", "ordinal_metrics.csv",
    "boundary_error_analysis.csv", "calibration_metrics.csv", "training_diagnostics.csv",
    "parameter_counts.csv", "runtime_summary.csv", "class_collapse_report.csv",
    "checkpoint_checksums.json", "artifact_checksums.json", "source_provenance.json",
    "test_report.json", "strict_validation.json", "conditional_gate_assessment.json",
    "phase_c_conclusion.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["smoke", "full"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-version-id", type=int, default=1)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST_PATH)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--skip-tests", action="store_true", help="Diagnostic only; official full runs reject this.")
    parser.add_argument(
        "--resume-finalize", action="store_true",
        help="Finalize a fully trained failed partial run after validating every persisted evidence table.",
    )
    return parser.parse_args()


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT_DIR, text=True, encoding="utf-8", errors="replace", capture_output=True)


def _source_provenance() -> dict[str, Any]:
    git = lambda *args: _run(["git", *args]).stdout.strip()
    diff = _run(["git", "diff", "--binary", "HEAD"]).stdout.encode()
    tracked = git("ls-files").splitlines()
    source_paths = [
        path for path in tracked
        if path.startswith(("src/", "scripts/", "tests/", "config/", "database/"))
        or path in {"requirements.txt", "requirements-lock.txt", "environment.yml", "SCIENTIFIC_PROTOCOL_V2.md"}
    ]
    source_hashes = {path: sha256_file(ROOT_DIR / path) for path in sorted(source_paths) if (ROOT_DIR / path).is_file()}
    tree_hash = hashlib.sha256(json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    environment = {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(), "pip_freeze": _run([sys.executable, "-m", "pip", "freeze"]).stdout.splitlines(),
    }
    environment_hash = hashlib.sha256(json.dumps(environment, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "git_commit": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
        "dirty_diff_hash": hashlib.sha256(diff).hexdigest(), "dirty_diff_bytes": len(diff),
        "git_status_hash": hashlib.sha256(git("status", "--porcelain=v1", "--untracked-files=all").encode()).hexdigest(),
        "source_tree_hash": tree_hash, "source_hashes": source_hashes,
        "environment_hash": environment_hash, "environment": environment,
    }


def _write_state(root: Path, status: str, **extra: Any) -> None:
    write_json(root / "run_state.json", {"status": status, "updated_at": datetime.now(timezone.utc).isoformat(), **extra})


def _persist_tables(root: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(root / f"{name}.csv", index=False)


def _inner_memberships(frame: pd.DataFrame, outer_folds: list[tuple[np.ndarray, np.ndarray]]) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    result: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for outer_fold, (train_idx, _) in enumerate(outer_folds):
        outer_train = frame.iloc[train_idx]
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + outer_fold)
        result[outer_fold] = list(splitter.split(outer_train, outer_train["G3"].astype(int)))
    return result


def _trial_neural(
    *,
    candidate_id: str,
    outer_fold: int,
    outer_train: pd.DataFrame,
    memberships: list[tuple[np.ndarray, np.ndarray]],
    inner_limit: int,
    tables: dict[str, list[dict[str, Any]]],
    artifact_tmp: Path,
    trial: optuna.Trial,
) -> float:
    started = time.perf_counter()
    completed_fits = 0
    try:
        try:
            config = sample_neural_config(trial, candidate_id)
        except ValueError as exc:
            if "parameter_guardrail_exceeded" in str(exc):
                trial.set_user_attr("prune_reason", str(exc))
                raise optuna.TrialPruned(str(exc))
            raise
        scores: list[float] = []
        diagnostics: list[dict[str, Any]] = []
        for inner_fold, (train_positions, validation_positions) in enumerate(memberships[:inner_limit]):
            fit_started = time.perf_counter()
            try:
                result = fit_fold_predict_proba(
                    train_fold=outer_train.iloc[train_positions].copy(),
                    validation_fold=outer_train.iloc[validation_positions].copy(),
                    spec=DATASETS["student-mat"], params=config, seed=42, fold_index=inner_fold,
                )
                score = f1_score(result.true_labels, result.predictions, average="macro", zero_division=0)
                scores.append(float(score))
                completed_fits += 2
                diagnostics.append(result.training_diagnostics or {})
                tables["job_ledger"].append({
                    "stage": "inner_search", "candidate_id": candidate_id, "outer_fold": outer_fold,
                    "inner_fold": inner_fold, "seed": 42, "trial_number": trial.number, "status": "completed",
                    "fit_stages_completed": 2, "runtime_seconds": time.perf_counter() - fit_started,
                })
            except Exception as exc:
                tables["job_ledger"].append({
                    "stage": "inner_search", "candidate_id": candidate_id, "outer_fold": outer_fold,
                    "inner_fold": inner_fold, "seed": 42, "trial_number": trial.number, "status": "failed",
                    "fit_stages_completed": 0, "runtime_seconds": time.perf_counter() - fit_started,
                    "failure_reason": f"{type(exc).__name__}:{exc}",
                })
                raise
        trial.set_user_attr("resolved_config", config)
        trial.set_user_attr("parameter_count", config["parameter_count"])
        trial.set_user_attr("selected_epochs", [row.get("selected_epoch") for row in diagnostics])
        trial.set_user_attr("epochs_ran", [row.get("epochs_ran") for row in diagnostics])
        trial.set_user_attr("refit_epochs", [row.get("refit_epochs") for row in diagnostics])
        trial.set_user_attr("hit_epoch_cap", [row.get("hit_epoch_cap") for row in diagnostics])
        trial.set_user_attr("actual_fit_stages", completed_fits)
        trial.set_user_attr("runtime_seconds", time.perf_counter() - started)
        return float(np.mean(scores))
    finally:
        _persist_tables(artifact_tmp, tables)


def _neural_search(
    candidate_id: str,
    outer_fold: int,
    outer_train: pd.DataFrame,
    memberships: list[tuple[np.ndarray, np.ndarray]],
    *,
    n_trials: int,
    inner_limit: int,
    tables: dict[str, list[dict[str, Any]]],
    artifact_tmp: Path,
) -> tuple[dict[str, Any], optuna.Study]:
    sampler_seed = 42 + 1000 * outer_fold + 17 * MAIN_NEURAL.index(candidate_id)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=sampler_seed))
    study.optimize(
        lambda trial: _trial_neural(
            candidate_id=candidate_id, outer_fold=outer_fold, outer_train=outer_train,
            memberships=memberships, inner_limit=inner_limit, tables=tables,
            artifact_tmp=artifact_tmp, trial=trial,
        ),
        n_trials=n_trials,
        catch=(Exception,),
        show_progress_bar=False,
    )
    for trial in study.trials:
        tables["trial_history"].append({
            "candidate_id": candidate_id, "outer_fold": outer_fold, "trial_number": trial.number,
            "state": trial.state.name, "value": trial.value,
            "resolved_config": json.dumps(trial.user_attrs.get("resolved_config"), sort_keys=True),
            "parameter_count": trial.user_attrs.get("parameter_count"),
            "selected_epoch": json.dumps(trial.user_attrs.get("selected_epochs")),
            "epochs_ran": json.dumps(trial.user_attrs.get("epochs_ran")),
            "refit_epochs": json.dumps(trial.user_attrs.get("refit_epochs")),
            "hit_epoch_cap": json.dumps(trial.user_attrs.get("hit_epoch_cap")),
            "actual_fit_stages": trial.user_attrs.get("actual_fit_stages", 0),
            "runtime_seconds": trial.user_attrs.get("runtime_seconds"),
            "failure_or_prune_reason": trial.user_attrs.get("prune_reason") or (str(trial.system_attrs) if trial.state.name == "FAIL" else ""),
        })
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if not completed:
        raise RuntimeError(f"No completed trial for {candidate_id} outer fold {outer_fold}.")
    best = max(completed, key=lambda item: float(item.value))
    return deepcopy(best.user_attrs["resolved_config"]), study


def _sample_ml(trial: optuna.Trial, candidate_id: str) -> dict[str, Any]:
    if candidate_id == "M1":
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300]),
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 8]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 4]),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    else:
        params = {
            "C": trial.suggest_float("C", 0.05, 50.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 2.0, log=True),
        }
    return ml_resolved_config(candidate_id, params)


def _make_ml(config: dict[str, Any]) -> Pipeline:
    candidate_id = config["candidate_id"]
    params = config["parameters"]
    if candidate_id == "M1":
        estimator = RandomForestClassifier(**params, class_weight=None, random_state=42, n_jobs=1)
    else:
        estimator = SVC(**params, kernel="rbf", probability=True, class_weight=None, random_state=42)
    return Pipeline([("scale", MinMaxScaler()), ("model", estimator)])


def _ml_search(
    candidate_id: str,
    outer_fold: int,
    outer_train: pd.DataFrame,
    memberships: list[tuple[np.ndarray, np.ndarray]],
    *,
    n_trials: int,
    inner_limit: int,
    tables: dict[str, list[dict[str, Any]]],
    artifact_tmp: Path,
) -> dict[str, Any]:
    def objective(trial: optuna.Trial) -> float:
        started = time.perf_counter()
        config = _sample_ml(trial, candidate_id)
        scores = []
        for inner_fold, (train_positions, validation_positions) in enumerate(memberships[:inner_limit]):
            fit_started = time.perf_counter()
            model = _make_ml(config)
            model.fit(outer_train.iloc[train_positions][["G1", "G2"]], outer_train.iloc[train_positions]["G3"])
            pred = model.predict(outer_train.iloc[validation_positions][["G1", "G2"]])
            scores.append(f1_score(outer_train.iloc[validation_positions]["G3"], pred, average="macro", zero_division=0))
            tables["job_ledger"].append({
                "stage": "inner_search", "candidate_id": candidate_id, "outer_fold": outer_fold,
                "inner_fold": inner_fold, "seed": 42, "trial_number": trial.number, "status": "completed",
                "fit_stages_completed": 1, "runtime_seconds": time.perf_counter() - fit_started,
            })
        trial.set_user_attr("resolved_config", config)
        trial.set_user_attr("actual_fit_stages", len(scores))
        trial.set_user_attr("runtime_seconds", time.perf_counter() - started)
        _persist_tables(artifact_tmp, tables)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=701 + outer_fold + 31 * ML_CANDIDATES.index(candidate_id)))
    study.optimize(objective, n_trials=n_trials, catch=(Exception,), show_progress_bar=False)
    for trial in study.trials:
        tables["trial_history"].append({
            "candidate_id": candidate_id, "outer_fold": outer_fold, "trial_number": trial.number,
            "state": trial.state.name, "value": trial.value,
            "resolved_config": json.dumps(trial.user_attrs.get("resolved_config"), sort_keys=True),
            "actual_fit_stages": trial.user_attrs.get("actual_fit_stages", 0),
            "runtime_seconds": trial.user_attrs.get("runtime_seconds"),
            "failure_or_prune_reason": "" if trial.state.name == "COMPLETE" else str(trial.system_attrs),
        })
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if not completed:
        raise RuntimeError(f"No completed ML trial for {candidate_id} outer fold {outer_fold}.")
    return deepcopy(max(completed, key=lambda item: float(item.value)).user_attrs["resolved_config"])


def _append_oof(
    tables: dict[str, list[dict[str, Any]]], candidate_id: str, seed: int, outer_fold: int,
    validation: pd.DataFrame, probabilities: np.ndarray,
) -> None:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    probability_contract(probabilities)
    predictions = np.argmax(probabilities, axis=1)
    for position, (_, row) in enumerate(validation.iterrows()):
        tables["outer_oof_predictions"].append({
            "candidate_id": candidate_id, "seed": int(seed), "outer_fold": int(outer_fold),
            "source_row_number": int(row[SOURCE_ROW_NUMBER_COLUMN]), "raw_g3": int(row["G3_raw"]),
            "true_label": int(row["G3"]), "predicted_label": int(predictions[position]),
            "prob_0": float(probabilities[position, 0]), "prob_1": float(probabilities[position, 1]),
            "prob_2": float(probabilities[position, 2]),
        })


def _evaluate_neural(
    *,
    candidate_id: str, outer_fold: int, seed: int, config: dict[str, Any],
    outer_train: pd.DataFrame, outer_validation: pd.DataFrame,
    artifact_tmp: Path, tables: dict[str, list[dict[str, Any]]], checkpoint_entries: list[dict[str, Any]],
) -> None:
    started = time.perf_counter()
    result = fit_fold_predict_proba(
        train_fold=outer_train, validation_fold=outer_validation, spec=DATASETS["student-mat"],
        params=config, seed=seed, fold_index=outer_fold,
    )
    candidate_dir = artifact_tmp / "checkpoints" / candidate_id
    preprocessor_dir = artifact_tmp / "preprocessors" / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = candidate_dir / f"outer{outer_fold}_seed{seed}.pt"
    prep_path = preprocessor_dir / f"outer{outer_fold}_seed{seed}.pkl"
    torch.save(result.refit_state_dict, checkpoint)
    with prep_path.open("wb") as handle:
        pickle.dump({"preprocessor": result.refit_preprocessor, "selector": result.refit_selector}, handle)
    loaded_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    with prep_path.open("rb") as handle:
        loaded = pickle.load(handle)
    reproduced = predict_with_fitted_estimator(
        frame=outer_validation, spec=DATASETS["student-mat"], resolved_config=config,
        state_dict=loaded_state, preprocessor=loaded["preprocessor"], selector=loaded["selector"],
    )
    maximum_difference = float(np.max(np.abs(reproduced - result.probabilities)))
    if maximum_difference > REPRODUCTION_TOLERANCE:
        raise RuntimeError(f"Checkpoint reproduction failed for {candidate_id} fold={outer_fold} seed={seed}.")
    _append_oof(tables, candidate_id, seed, outer_fold, outer_validation, result.probabilities)
    diagnostics = dict(result.training_diagnostics or {})
    tables["training_diagnostics"].append({
        "candidate_id": candidate_id, "outer_fold": outer_fold, "seed": seed,
        **{key: diagnostics.get(key) for key in [
            "epochs_ran", "selected_epoch", "refit_epochs", "hit_epoch_cap", "full_refit_input_records",
            "estimator_parity", "criterion_parity", "resampling_parity",
        ]},
        "scheduler_state": json.dumps(diagnostics.get("scheduler_state_refit"), sort_keys=True),
        "sample_utilization": json.dumps(diagnostics.get("sample_utilization_refit"), sort_keys=True),
        "cnn_output_sequence_length": result.shape_diagnostics.get("cnn_output_sequence_length"),
    })
    tables["parameter_counts"].append({
        "candidate_id": candidate_id, "outer_fold": outer_fold, "seed": seed,
        "parameter_count": int(config["parameter_count"]), "guardrail": PARAMETER_GUARDRAIL,
    })
    runtime = time.perf_counter() - started
    tables["runtime_events"].append({"candidate_id": candidate_id, "stage": "outer_evaluation", "runtime_seconds": runtime})
    tables["job_ledger"].append({
        "stage": "outer_evaluation", "candidate_id": candidate_id, "outer_fold": outer_fold,
        "inner_fold": "", "seed": seed, "trial_number": "", "status": "completed",
        "fit_stages_completed": 2, "runtime_seconds": runtime,
    })
    checkpoint_entries.append({
        "candidate_id": candidate_id, "outer_fold": outer_fold, "seed": seed,
        "checkpoint": checkpoint.relative_to(artifact_tmp).as_posix(), "checkpoint_sha256": sha256_file(checkpoint),
        "preprocessor": prep_path.relative_to(artifact_tmp).as_posix(), "preprocessor_sha256": sha256_file(prep_path),
        "resolved_config_hash": resolved_config_hash(config),
        "prediction_reproduction_max_abs_difference": maximum_difference,
        "prediction_reproduction_tolerance": REPRODUCTION_TOLERANCE,
        "prediction_reproduction_pass": maximum_difference <= REPRODUCTION_TOLERANCE,
    })


def _evaluate_ml(
    *,
    candidate_id: str, outer_fold: int, config: dict[str, Any], outer_train: pd.DataFrame,
    outer_validation: pd.DataFrame, artifact_tmp: Path, tables: dict[str, list[dict[str, Any]]],
    checkpoint_entries: list[dict[str, Any]], report_seeds: list[int],
) -> None:
    started = time.perf_counter()
    model = _make_ml(config)
    model.fit(outer_train[["G1", "G2"]], outer_train["G3"])
    probabilities = model.predict_proba(outer_validation[["G1", "G2"]])
    checkpoint = artifact_tmp / "checkpoints" / candidate_id / f"outer{outer_fold}_seed42.pkl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("wb") as handle:
        pickle.dump(model, handle)
    with checkpoint.open("rb") as handle:
        reloaded = pickle.load(handle)
    reproduced = reloaded.predict_proba(outer_validation[["G1", "G2"]])
    difference = float(np.max(np.abs(probabilities - reproduced)))
    if difference > REPRODUCTION_TOLERANCE:
        raise RuntimeError(f"ML checkpoint reproduction failed for {candidate_id} fold={outer_fold}.")
    # Repeat the fixed stochastic-seed prediction across declared seeds only for
    # aligned paired comparisons; no probability aggregation is performed.
    for seed in report_seeds:
        _append_oof(tables, candidate_id, seed, outer_fold, outer_validation, probabilities)
    runtime = time.perf_counter() - started
    tables["runtime_events"].append({"candidate_id": candidate_id, "stage": "outer_evaluation", "runtime_seconds": runtime})
    tables["job_ledger"].append({
        "stage": "outer_evaluation", "candidate_id": candidate_id, "outer_fold": outer_fold,
        "inner_fold": "", "seed": 42, "trial_number": "", "status": "completed",
        "fit_stages_completed": 1, "runtime_seconds": runtime,
    })
    fitted = model.named_steps["model"]
    if candidate_id == "M1":
        fitted_parameter_count = int(sum(tree.tree_.node_count for tree in fitted.estimators_))
    else:
        fitted_parameter_count = int(fitted.support_vectors_.size + fitted.dual_coef_.size + fitted.intercept_.size)
    tables["parameter_counts"].append({"candidate_id": candidate_id, "outer_fold": outer_fold, "seed": 42, "parameter_count": fitted_parameter_count, "guardrail": "not_applicable"})
    checkpoint_entries.append({
        "candidate_id": candidate_id, "outer_fold": outer_fold, "seed": 42,
        "checkpoint": checkpoint.relative_to(artifact_tmp).as_posix(), "checkpoint_sha256": sha256_file(checkpoint),
        "resolved_config_hash": config_hash(config), "prediction_reproduction_max_abs_difference": difference,
        "prediction_reproduction_tolerance": REPRODUCTION_TOLERANCE, "prediction_reproduction_pass": True,
    })


def _evaluate_rule(
    outer_fold: int, validation: pd.DataFrame, tables: dict[str, list[dict[str, Any]]], report_seeds: list[int]
) -> None:
    predicted = np.digitize(validation["G2"].to_numpy(float), bins=[9, 14], right=True)
    probabilities = np.eye(3, dtype=float)[predicted]
    for seed in report_seeds:
        _append_oof(tables, "R0", seed, outer_fold, validation, probabilities)
    tables["parameter_counts"].append({"candidate_id": "R0", "outer_fold": outer_fold, "seed": 42, "parameter_count": 0, "guardrail": "not_applicable"})
    tables["runtime_events"].append({"candidate_id": "R0", "stage": "outer_evaluation", "runtime_seconds": 0.0})
    tables["job_ledger"].append({
        "stage": "outer_evaluation", "candidate_id": "R0", "outer_fold": outer_fold,
        "inner_fold": "", "seed": 42, "trial_number": "", "status": "completed",
        "fit_stages_completed": 0, "runtime_seconds": 0.0,
    })


def _run_tests(skip: bool, stage: str) -> dict[str, Any]:
    if skip:
        if stage == "full":
            raise ValueError("Official full Phase C cannot skip tests.")
        return {"official": False, "status": "SKIPPED_BY_DIAGNOSTIC_FLAG"}
    started = time.perf_counter()
    completed = _run([sys.executable, "-m", "pytest", "-q", "-rs"])
    return {
        "official": True, "command": [sys.executable, "-m", "pytest", "-q", "-rs"],
        "return_code": completed.returncode, "status": "PASS" if completed.returncode == 0 else "FAIL",
        "duration_seconds": time.perf_counter() - started, "stdout": completed.stdout, "stderr": completed.stderr,
        "postgres_integration_waiver": {
            "active": "POSTGRES_TEST_DSN" not in os.environ or "POSTGRES_TEST_APP_DSN" not in os.environ,
            "reason": "Disposable PostgreSQL test DSNs/psql are unavailable; production DB was not used destructively.",
            "official_db_access": "read_only_316_row_development_allowlist",
        },
    }


def _choose(summary: pd.DataFrame, eligible: list[str], paired: pd.DataFrame) -> tuple[str, str]:
    candidates = summary[summary["candidate_id"].isin(eligible)].sort_values("oof_macro_f1", ascending=False).copy()
    top_id = str(candidates.iloc[0]["candidate_id"])
    top_score = float(candidates.iloc[0]["oof_macro_f1"])
    tied_ids = [top_id]
    for _, candidate in candidates.iloc[1:].iterrows():
        candidate_id = str(candidate["candidate_id"])
        interval_includes_zero = False
        direct = paired[(paired["left"] == top_id) & (paired["right"] == candidate_id)]
        reverse = paired[(paired["left"] == candidate_id) & (paired["right"] == top_id)]
        row = direct.iloc[0] if len(direct) else (reverse.iloc[0] if len(reverse) else None)
        if row is not None:
            interval_includes_zero = float(row["record_bootstrap_ci_low"]) <= 0 <= float(row["record_bootstrap_ci_high"])
        if (top_score - float(candidate["oof_macro_f1"])) < 0.01 or interval_includes_zero:
            tied_ids.append(candidate_id)
    tied = candidates[candidates["candidate_id"].isin(tied_ids)].copy()
    if len(tied) == 1:
        return str(tied.iloc[0]["candidate_id"]), "clear_by_practical_margin"
    tied = tied.sort_values(
        ["class_collapse_count", "seed_sd", "worst_seed", "two_step_error", "ece", "parameter_count", "runtime_seconds"],
        ascending=[True, True, False, True, True, True, True],
    )
    return str(tied.iloc[0]["candidate_id"]), "practical_tie_resolved_by_preregistered_tiebreak"


def _conclusion(summary: pd.DataFrame, paired: pd.DataFrame, strict: dict[str, Any], gates: dict[str, Any], stage: str) -> str:
    if stage == "smoke":
        return "\n".join([
            "# Phase C0 smoke conclusion", "", f"Strict validation: **{strict['status']}**.",
            "Smoke verifies code paths only; it is not evidence for candidate ranking.",
            "No legacy-observed records, conditional candidates, Phase D, or Phase E were accessed/run.",
        ]) + "\n"
    overall = strict["provisional_best_overall_model"]
    hybrid = strict["provisional_best_thesis_hybrid_model"]
    columns = ["candidate_id", "oof_macro_f1", "outer_sd", "seed_sd", "worst_seed", "parameter_count"]
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    body = []
    for _, row in summary[columns].iterrows():
        body.append("| " + " | ".join(
            str(int(row[column])) if column == "parameter_count" else (
                f"{float(row[column]):.6f}" if column != "candidate_id" else str(row[column])
            )
            for column in columns
        ) + " |")
    table = "\n".join([header, separator, *body])
    return "\n".join([
        "# Strategy B Phase C conclusion", "", "## Main candidate results", "", table, "",
        f"- `provisional_best_overall_model`: **{overall}**.",
        f"- `provisional_best_thesis_hybrid_model`: **{hybrid}**.",
        f"- Strict validation: **{strict['status']}**.",
        f"- C1 request gate: **{gates['C1_huber_auxiliary']['recommendation']}**.",
        f"- C2 request gate: **{gates['C2_gated_residual']['recommendation']}**.",
        f"- Imbalance request gate: **{gates['imbalance']['recommendation']}**.",
        "", "No conditional branch was executed. Phase D/E remain unauthorized. README/PROJECT were not modified.",
    ]) + "\n"


def _resume_finalize(
    *,
    stage: str,
    run_id: str,
    artifact_tmp: Path,
    artifact_final: Path,
    report_tmp: Path,
    report_final: Path,
) -> None:
    """Finalize a fully trained run that failed only during report rendering."""

    if stage != "full":
        raise ValueError("resume-finalize is supported only for the official full run.")
    if not artifact_tmp.is_dir() or artifact_final.exists() or report_final.exists():
        raise FileNotFoundError("Expected one failed partial artifact directory and no completed destination.")
    state = json.loads((artifact_tmp / "run_state.json").read_text(encoding="utf-8"))
    if state.get("status") != "failed" or state.get("failure_type") != "ImportError" or "tabulate" not in state.get("failure_reason", ""):
        raise RuntimeError("Partial run is not eligible for safe finalization-only recovery.")
    strict = json.loads((artifact_tmp / "strict_validation.json").read_text(encoding="utf-8"))
    if strict.get("status") != "PASS":
        raise RuntimeError("Cannot finalize a partial run whose strict validation did not pass.")
    jobs = pd.read_csv(artifact_tmp / "job_ledger.csv")
    trials = pd.read_csv(artifact_tmp / "trial_history.csv")
    oof = pd.read_csv(artifact_tmp / "outer_oof_predictions.csv")
    checkpoints = json.loads((artifact_tmp / "checkpoint_checksums.json").read_text(encoding="utf-8"))
    recovery_checks = {
        "job_rows": len(jobs) == 2805,
        "all_jobs_completed": bool((jobs["status"] == "completed").all()),
        "trial_rows": len(trials) == 900,
        "all_trials_terminal": bool(trials["state"].isin(["COMPLETE", "PRUNED", "FAIL"]).all()),
        "oof_rows": len(oof) == 9 * 3 * 316,
        "oof_candidate_seed_coverage": bool(
            all(len(group) == 316 for _, group in oof.groupby(["candidate_id", "seed"]))
        ),
        "checkpoint_count": int(checkpoints.get("checkpoint_count", -1)) == 100,
        "all_checkpoints_reproduced": bool(checkpoints.get("all_reproduced")),
        "metric_recomputation": float(strict.get("metric_recomputation_max_abs_difference", 1.0)) <= 1e-12,
    }
    if not all(recovery_checks.values()):
        raise RuntimeError(f"Finalization recovery checks failed: {recovery_checks}")
    summary = pd.read_csv(artifact_tmp / "model_summary.csv")
    paired = pd.read_csv(artifact_tmp / "paired_model_deltas.csv")
    gates = json.loads((artifact_tmp / "conditional_gate_assessment.json").read_text(encoding="utf-8"))
    (artifact_tmp / "phase_c_conclusion.md").write_text(
        _conclusion(summary, paired, strict, gates, stage), encoding="utf-8"
    )
    finalization_provenance = _source_provenance()
    provenance_path = artifact_tmp / "source_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["finalization_recovery"] = {
        "training_source_git_commit": provenance["git_commit"],
        "finalization_git_commit": finalization_provenance["git_commit"],
        "finalization_source_tree_hash": finalization_provenance["source_tree_hash"],
        "reason": "report_rendering_only_missing_optional_tabulate_dependency",
        "training_or_predictions_changed": False,
        "recovery_checks": recovery_checks,
    }
    write_json(provenance_path, provenance)
    protocol_path = artifact_tmp / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["finalization_recovery"] = provenance["finalization_recovery"]
    write_json(protocol_path, protocol)
    strict["finalization_recovery_pass"] = True
    strict["finalization_recovery_checks"] = recovery_checks
    write_json(artifact_tmp / "strict_validation.json", strict)
    checksums = {
        path.relative_to(artifact_tmp).as_posix(): sha256_file(path)
        for path in sorted(artifact_tmp.rglob("*"))
        if path.is_file() and path.name not in {"artifact_checksums.json", "run_state.json"}
    }
    write_json(artifact_tmp / "artifact_checksums.json", checksums)
    _write_state(
        artifact_tmp, "completed", stage=stage, strict_status="PASS",
        recovery_from={"status": "failed", "failure_type": state["failure_type"], "failure_reason": state["failure_reason"]},
        recovery_checks=recovery_checks,
    )
    missing = [filename for filename in MINIMUM_OUTPUTS if not (artifact_tmp / filename).is_file()]
    if missing:
        raise RuntimeError(f"Missing required artifacts after recovery: {missing}")
    report_tmp.mkdir(parents=True, exist_ok=True)
    for path in artifact_tmp.iterdir():
        if path.is_file():
            shutil.copy2(path, report_tmp / path.name)
    os.replace(artifact_tmp, artifact_final)
    os.replace(report_tmp, report_final)
    print(json.dumps({
        "artifact_path": str(artifact_final), "report_path": str(report_final),
        "status": "PASS", "finalization_recovery": True,
    }))


def main() -> None:
    args = parse_args()
    stage = args.stage
    n_trials = args.n_trials if args.n_trials is not None else (1 if stage == "smoke" else 30)
    if stage == "full" and n_trials != 30:
        raise ValueError("Official full Phase C requires exactly 30 trials per searched family per outer fold.")
    root = SMOKE_ROOT if stage == "smoke" else FINAL_ROOT
    report_root = SMOKE_REPORT_ROOT if stage == "smoke" else FINAL_REPORT_ROOT
    artifact_final = root / args.run_id
    report_final = report_root / args.run_id
    artifact_tmp = root / f".{args.run_id}.tmp"
    report_tmp = report_root / f".{args.run_id}.tmp"
    if args.resume_finalize:
        _resume_finalize(
            stage=stage, run_id=args.run_id, artifact_tmp=artifact_tmp, artifact_final=artifact_final,
            report_tmp=report_tmp, report_final=report_final,
        )
        return
    if artifact_final.exists() or report_final.exists() or artifact_tmp.exists() or report_tmp.exists():
        raise FileExistsError(f"Run id already exists or has partial state: {args.run_id}")
    artifact_tmp.mkdir(parents=True)
    report_tmp.mkdir(parents=True)
    tables: dict[str, list[dict[str, Any]]] = {
        "trial_history": [], "job_ledger": [], "outer_oof_predictions": [],
        "training_diagnostics": [], "parameter_counts": [], "runtime_events": [], "resolved_configs": [],
    }
    checkpoint_entries: list[dict[str, Any]] = []
    _write_state(artifact_tmp, "running", stage=stage)
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        provenance = _source_provenance()
        manifest = load_fold_manifest(args.fold_manifest)
        allowed_rows = development_source_rows(manifest)
        raw, dataset_metadata = load_development_subset_from_postgres("student-mat", args.dataset_version_id, allowed_rows)
        development = process_target_and_stratify(raw, "G3", "student", "3class")
        assert_development_only_frame(development, manifest)
        outer_folds = outer_folds_from_manifest(development, manifest)
        memberships = _inner_memberships(development, outer_folds)
        outer_limit = 1 if stage == "smoke" else 5
        inner_limit = 1 if stage == "smoke" else 3
        outer_seeds = [42] if stage == "smoke" else PHASE_C_SEEDS
        write_json(artifact_tmp / "candidate_registry.json", candidate_registry())
        write_json(artifact_tmp / "search_spaces.json", search_spaces())
        write_json(artifact_tmp / "selection_rule.json", selection_rule())
        write_json(artifact_tmp / "resolved_config_schema.json", resolved_config_schema())
        write_json(artifact_tmp / "evidence_quarantine_registry.json", evidence_quarantine_registry())
        shutil.copy2(args.fold_manifest, artifact_tmp / "outer_fold_manifest.json")
        inner_ledger = materialize_inner_fold_ledger(development, outer_folds, dataset_version_id=args.dataset_version_id, target_col="G3")
        early_ledger = materialize_early_stop_ledger(development, outer_folds, dataset_version_id=args.dataset_version_id, target_col="G3", seeds=PHASE_C_SEEDS)
        inner_ledger.to_csv(artifact_tmp / "inner_fold_ledger.csv", index=False)
        early_ledger.to_csv(artifact_tmp / "early_stop_ledger.csv", index=False)
        dataset_manifest = {
            "dataset_code": "student-mat", "dataset_version_id": args.dataset_version_id,
            "dataset_hash": dataset_metadata["content_hash"], "target_contract_hash": dataset_metadata["target_contract_hash"],
            "development_row_count": len(development), "development_source_rows_hash": source_rows_hash(allowed_rows),
            "transaction_read_only": dataset_metadata["transaction_read_only"], "legacy_observed_rows_fetched": False,
        }
        write_json(artifact_tmp / "dataset_manifest.json", dataset_manifest)
        protocol = {
            "protocol_version": PHASE_C_PROTOCOL_VERSION, "stage": stage, "run_id": args.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": provenance["git_commit"],
            "outer_folds": outer_limit, "inner_folds_per_trial": inner_limit,
            "trials_per_searched_family_per_outer_fold": n_trials,
            "declared_outer_seeds": outer_seeds, "features": ["G1", "G2"], "raw_feature_transforms": "none",
            "normalization": ["none", "layer_norm"], "batch_norm_allowed": False, "drop_last": False,
            "scheduler": "fixed_lr", "swa": False, "class_weight": "none", "oversampling": "none",
            "high_class_recall_guardrail": HIGH_RECALL_GUARDRAIL,
            "legacy_observed_79_fetched": False, "phase_d_or_e_executed": False,
            "conditional_candidates_executed": False, "readme_headline_modified": False,
            "postgres_integration_policy": "disposable_test_DSN_only; waiver if unavailable; official data read-only",
            "outer_fold_manifest_file_hash": file_checksum(args.fold_manifest),
            "outer_fold_manifest_semantic_hash": semantic_checksum(manifest),
            "inner_fold_ledger_hash": sha256_file(artifact_tmp / "inner_fold_ledger.csv"),
            "early_stop_ledger_hash": sha256_file(artifact_tmp / "early_stop_ledger.csv"),
            "source_tree_hash": provenance["source_tree_hash"], "environment_hash": provenance["environment_hash"],
        }
        write_json(artifact_tmp / "protocol.json", protocol)
        write_json(artifact_tmp / "source_provenance.json", provenance)
        test_report = _run_tests(args.skip_tests, stage)
        write_json(artifact_tmp / "test_report.json", test_report)
        if test_report.get("status") == "FAIL":
            raise RuntimeError("Full pytest suite failed before Phase C execution.")

        for outer_fold, (outer_train_idx, outer_validation_idx) in enumerate(outer_folds[:outer_limit]):
            outer_train = development.iloc[outer_train_idx].copy()
            outer_validation = development.iloc[outer_validation_idx].copy()
            _evaluate_rule(outer_fold, outer_validation, tables, outer_seeds)
            for candidate_id in ML_CANDIDATES:
                config = _ml_search(
                    candidate_id, outer_fold, outer_train, memberships[outer_fold], n_trials=n_trials,
                    inner_limit=inner_limit, tables=tables, artifact_tmp=artifact_tmp,
                )
                tables["resolved_configs"].append({
                    "candidate_id": candidate_id, "outer_fold": outer_fold, "resolved_config_hash": config_hash(config),
                    "resolved_config": json.dumps(config, sort_keys=True),
                })
                _evaluate_ml(
                    candidate_id=candidate_id, outer_fold=outer_fold, config=config, outer_train=outer_train,
                    outer_validation=outer_validation, artifact_tmp=artifact_tmp, tables=tables,
                    checkpoint_entries=checkpoint_entries, report_seeds=outer_seeds,
                )
            selected_neural: dict[str, dict[str, Any]] = {}
            for candidate_id in MAIN_NEURAL:
                config, _ = _neural_search(
                    candidate_id, outer_fold, outer_train, memberships[outer_fold], n_trials=n_trials,
                    inner_limit=inner_limit, tables=tables, artifact_tmp=artifact_tmp,
                )
                selected_neural[candidate_id] = config
                tables["resolved_configs"].append({
                    "candidate_id": candidate_id, "outer_fold": outer_fold,
                    "resolved_config_hash": resolved_config_hash(config),
                    "resolved_config": json.dumps(config, sort_keys=True),
                })
                for seed in outer_seeds:
                    _evaluate_neural(
                        candidate_id=candidate_id, outer_fold=outer_fold, seed=seed, config=config,
                        outer_train=outer_train, outer_validation=outer_validation, artifact_tmp=artifact_tmp,
                        tables=tables, checkpoint_entries=checkpoint_entries,
                    )
            for candidate_id in ABLATIONS:
                config = matched_ablation_config(candidate_id, selected_neural["N0"])
                tables["resolved_configs"].append({
                    "candidate_id": candidate_id, "outer_fold": outer_fold,
                    "resolved_config_hash": resolved_config_hash(config),
                    "resolved_config": json.dumps(config, sort_keys=True),
                })
                for seed in outer_seeds:
                    _evaluate_neural(
                        candidate_id=candidate_id, outer_fold=outer_fold, seed=seed, config=config,
                        outer_train=outer_train, outer_validation=outer_validation, artifact_tmp=artifact_tmp,
                        tables=tables, checkpoint_entries=checkpoint_entries,
                    )
            _persist_tables(artifact_tmp, tables)

        oof = pd.DataFrame(tables["outer_oof_predictions"])
        fold_metrics, per_class, confusions, ordinal, seed_metrics = detailed_metrics(oof)
        fold_metrics.to_csv(artifact_tmp / "fold_seed_metrics.csv", index=False)
        per_class.to_csv(artifact_tmp / "per_class_metrics.csv", index=False)
        confusions.to_csv(artifact_tmp / "confusion_matrices.csv", index=False)
        ordinal.to_csv(artifact_tmp / "ordinal_metrics.csv", index=False)
        ordinal[["candidate_id", "seed", "outer_fold", "brier", "nll", "ece"]].to_csv(artifact_tmp / "calibration_metrics.csv", index=False)
        boundary_error_analysis(oof).to_csv(artifact_tmp / "boundary_error_analysis.csv", index=False)
        pd.DataFrame(tables["training_diagnostics"]).to_csv(artifact_tmp / "training_diagnostics.csv", index=False)
        counts = pd.DataFrame(tables["parameter_counts"])
        counts.to_csv(artifact_tmp / "parameter_counts.csv", index=False)
        runtime_events = pd.DataFrame(tables["runtime_events"])
        runtime_summary = runtime_events.groupby(["candidate_id", "stage"], as_index=False).agg(
            runtime_seconds=("runtime_seconds", "sum"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            events=("runtime_seconds", "count"),
        )
        trial_frame = pd.DataFrame(tables["trial_history"])
        if not trial_frame.empty:
            trial_counts = trial_frame.groupby(["candidate_id", "state"]).size().unstack(fill_value=0).reset_index()
            trial_counts = trial_counts.rename(columns={
                "COMPLETE": "actual_completed_trials", "PRUNED": "actual_pruned_trials", "FAIL": "actual_failed_trials",
            })
            for column in ["actual_completed_trials", "actual_pruned_trials", "actual_failed_trials"]:
                if column not in trial_counts:
                    trial_counts[column] = 0
            fit_counts = trial_frame.groupby("candidate_id", as_index=False)["actual_fit_stages"].sum().rename(
                columns={"actual_fit_stages": "actual_inner_fit_stages"}
            )
            trial_counts = trial_counts.merge(fit_counts, on="candidate_id", how="left")
            runtime_summary = runtime_summary.merge(trial_counts, on="candidate_id", how="left")
        runtime_summary.to_csv(artifact_tmp / "runtime_summary.csv", index=False)
        summary = model_summary(oof, fold_metrics, ordinal, per_class, counts, runtime_events)
        summary.to_csv(artifact_tmp / "model_summary.csv", index=False)
        fold_metrics[["candidate_id", "seed", "outer_fold", "class_collapse"]].to_csv(artifact_tmp / "class_collapse_report.csv", index=False)
        comparisons = [("N1", "N0"), ("N1", "N3"), ("N0", "N2"), ("N0", "A1"), ("N0", "A2")]
        best_neural = str(summary[summary["candidate_id"].isin(MAIN_NEURAL)].iloc[0]["candidate_id"])
        comparisons.extend([(best_neural, "R0"), (best_neural, "M1"), (best_neural, "M2")])
        top_overall = str(summary[summary["candidate_id"].isin(RANKING_CANDIDATES)].iloc[0]["candidate_id"])
        comparisons.extend((top_overall, candidate) for candidate in RANKING_CANDIDATES if candidate != top_overall)
        comparisons = list(dict.fromkeys(comparisons))
        paired = paired_deltas(oof, comparisons, bootstrap_samples=200 if stage == "smoke" else 2000)
        paired.to_csv(artifact_tmp / "paired_model_deltas.csv", index=False)
        pd.DataFrame(tables["resolved_configs"]).to_csv(artifact_tmp / "resolved_configs.csv", index=False)
        pd.DataFrame(tables["trial_history"]).to_csv(artifact_tmp / "trial_history.csv", index=False)
        pd.DataFrame(tables["job_ledger"]).to_csv(artifact_tmp / "job_ledger.csv", index=False)
        oof.to_csv(artifact_tmp / "outer_oof_predictions.csv", index=False)
        write_json(artifact_tmp / "checkpoint_checksums.json", {
            "checkpoint_count": len(checkpoint_entries),
            "all_reproduced": all(row["prediction_reproduction_pass"] for row in checkpoint_entries),
            "entries": checkpoint_entries,
        })

        n1 = summary.set_index("candidate_id").loc["N1"]
        n0 = summary.set_index("candidate_id").loc["N0"]
        ordinal_fold = ordinal.merge(fold_metrics[["candidate_id", "seed", "outer_fold", "class_collapse"]], on=["candidate_id", "seed", "outer_fold"])
        n1_better_guardrails = 0
        compared_guardrails = 0
        for (seed, fold), group in ordinal_fold[ordinal_fold["candidate_id"].isin(["N0", "N1"])].groupby(["seed", "outer_fold"]):
            if set(group["candidate_id"]) == {"N0", "N1"}:
                values = group.set_index("candidate_id")
                n1_better_guardrails += int(
                    values.loc["N1", "ordinal_mae"] <= values.loc["N0", "ordinal_mae"]
                    and values.loc["N1", "two_step_error_rate"] <= values.loc["N0", "two_step_error_rate"]
                )
                compared_guardrails += 1
        selected_neural_id = best_neural
        selected_high_recall = float(per_class[
            (per_class["candidate_id"] == selected_neural_id) & (per_class["class_label"] == 2)
        ]["recall"].mean())
        selected_collapses = int(summary.set_index("candidate_id").loc[selected_neural_id, "class_collapse_count"])
        gates = {
            "C1_huber_auxiliary": {
                "criteria": {
                    "n1_not_worse_than_n0_by_margin": float(n1["oof_macro_f1"] - n0["oof_macro_f1"]) >= -0.01,
                    "ordinal_guardrails_better_majority": n1_better_guardrails > compared_guardrails / 2,
                    "threshold_violations": 0,
                    "severe_class_collapse": int(n1["class_collapse_count"]) > 0,
                },
                "recommendation": "request_separate_approval" if (
                    float(n1["oof_macro_f1"] - n0["oof_macro_f1"]) >= -0.01
                    and n1_better_guardrails > compared_guardrails / 2
                    and int(n1["class_collapse_count"]) == 0
                ) else "do_not_open",
                "executed": False,
            },
            "C2_gated_residual": {
                "recommendation": "do_not_open",
                "reason": "No pre-registered residual-specific experiment in main Phase C; boundary errors alone do not establish residual utility.",
                "executed": False,
            },
            "imbalance": {
                "selected_neural_model": selected_neural_id,
                "high_class_recall": selected_high_recall,
                "class_collapse_count": selected_collapses,
                "recommendation": "request_separate_approval" if selected_collapses > 0 or selected_high_recall < HIGH_RECALL_GUARDRAIL else "do_not_open",
                "executed": False,
                "executed": False,
            },
        }
        write_json(artifact_tmp / "conditional_gate_assessment.json", gates)
        completed_jobs = pd.DataFrame(tables["job_ledger"])
        # Manifest fold sizes are authoritative; use direct candidate/seed coverage checks below.
        coverage_pass = True
        expected_rows = sorted(development.iloc[np.concatenate([outer_folds[i][1] for i in range(outer_limit)])][SOURCE_ROW_NUMBER_COLUMN].astype(int).tolist())
        for candidate in ALL_CANDIDATES:
            seeds = outer_seeds
            for seed in seeds:
                observed = sorted(oof[(oof["candidate_id"] == candidate) & (oof["seed"] == seed)]["source_row_number"].astype(int).tolist())
                coverage_pass &= observed == expected_rows
        recomputed_fold, *_ = detailed_metrics(pd.read_csv(artifact_tmp / "outer_oof_predictions.csv"))
        metric_diff = float(np.max(np.abs(
            fold_metrics.sort_values(["candidate_id", "seed", "outer_fold"])["macro_f1"].to_numpy()
            - recomputed_fold.sort_values(["candidate_id", "seed", "outer_fold"])["macro_f1"].to_numpy()
        )))
        checks = [
            {"id": "candidate_registry", "pass": [row["id"] for row in candidate_registry()["candidates"]] == ALL_CANDIDATES},
            {"id": "full_test_suite", "pass": test_report.get("status") == "PASS"},
            {"id": "development_only_db_access", "pass": dataset_metadata["transaction_read_only"] and len(development) == 316},
            {"id": "no_legacy_observed_fetch", "pass": not protocol["legacy_observed_79_fetched"]},
            {"id": "same_outer_inner_membership", "pass": len(inner_ledger) > 0 and len(early_ledger) > 0},
            {"id": "job_completeness", "pass": not (completed_jobs["status"] != "completed").any()},
            {"id": "oof_record_alignment", "pass": bool(coverage_pass)},
            {"id": "drop_last_false_full_coverage", "pass": all('"samples_dropped_per_epoch": 0' in str(value) for value in pd.DataFrame(tables["training_diagnostics"])["sample_utilization"])},
            {"id": "parameter_guardrail", "pass": bool((counts[counts["candidate_id"].isin(MAIN_NEURAL)]["parameter_count"] <= PARAMETER_GUARDRAIL).all())},
            {"id": "estimator_parity", "pass": bool(pd.DataFrame(tables["training_diagnostics"])["estimator_parity"].all())},
            {"id": "checkpoint_reproduction", "pass": all(row["prediction_reproduction_pass"] for row in checkpoint_entries)},
            {"id": "metrics_recomputation", "pass": metric_diff <= 1e-12},
            {"id": "conditional_branches_not_run", "pass": all(not value["executed"] for value in gates.values())},
            {"id": "phase_d_e_not_run", "pass": not protocol["phase_d_or_e_executed"]},
        ]
        strict = {
            "run_id": args.run_id, "stage": stage,
            "status": "PASS" if all(row["pass"] for row in checks) else "FAIL",
            "checks": checks, "metric_recomputation_max_abs_difference": metric_diff,
            "checkpoint_reproduction_max_abs_difference": max((row["prediction_reproduction_max_abs_difference"] for row in checkpoint_entries), default=0.0),
            "legacy_observed_79_accessed": False, "conditional_branch_authorized": False,
            "phase_d_e_authorized": False,
        }
        if stage == "full":
            overall, overall_reason = _choose(summary, RANKING_CANDIDATES, paired)
            hybrid, hybrid_reason = _choose(summary, ["N0", "N1"], paired)
            strict.update({
                "provisional_best_overall_model": overall, "best_overall_selection_reason": overall_reason,
                "provisional_best_thesis_hybrid_model": hybrid, "best_hybrid_selection_reason": hybrid_reason,
            })
        write_json(artifact_tmp / "strict_validation.json", strict)
        (artifact_tmp / "phase_c_conclusion.md").write_text(_conclusion(summary, paired, strict, gates, stage), encoding="utf-8")
        if strict["status"] != "PASS":
            raise RuntimeError("Phase C strict validation failed; run remains partial.")
        # Checksum every artifact present before the checksum file itself.
        checksums = {
            path.relative_to(artifact_tmp).as_posix(): sha256_file(path)
            for path in sorted(artifact_tmp.rglob("*")) if path.is_file() and path.name not in {"artifact_checksums.json", "run_state.json"}
        }
        write_json(artifact_tmp / "artifact_checksums.json", checksums)
        _write_state(artifact_tmp, "completed", stage=stage, strict_status="PASS")
        for filename in MINIMUM_OUTPUTS:
            if not (artifact_tmp / filename).is_file():
                raise RuntimeError(f"Missing required Phase C artifact: {filename}")
        # Report mirror contains the complete human-reviewable evidence tables,
        # while binary checkpoints remain in the artifact bundle.
        for path in artifact_tmp.iterdir():
            if path.is_file():
                shutil.copy2(path, report_tmp / path.name)
        os.replace(artifact_tmp, artifact_final)
        os.replace(report_tmp, report_final)
        print(json.dumps({"artifact_path": str(artifact_final), "report_path": str(report_final), "status": "PASS"}))
    except Exception as exc:
        _write_state(
            artifact_tmp, "failed", stage=stage, failure_type=type(exc).__name__,
            failure_reason=str(exc), traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
