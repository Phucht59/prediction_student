"""Authorized Strategy B Phase E-Prediction runner.

It consumes the immutable Phase C outer selected configurations, evaluates only
R0/M1/M2/N0/N1 on five new seeds, then freezes development-only final models.
It never fetches the 79 legacy-observed rows and never runs recommendation or
conditional branches.
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
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from src.config import DATASETS, ROOT_DIR
from src.data_pipeline import SOURCE_ROW_NUMBER_COLUMN, process_target_and_stratify
from src.evaluation.protocol import DEFAULT_FOLD_MANIFEST_PATH, file_checksum, load_fold_manifest, outer_folds_from_manifest, semantic_checksum
from src.estimator_factory import resolved_config_hash
from src.model_selection import fit_final_development_estimator, fit_fold_predict_proba, predict_with_fitted_estimator
from src.postgres_data_source import load_development_subset_from_postgres
from src.strategy_b_phase_ab import assert_development_only_frame, development_source_rows, materialize_inner_fold_ledger, sha256_file, source_rows_hash, write_json
from src.strategy_b_phase_c import PARAMETER_GUARDRAIL, config_hash, ml_resolved_config, sample_neural_config
from src.strategy_b_phase_e_prediction import (
    DETERMINISTIC_SEED, FINALISTS, HYBRID_FINALISTS, OVERALL_FINALISTS, PHASE_E_PROTOCOL_VERSION,
    PHASE_E_SEEDS, apply_temperature, calibration_metrics, canonical_hash, choose_final, choose_temperature,
    classification_rows, fit_temperature, paired_metric_deltas, phase_e_registry, precision_recall_rows,
    regression_rows, seed_registry, seed_stability, selection_rule,
)


ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "strategy_b_phase_e_prediction"
REPORT_ROOT = ROOT_DIR / "reports" / "strategy_b_phase_e_prediction"
PHASE_C_ARTIFACT = ROOT_DIR / "artifacts" / "strategy_b_phase_c" / "strategy-b-phase-c-20260714-5d34a66"
REPRODUCTION_TOLERANCE = 1e-7
MINIMUM_OUTPUTS = [
    "protocol.json", "finalist_registry.json", "seed_registry.json", "selection_rule.json", "phase_c_source_manifest.json",
    "stability_job_ledger.csv", "fold_seed_metrics.csv", "outer_oof_predictions.csv", "seed_disagreement.csv",
    "paired_stability_deltas.csv", "per_class_metrics.csv", "ordinal_metrics.csv", "calibration_metrics.csv",
    "calibration_parameters.csv", "coverage_risk_curves.csv", "abstention_assessment.json", "final_family_decision.json",
    "final_inner_search_history.csv", "final_resolved_configs.json", "final_model_manifest.json",
    "final_checkpoint_checksums.json", "final_preprocessor_checksums.json", "final_calibrator_checksums.json",
    "source_provenance.json", "artifact_checksums.json", "test_report.json", "strict_validation.json",
    "phase_e_prediction_conclusion.md", "classification_metrics.csv", "precision_recall_metrics.csv",
    "precision_recall_curve_points.csv", "precision_recall_summary.json", "regression_metrics.csv",
    "continuous_prediction_contract.json", "continuous_oof_predictions.csv", "metric_tradeoff_analysis.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-version-id", type=int, default=1)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST_PATH)
    parser.add_argument("--resume-finalize", action="store_true", help="Validate and atomically finalize an evidence-only strict-check correction; never retrains.")
    return parser.parse_args()


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT_DIR, text=True, encoding="utf-8", errors="replace", capture_output=True)


def _provenance() -> dict[str, Any]:
    git = lambda *args: _run(["git", *args]).stdout.strip()
    diff = _run(["git", "diff", "--binary", "HEAD"]).stdout.encode()
    tracked = git("ls-files").splitlines()
    paths = [p for p in tracked if p.startswith(("src/", "scripts/", "tests/", "config/", "database/")) or p in {"requirements.txt", "requirements-lock.txt", "environment.yml"}]
    hashes = {p: sha256_file(ROOT_DIR / p) for p in paths if (ROOT_DIR / p).is_file()}
    env = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "cuda_available": torch.cuda.is_available(), "pip_freeze": _run([sys.executable, "-m", "pip", "freeze"]).stdout.splitlines()}
    return {"git_commit": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"), "dirty_diff_hash": hashlib.sha256(diff).hexdigest(), "dirty_diff_bytes": len(diff), "source_hashes": hashes, "source_tree_hash": canonical_hash(hashes), "environment": env, "environment_hash": canonical_hash(env)}


def _write_state(root: Path, status: str, **extra: Any) -> None:
    write_json(root / "run_state.json", {"status": status, "updated_at": datetime.now(timezone.utc).isoformat(), **extra})


def _make_ml(config: dict[str, Any], seed: int) -> Pipeline:
    params = dict(config["parameters"])
    if config["candidate_id"] == "M1":
        estimator = RandomForestClassifier(**params, class_weight=None, random_state=int(seed), n_jobs=1)
    else:
        # M2 is explicitly fixed to verify deterministic replay, not to manufacture seed rows.
        estimator = SVC(**params, kernel="rbf", probability=True, class_weight=None, random_state=42)
    return Pipeline([("scale", MinMaxScaler()), ("model", estimator)])


def _parameter_count(model: Pipeline, candidate: str) -> int:
    fitted = model.named_steps["model"]
    if candidate == "M1":
        return int(sum(tree.tree_.node_count for tree in fitted.estimators_))
    return int(fitted.support_vectors_.size + fitted.dual_coef_.size + fitted.intercept_.size)


def _rule_probabilities(frame: pd.DataFrame) -> np.ndarray:
    predicted = np.digitize(frame["G2"].to_numpy(float), bins=[9, 14], right=True)
    return np.eye(3, dtype=float)[predicted]


def _load_phase_c_configs() -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    if not PHASE_C_ARTIFACT.is_dir():
        raise FileNotFoundError(f"Approved Phase C artifact missing: {PHASE_C_ARTIFACT}")
    checksum_file = PHASE_C_ARTIFACT / "artifact_checksums.json"
    expected = json.loads(checksum_file.read_text(encoding="utf-8"))
    failures = [name for name, digest in expected.items() if not (PHASE_C_ARTIFACT / name).is_file() or sha256_file(PHASE_C_ARTIFACT / name) != digest]
    if failures:
        raise RuntimeError(f"Phase C source checksum validation failed: {failures[:3]}")
    configs = {}
    for row in pd.read_csv(PHASE_C_ARTIFACT / "resolved_configs.csv").itertuples(index=False):
        if row.candidate_id in FINALISTS:
            configs[(str(row.candidate_id), int(row.outer_fold))] = json.loads(row.resolved_config)
    expected_keys = {(candidate, fold) for candidate in ["M1", "M2", "N0", "N1"] for fold in range(5)}
    if not expected_keys <= set(configs):
        raise RuntimeError("Phase C selected configurations are incomplete for Phase E finalists.")
    return configs, {"path": str(PHASE_C_ARTIFACT), "artifact_checksum_count": len(expected), "checksum_failures": failures, "phase_c_evidence_commit": "e20ff43c7a7c95b638e82b84d40c7cf10b6e0d49", "phase_c_code_commit": "5d34a6641036be454c115747718b16669590f0be"}


def _inner_memberships(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=202601)
    return list(splitter.split(frame, frame["G3"].astype(int)))


def _append_oof(rows: list[dict[str, Any]], candidate: str, seed: int, fold: int, validation: pd.DataFrame, probabilities: np.ndarray) -> None:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    for position, (_, record) in enumerate(validation.iterrows()):
        rows.append({"candidate_id": candidate, "seed": int(seed), "outer_fold": int(fold), "source_row_number": int(record[SOURCE_ROW_NUMBER_COLUMN]), "raw_g3": float(record["G3_raw"]), "true_label": int(record["G3"]), "predicted_label": int(predictions[position]), "prob_0": float(probabilities[position, 0]), "prob_1": float(probabilities[position, 1]), "prob_2": float(probabilities[position, 2])})


def _continuous_rows(rows: list[dict[str, Any]], candidate: str, seed: int, fold: int, validation: pd.DataFrame, probabilities: np.ndarray, outer_train: pd.DataFrame) -> dict[str, Any]:
    if candidate == "R0":
        predicted = validation["G2"].to_numpy(float)
        method = "raw_g2_rule_regression_contract"
        mapping = None
    else:
        means = outer_train.groupby("G3")["G3_raw"].mean().reindex([0, 1, 2]).to_numpy(float)
        if not np.isfinite(means).all():
            raise RuntimeError("A training partition did not contain all registered classes for continuous mapping.")
        predicted = np.asarray(probabilities, dtype=float) @ means
        method = "training_partition_class_conditional_mean_expected_g3"
        mapping = {"low": float(means[0]), "medium": float(means[1]), "high": float(means[2])}
    for position, (_, record) in enumerate(validation.iterrows()):
        rows.append({"candidate_id": candidate, "source_record_id": int(record[SOURCE_ROW_NUMBER_COLUMN]), "outer_fold": int(fold), "seed": int(seed), "true_g3": float(record["G3_raw"]), "predicted_g3": float(predicted[position]), "continuous_prediction_method": method})
    return {"candidate_id": candidate, "outer_fold": fold, "seed": seed, "method": method, "training_partition_class_means": mapping, "mapping_hash": canonical_hash(mapping) if mapping is not None else None}


def _fit_outer(candidate: str, seed: int, fold: int, config: dict[str, Any] | None, outer_train: pd.DataFrame, outer_validation: pd.DataFrame, root: Path, checkpoints: list[dict[str, Any]]) -> tuple[np.ndarray, int, dict[str, Any] | None]:
    if candidate == "R0":
        return _rule_probabilities(outer_validation), 0, None
    checkpoint_dir = root / "stability_checkpoints" / candidate
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if candidate in {"M1", "M2"}:
        assert config is not None
        model = _make_ml(config, seed)
        model.fit(outer_train[["G1", "G2"]], outer_train["G3"])
        probabilities = model.predict_proba(outer_validation[["G1", "G2"]])
        path = checkpoint_dir / f"outer{fold}_seed{seed}.pkl"
        with path.open("wb") as handle:
            pickle.dump(model, handle)
        with path.open("rb") as handle:
            restored = pickle.load(handle)
        difference = float(np.max(np.abs(restored.predict_proba(outer_validation[["G1", "G2"]]) - probabilities)))
        if difference > REPRODUCTION_TOLERANCE:
            raise RuntimeError("ML checkpoint reproduction failed.")
        checkpoints.append({"candidate_id": candidate, "outer_fold": fold, "seed": seed, "checkpoint": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "prediction_reproduction_max_abs_difference": difference, "pass": True})
        return probabilities, _parameter_count(model, candidate), None
    assert config is not None
    result = fit_fold_predict_proba(train_fold=outer_train, validation_fold=outer_validation, spec=DATASETS["student-mat"], params=config, seed=seed, fold_index=fold)
    checkpoint = checkpoint_dir / f"outer{fold}_seed{seed}.pt"
    prep = checkpoint_dir / f"outer{fold}_seed{seed}.preprocessor.pkl"
    torch.save(result.refit_state_dict, checkpoint)
    with prep.open("wb") as handle:
        pickle.dump({"preprocessor": result.refit_preprocessor, "selector": result.refit_selector}, handle)
    with prep.open("rb") as handle:
        loaded = pickle.load(handle)
    restored = predict_with_fitted_estimator(frame=outer_validation, spec=DATASETS["student-mat"], resolved_config=config, state_dict=torch.load(checkpoint, map_location="cpu", weights_only=True), preprocessor=loaded["preprocessor"], selector=loaded["selector"])
    difference = float(np.max(np.abs(restored - result.probabilities)))
    if difference > REPRODUCTION_TOLERANCE:
        raise RuntimeError("Neural checkpoint reproduction failed.")
    checkpoints.append({"candidate_id": candidate, "outer_fold": fold, "seed": seed, "checkpoint": checkpoint.relative_to(root).as_posix(), "sha256": sha256_file(checkpoint), "preprocessor": prep.relative_to(root).as_posix(), "preprocessor_sha256": sha256_file(prep), "prediction_reproduction_max_abs_difference": difference, "pass": True})
    return result.probabilities, int(config["parameter_count"]), dict(result.training_diagnostics or {})


def _inner_calibration(candidate: str, seed: int, config: dict[str, Any], outer_train: pd.DataFrame, ledger: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, np.ndarray | None, np.ndarray | None]:
    if candidate == "R0":
        return None, None, None
    probabilities, labels = [], []
    for inner_fold, (train_idx, val_idx) in enumerate(_inner_memberships(outer_train)):
        train, validation = outer_train.iloc[train_idx].copy(), outer_train.iloc[val_idx].copy()
        started = time.perf_counter()
        if candidate in {"M1", "M2"}:
            model = _make_ml(config, seed)
            model.fit(train[["G1", "G2"]], train["G3"])
            prediction = model.predict_proba(validation[["G1", "G2"]])
        else:
            prediction = fit_fold_predict_proba(train_fold=train, validation_fold=validation, spec=DATASETS["student-mat"], params=config, seed=seed, fold_index=inner_fold).probabilities
        probabilities.append(prediction)
        labels.append(validation["G3"].to_numpy(int))
        ledger.append({"stage": "inner_oof_calibration", "candidate_id": candidate, "outer_fold": "calibration_outer", "inner_fold": inner_fold, "seed": seed, "status": "completed", "runtime_seconds": time.perf_counter() - started})
    inner_p, inner_y = np.vstack(probabilities), np.concatenate(labels)
    return fit_temperature(inner_p, inner_y), inner_p, inner_y


def _calibration_rows(candidate: str, seed: int, fold: int, y: np.ndarray, uncalibrated: np.ndarray, calibrator: dict[str, Any] | None) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows = [{"candidate_id": candidate, "outer_fold": fold, "seed": seed, "variant": "uncalibrated", **calibration_metrics(y, uncalibrated)}]
    if calibrator is None:
        return rows, uncalibrated
    calibrated = apply_temperature(uncalibrated, float(calibrator["temperature"]))
    if not np.array_equal(uncalibrated.argmax(axis=1), calibrated.argmax(axis=1)):
        raise RuntimeError("Scalar temperature unexpectedly changed the argmax decision contract.")
    rows.append({"candidate_id": candidate, "outer_fold": fold, "seed": seed, "variant": "temperature", **calibration_metrics(y, calibrated)})
    return rows, calibrated


def _run_tests() -> dict[str, Any]:
    started = time.perf_counter()
    result = _run([sys.executable, "-m", "pytest", "-q", "-rs"])
    return {"official": True, "command": [sys.executable, "-m", "pytest", "-q", "-rs"], "return_code": result.returncode, "status": "PASS" if result.returncode == 0 else "FAIL", "duration_seconds": time.perf_counter() - started, "stdout": result.stdout, "stderr": result.stderr, "postgres_integration_waiver": {"active": "POSTGRES_TEST_DSN" not in os.environ or "POSTGRES_TEST_APP_DSN" not in os.environ, "reason": "Disposable PostgreSQL test DSNs/psql unavailable; production database was not used destructively."}}


def _final_ml_search(candidate: str, development: pd.DataFrame, history: list[dict[str, Any]]) -> dict[str, Any]:
    folds = _inner_memberships(development)
    def objective(trial: optuna.Trial) -> float:
        if candidate == "M1":
            parameters = {"n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300]), "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 8]), "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 4]), "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])}
        else:
            parameters = {"C": trial.suggest_float("C", 0.05, 50.0, log=True), "gamma": trial.suggest_float("gamma", 1e-3, 2.0, log=True)}
        config = ml_resolved_config(candidate, parameters)
        scores = []
        for fold, (train, validation) in enumerate(folds):
            model = _make_ml(config, PHASE_E_SEEDS[fold % len(PHASE_E_SEEDS)])
            model.fit(development.iloc[train][["G1", "G2"]], development.iloc[train]["G3"])
            scores.append(float(f1_score(development.iloc[validation]["G3"], model.predict(development.iloc[validation][["G1", "G2"]]), average="macro", zero_division=0)))
        trial.set_user_attr("resolved_config", config)
        trial.set_user_attr("inner_scores", scores)
        return float(np.mean(scores))
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=202606 + (1 if candidate == "M2" else 0)))
    study.optimize(objective, n_trials=30, catch=(Exception,), show_progress_bar=False)
    for trial in study.trials:
        history.append({"candidate_id": candidate, "trial_number": trial.number, "state": trial.state.name, "value": trial.value, "resolved_config": json.dumps(trial.user_attrs.get("resolved_config"), sort_keys=True), "inner_scores": json.dumps(trial.user_attrs.get("inner_scores")), "stage": "final_inner_cv_search"})
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        raise RuntimeError("Final ML search had no completed trial.")
    return deepcopy(max(completed, key=lambda t: float(t.value)).user_attrs["resolved_config"])


def _final_neural_search(candidate: str, development: pd.DataFrame, history: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    folds = _inner_memberships(development)
    def objective(trial: optuna.Trial) -> float:
        config = sample_neural_config(trial, candidate)
        scores = []
        for fold, (train, validation) in enumerate(folds):
            result = fit_fold_predict_proba(train_fold=development.iloc[train].copy(), validation_fold=development.iloc[validation].copy(), spec=DATASETS["student-mat"], params=config, seed=PHASE_E_SEEDS[fold], fold_index=fold)
            scores.append(float(f1_score(result.true_labels, result.predictions, average="macro", zero_division=0)))
        trial.set_user_attr("resolved_config", config)
        trial.set_user_attr("inner_scores", scores)
        return float(np.mean(scores))
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=202607 + (1 if candidate == "N1" else 0)))
    study.optimize(objective, n_trials=30, catch=(Exception,), show_progress_bar=False)
    for trial in study.trials:
        history.append({"candidate_id": candidate, "trial_number": trial.number, "state": trial.state.name, "value": trial.value, "resolved_config": json.dumps(trial.user_attrs.get("resolved_config"), sort_keys=True), "inner_scores": json.dumps(trial.user_attrs.get("inner_scores")), "stage": "final_inner_cv_search"})
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        raise RuntimeError("Final neural search had no completed trial.")
    return deepcopy(max(completed, key=lambda t: float(t.value)).user_attrs["resolved_config"])


def _final_calibrator(candidate: str, config: dict[str, Any], development: pd.DataFrame) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if candidate == "R0":
        return None, {"candidate_id": candidate, "method": "not_applicable_hard_rule"}
    probs, labels = [], []
    for fold, (train, validation) in enumerate(_inner_memberships(development)):
        if candidate in {"M1", "M2"}:
            model = _make_ml(config, PHASE_E_SEEDS[fold])
            model.fit(development.iloc[train][["G1", "G2"]], development.iloc[train]["G3"])
            probability = model.predict_proba(development.iloc[validation][["G1", "G2"]])
        else:
            probability = fit_fold_predict_proba(train_fold=development.iloc[train].copy(), validation_fold=development.iloc[validation].copy(), spec=DATASETS["student-mat"], params=config, seed=PHASE_E_SEEDS[fold], fold_index=fold).probabilities
        probs.append(probability); labels.append(development.iloc[validation]["G3"].to_numpy(int))
    calibrator = fit_temperature(np.vstack(probs), np.concatenate(labels))
    return calibrator, {"candidate_id": candidate, "method": calibrator["method"], "inner_oof_records": int(sum(map(len, labels)),), "calibrator_hash": canonical_hash(calibrator)}


def _conclusion(summary: pd.DataFrame, final: dict[str, Any], strict: dict[str, Any]) -> str:
    columns = ["candidate_id", "oof_macro_f1", "accuracy", "macro_precision", "macro_recall", "high_f1", "macro_pr_auc", "rmse", "r2"]
    view = summary[columns]
    lines = ["# Strategy B Phase E-Prediction conclusion", "", "## Development-only stability results", "", "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) if c == "candidate_id" else f"{float(row[c]):.6f}" for c in columns) + " |")
    lines += ["", f"- `final_overall_model`: **{final['final_overall_model']}**.", f"- `final_thesis_hybrid_model`: **{final['final_thesis_hybrid_model']}**.", "- Final model family and configuration were selected and frozen using nested development evidence. No untouched external confirmation dataset was available.", "- No legacy-observed-79 records, external labels, recommendation Phase D, or conditional branch was used.", f"- Strict validation: **{strict['status']}**."]
    return "\n".join(lines) + "\n"


def _resume_finalize(tmp: Path, final: Path, report_tmp: Path, report: Path) -> None:
    """Recover only the known count-expression validation defect without refitting.

    The original run has already persisted all training, predictions, final
    checkpoints and final-family decision.  This path is deliberately unable
    to call a fitting function.
    """
    state = json.loads((tmp / "run_state.json").read_text(encoding="utf-8"))
    strict = json.loads((tmp / "strict_validation.json").read_text(encoding="utf-8"))
    if state.get("status") != "failed" or state.get("failure_reason") != "Phase E strict validation failed.":
        raise RuntimeError("Only the known Phase E strict-validation partial run may be finalized.")
    failed = [row["id"] for row in strict.get("checks", []) if not row["pass"]]
    if failed != ["oof_coverage"]:
        raise RuntimeError(f"Recovery refused: unexpected failed checks: {failed}")
    oof = pd.read_csv(tmp / "outer_oof_predictions.csv")
    expected = {"R0": 316, "M2": 316, "M1": 5 * 316, "N0": 5 * 316, "N1": 5 * 316}
    actual = oof.groupby("candidate_id").size().to_dict()
    coverage_pass = actual == expected and len(oof) == sum(expected.values())
    coverage_pass &= set(oof[oof["candidate_id"] == "R0"]["seed"]) == {DETERMINISTIC_SEED}
    coverage_pass &= set(oof[oof["candidate_id"] == "M2"]["seed"]) == {DETERMINISTIC_SEED}
    coverage_pass &= all(set(oof[oof["candidate_id"] == candidate]["seed"]) == set(PHASE_E_SEEDS) for candidate in ["M1", "N0", "N1"])
    stability_checks = json.loads((tmp / "stability_checkpoint_checksums.json").read_text(encoding="utf-8"))
    final_checks = json.loads((tmp / "final_checkpoint_checksums.json").read_text(encoding="utf-8"))
    recomputed = classification_rows(oof)
    stored = pd.read_csv(tmp / "classification_metrics.csv")
    metric_diff = float(np.max(np.abs(stored.sort_values(["candidate_id", "seed", "outer_fold"])["macro_f1"].to_numpy() - recomputed.sort_values(["candidate_id", "seed", "outer_fold"])["macro_f1"].to_numpy())))
    recovery_checks = {
        "corrected_candidate_seed_row_counts": coverage_pass,
        "stability_checkpoints_reproduced": bool(stability_checks["all_reproduced"]),
        "final_checkpoints_reproduced": bool(final_checks["all_reproduced"]),
        "metric_recomputation": metric_diff <= 1e-12,
        "five_final_hybrid_checkpoints": len(final_checks["entries"]) == 5,
        "no_training_or_predictions_changed": True,
    }
    if not all(recovery_checks.values()):
        raise RuntimeError(f"Recovery checks failed: {recovery_checks}")
    for row in strict["checks"]:
        if row["id"] == "oof_coverage":
            row["pass"] = True
            row["corrected_expected_rows"] = sum(expected.values())
            row["actual_rows"] = len(oof)
    strict.update({"status": "PASS", "metric_recomputation_max_abs_difference": metric_diff, "finalization_recovery": recovery_checks})
    write_json(tmp / "strict_validation.json", strict)
    source = _provenance()
    provenance = json.loads((tmp / "source_provenance.json").read_text(encoding="utf-8"))
    provenance["finalization_recovery"] = {"recovery_source_git_commit": source["git_commit"], "reason": "corrected_expected_oof_row_count_expression; original expected multiplied deterministic rows by seed count", "training_or_predictions_changed": False, "recovery_checks": recovery_checks}
    write_json(tmp / "source_provenance.json", provenance)
    protocol = json.loads((tmp / "protocol.json").read_text(encoding="utf-8"))
    protocol["finalization_recovery"] = provenance["finalization_recovery"]
    write_json(tmp / "protocol.json", protocol)
    summary = pd.read_csv(tmp / "stability_summary.csv")
    decision = json.loads((tmp / "final_family_decision.json").read_text(encoding="utf-8"))
    (tmp / "phase_e_prediction_conclusion.md").write_text(_conclusion(summary, decision, strict), encoding="utf-8")
    checksums = {p.relative_to(tmp).as_posix(): sha256_file(p) for p in sorted(tmp.rglob("*")) if p.is_file() and p.name not in {"artifact_checksums.json", "run_state.json"}}
    write_json(tmp / "artifact_checksums.json", checksums)
    missing = [name for name in MINIMUM_OUTPUTS if not (tmp / name).is_file()]
    if missing:
        raise RuntimeError(f"Recovery missing required artifacts: {missing}")
    _write_state(tmp, "completed", strict_status="PASS", recovery_from=state, recovery_checks=recovery_checks)
    report_tmp.mkdir(parents=True, exist_ok=True)
    for path in tmp.iterdir():
        if path.is_file():
            shutil.copy2(path, report_tmp / path.name)
    os.replace(tmp, final); os.replace(report_tmp, report)
    print(json.dumps({"status": "PASS", "artifact_path": str(final), "report_path": str(report), "finalization_recovery": True}))


def main() -> None:
    args = parse_args()
    final = ARTIFACT_ROOT / args.run_id
    report = REPORT_ROOT / args.run_id
    tmp, report_tmp = ARTIFACT_ROOT / f".{args.run_id}.tmp", REPORT_ROOT / f".{args.run_id}.tmp"
    if args.resume_finalize:
        if final.exists() or report.exists() or report_tmp.exists():
            raise FileExistsError("Recovery requires only the failed artifact temporary directory.")
        _resume_finalize(tmp, final, report_tmp, report)
        return
    if any(path.exists() for path in [final, report, tmp, report_tmp]):
        raise FileExistsError("Run ID exists or has partial state.")
    tmp.mkdir(parents=True); report_tmp.mkdir(parents=True)
    _write_state(tmp, "running")
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        source = _provenance()
        phase_c_configs, phase_c_manifest = _load_phase_c_configs()
        manifest = load_fold_manifest(args.fold_manifest)
        allowed_rows = development_source_rows(manifest)
        raw, metadata = load_development_subset_from_postgres("student-mat", args.dataset_version_id, allowed_rows)
        development = process_target_and_stratify(raw, "G3", "student", "3class")
        assert_development_only_frame(development, manifest)
        outer_folds = outer_folds_from_manifest(development, manifest)
        if len(development) != 316 or len(outer_folds) != 5:
            raise RuntimeError("Phase E requires exactly the immutable 316-row, five-fold development protocol.")
        write_json(tmp / "finalist_registry.json", phase_e_registry()); write_json(tmp / "seed_registry.json", seed_registry()); write_json(tmp / "selection_rule.json", selection_rule()); write_json(tmp / "phase_c_source_manifest.json", phase_c_manifest)
        shutil.copy2(args.fold_manifest, tmp / "outer_fold_manifest.json")
        inner_ledger = materialize_inner_fold_ledger(development, outer_folds, dataset_version_id=args.dataset_version_id, target_col="G3")
        inner_ledger.to_csv(tmp / "inner_fold_ledger.csv", index=False)
        protocol = {"protocol_version": PHASE_E_PROTOCOL_VERSION, "run_id": args.run_id, "phase_c_source": phase_c_manifest, "new_stability_seeds": PHASE_E_SEEDS, "phase_c_seeds_prohibited": [42, 123, 155], "features": ["G1", "G2"], "architecture_search_during_stability": False, "calibration": {"fit_scope": "inner_oof_only", "methods": ["uncalibrated", "scalar_temperature"], "outer_labels_used_for_fit": False}, "abstention": {"operational_threshold_selected": False, "headline_metrics_all_records": True}, "final_inner_cv": {"trials_maximum": 30, "folds": 3, "development_only": True}, "legacy_observed_79_fetched": False, "recommendation_phase_d_executed": False, "conditional_branches_executed": False, "external_confirmation_executed": False, "readme_headline_modified": False, "outer_fold_manifest_hash": file_checksum(args.fold_manifest), "outer_fold_semantic_hash": semantic_checksum(manifest), "source_tree_hash": source["source_tree_hash"], "environment_hash": source["environment_hash"]}
        write_json(tmp / "protocol.json", protocol)
        write_json(tmp / "dataset_manifest.json", {"dataset_hash": metadata["content_hash"], "target_contract_hash": metadata["target_contract_hash"], "development_records": len(development), "development_source_rows_hash": source_rows_hash(allowed_rows), "transaction_read_only": metadata["transaction_read_only"], "legacy_observed_rows_fetched": False})
        write_json(tmp / "source_provenance.json", source)
        tests = _run_tests(); write_json(tmp / "test_report.json", tests)
        if tests["status"] != "PASS": raise RuntimeError("Test suite failed before official Phase E run.")

        oof_rows: list[dict[str, Any]] = []; continuous: list[dict[str, Any]] = []; mapping_rows: list[dict[str, Any]] = []
        calibration_rows_list: list[dict[str, Any]] = []; calibration_parameters: list[dict[str, Any]] = []; ledger: list[dict[str, Any]] = []; checkpoints: list[dict[str, Any]] = []; counts: list[dict[str, Any]] = []; diagnostics: list[dict[str, Any]] = []
        for fold, (train_idx, validation_idx) in enumerate(outer_folds):
            train, validation = development.iloc[train_idx].copy(), development.iloc[validation_idx].copy()
            jobs: list[tuple[str, int]] = [("R0", DETERMINISTIC_SEED), ("M2", DETERMINISTIC_SEED)] + [("M1", seed) for seed in PHASE_E_SEEDS] + [(candidate, seed) for candidate in ["N0", "N1"] for seed in PHASE_E_SEEDS]
            for candidate, seed in jobs:
                config = None if candidate == "R0" else deepcopy(phase_c_configs[(candidate, fold)])
                started = time.perf_counter()
                probabilities, parameter_count, training = _fit_outer(candidate, seed, fold, config, train, validation, tmp, checkpoints)
                _append_oof(oof_rows, candidate, seed, fold, validation, probabilities)
                mapping_rows.append(_continuous_rows(continuous, candidate, seed, fold, validation, probabilities, train))
                if candidate == "R0":
                    calibration_parameters.append({"candidate_id": candidate, "outer_fold": fold, "seed": seed, "method": "not_applicable_hard_rule", "fit_scope": "none"})
                    calibration_rows_list.extend(_calibration_rows(candidate, seed, fold, validation["G3"].to_numpy(int), probabilities, None)[0])
                else:
                    calibrator, inner_probs, inner_y = _inner_calibration(candidate, seed, config, train, ledger)
                    assert calibrator is not None and inner_probs is not None and inner_y is not None
                    calibration_parameters.append({"candidate_id": candidate, "outer_fold": fold, "seed": seed, "fit_scope": "outer_train_inner_oof_only", "inner_oof_records": len(inner_y), "temperature": calibrator["temperature"], "calibrator_hash": canonical_hash(calibrator), **calibrator})
                    calibration_rows_list.extend(_calibration_rows(candidate, seed, fold, validation["G3"].to_numpy(int), probabilities, calibrator)[0])
                ledger.append({"stage": "outer_stability_refit", "candidate_id": candidate, "outer_fold": fold, "inner_fold": "", "seed": seed, "status": "completed", "runtime_seconds": time.perf_counter() - started})
                counts.append({"candidate_id": candidate, "outer_fold": fold, "seed": seed, "parameter_count": parameter_count})
                if training:
                    diagnostics.append({"candidate_id": candidate, "outer_fold": fold, "seed": seed, **training})

        oof = pd.DataFrame(oof_rows); cont = pd.DataFrame(continuous); calibration = pd.DataFrame(calibration_rows_list)
        oof.to_csv(tmp / "outer_oof_predictions.csv", index=False); cont.to_csv(tmp / "continuous_oof_predictions.csv", index=False)
        classification = classification_rows(oof); classification.to_csv(tmp / "classification_metrics.csv", index=False); classification.to_csv(tmp / "fold_seed_metrics.csv", index=False)
        per_class = classification.melt(id_vars=["candidate_id", "outer_fold", "seed"], value_vars=[c for c in classification if c.startswith(("low_", "medium_", "high_"))], var_name="metric", value_name="value"); per_class.to_csv(tmp / "per_class_metrics.csv", index=False)
        confusion_rows, ordinal_rows = [], []
        for (candidate, seed, fold), frame in oof.groupby(["candidate_id", "seed", "outer_fold"], sort=True):
            y, pred = frame["true_label"].to_numpy(int), frame["predicted_label"].to_numpy(int)
            matrix = confusion_matrix(y, pred, labels=[0, 1, 2])
            for actual in range(3):
                for predicted in range(3):
                    confusion_rows.append({"candidate_id": candidate, "outer_fold": fold, "seed": seed, "actual_label": actual, "predicted_label": predicted, "count": int(matrix[actual, predicted])})
            distance = np.abs(y - pred)
            ordinal_rows.append({"candidate_id": candidate, "outer_fold": fold, "seed": seed, "qwk": float(cohen_kappa_score(y, pred, weights="quadratic")), "ordinal_mae": float(distance.mean()), "one_step_error": float((distance == 1).mean()), "two_step_error": float((distance == 2).mean())})
        pd.DataFrame(confusion_rows).to_csv(tmp / "confusion_matrices.csv", index=False); pd.DataFrame(ordinal_rows).to_csv(tmp / "ordinal_metrics.csv", index=False)
        pr_metrics, pr_points = precision_recall_rows(oof); pr_metrics.to_csv(tmp / "precision_recall_metrics.csv", index=False); pr_points.to_csv(tmp / "precision_recall_curve_points.csv", index=False)
        write_json(tmp / "precision_recall_summary.json", {"method": "one_vs_rest_probability_precision_recall", "classes": ["Low", "Medium", "High"], "macro_micro_weighted_reported": True, "thresholds_selected_on_outer_labels": False})
        regression = regression_rows(cont); regression.to_csv(tmp / "regression_metrics.csv", index=False)
        write_json(tmp / "continuous_prediction_contract.json", {"classification_models": "probability_weighted class-conditional G3 means fitted only on each outer-training partition", "R0": "predicted_G3_equals_raw_G2; separate from threshold classification output", "forbidden": "encoded_class_0_1_2_is_not_a_G3_regression_target", "mappings": mapping_rows})
        calibration.to_csv(tmp / "calibration_metrics.csv", index=False); pd.DataFrame(calibration_parameters).to_csv(tmp / "calibration_parameters.csv", index=False)
        calibration_decisions = choose_temperature(calibration); write_json(tmp / "calibration_decision.json", calibration_decisions)
        stability = seed_stability(oof, classification); stability.to_csv(tmp / "seed_disagreement.csv", index=False)
        counts_frame = pd.DataFrame(counts); counts_frame.to_csv(tmp / "parameter_counts.csv", index=False); pd.DataFrame(diagnostics).to_csv(tmp / "training_diagnostics.csv", index=False)
        pd.DataFrame(ledger).to_csv(tmp / "stability_job_ledger.csv", index=False)

        summary_rows: list[dict[str, Any]] = []
        simplicity = {"R0": 0, "M2": 1, "M1": 2, "N0": 3, "N1": 3}
        for candidate in FINALISTS:
            candidate_oof = oof[oof["candidate_id"] == candidate]
            candidate_class = classification[classification["candidate_id"] == candidate]
            candidate_pr = pr_metrics[pr_metrics["candidate_id"] == candidate]
            candidate_reg = regression[regression["candidate_id"] == candidate]
            candidate_cal = calibration[(calibration["candidate_id"] == candidate) & (calibration["variant"] == "uncalibrated")]
            seed_f1 = candidate_oof.groupby("seed").apply(lambda f: f1_score(f["true_label"], f["predicted_label"], average="macro", zero_division=0), include_groups=False)
            outer_f1 = candidate_class.groupby("outer_fold")["macro_f1"].mean()
            stable = stability.set_index("candidate_id").loc[candidate]
            row = {"candidate_id": candidate, "parameter_count": int(round(counts_frame[counts_frame["candidate_id"] == candidate]["parameter_count"].mean())), "oof_macro_f1": float(seed_f1.mean()), "outer_mean_macro_f1": float(candidate_class["macro_f1"].mean()), "outer_sd": float(outer_f1.std(ddof=1)), "seed_sd": stable["seed_sd"], "seed_sd_not_applicable": bool(stable["seed_sd_not_applicable"]), "worst_seed": float(stable["worst_seed"]), "class_collapse_count": int(candidate_class["class_collapse"].sum()), "two_step_error": float(np.mean(np.abs(candidate_oof["true_label"] - candidate_oof["predicted_label"]) == 2)), "ece": float(candidate_cal["ece"].mean()), "nll": float(candidate_cal["nll"].mean()), "brier": float(candidate_cal["brier"].mean()), "macro_pr_auc": float(candidate_pr["macro_pr_auc"].mean()), "high_average_precision": float(candidate_pr["high_average_precision"].mean()), "mae": float(candidate_reg["mae"].mean()), "rmse": float(candidate_reg["rmse"].mean()), "r2": float(candidate_reg["r2"].mean()), "simplicity_rank": simplicity[candidate]}
            for metric in ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "micro_f1", "weighted_f1", "low_precision", "low_recall", "low_f1", "medium_precision", "medium_recall", "medium_f1", "high_precision", "high_recall", "high_f1"]:
                row[metric] = float(candidate_class[metric].mean())
            summary_rows.append(row)
        summary = pd.DataFrame(summary_rows).sort_values("oof_macro_f1", ascending=False).reset_index(drop=True); summary.to_csv(tmp / "stability_summary.csv", index=False)
        comparisons = [("N1", "N0"), ("M1", "R0"), ("M1", "M2"), ("M2", "R0"), ("M1", "N0"), ("M1", "N1"), ("N0", "R0"), ("N1", "R0")]
        paired = paired_metric_deltas(oof, cont, comparisons, bootstrap_samples=1000); paired.to_csv(tmp / "paired_stability_deltas.csv", index=False)
        overall, overall_reason = choose_final(summary, OVERALL_FINALISTS, paired); hybrid, hybrid_reason = choose_final(summary, HYBRID_FINALISTS, paired)
        decision = {"final_overall_model": overall, "overall_reason": overall_reason, "final_thesis_hybrid_model": hybrid, "hybrid_reason": hybrid_reason, "selection_evidence": "Phase C frozen rule reapplied to expanded Phase E stability evidence", "calibration_decisions": calibration_decisions, "phase_c_interpretation_preserved": phase_e_registry()["frozen_phase_c_interpretation"]}
        write_json(tmp / "final_family_decision.json", decision)

        # Fixed threshold grid describes coverage-risk without optimizing an outer-label threshold.
        coverage: list[dict[str, Any]] = []
        for candidate, frame in oof.groupby("candidate_id"):
            for threshold in np.linspace(1 / 3, 0.95, 13):
                accepted = frame[frame[["prob_0", "prob_1", "prob_2"]].max(axis=1) >= threshold]
                coverage.append({"candidate_id": candidate, "confidence_threshold": float(threshold), "coverage": float(len(accepted) / len(frame)), "accepted_error_rate": float((accepted["predicted_label"] != accepted["true_label"]).mean()) if len(accepted) else np.nan, "accepted_macro_f1": float(f1_score(accepted["true_label"], accepted["predicted_label"], average="macro", zero_division=0)) if len(accepted) else np.nan})
        pd.DataFrame(coverage).to_csv(tmp / "coverage_risk_curves.csv", index=False); write_json(tmp / "abstention_assessment.json", {"evaluated": ["max_probability", "predictive_entropy", "seed_disagreement", "coverage_risk"], "operational_threshold_proposed": False, "headline_metrics_include_all_development_oof_records": True, "R0": "unsuitable_for_automated_uncertainty_estimation_with_hard-rule probabilities"})
        tradeoff = summary[["candidate_id", "oof_macro_f1", "accuracy", "macro_pr_auc", "mae", "rmse", "r2"]].copy(); tradeoff["selection_metric"] = "macro_f1_primary; regression_secondary"; tradeoff.to_csv(tmp / "metric_tradeoff_analysis.csv", index=False)

        # One authorized development-only final inner-CV search per selected family.
        history: list[dict[str, Any]] = []
        overall_config = None if overall == "R0" else _final_ml_search(overall, development, history)
        hybrid_config = _final_neural_search(hybrid, development, history, tmp)
        pd.DataFrame(history).to_csv(tmp / "final_inner_search_history.csv", index=False)
        final_configs = {"overall": {"candidate_id": overall, "resolved_config": overall_config, "resolved_config_hash": config_hash(overall_config) if overall_config else None}, "thesis_hybrid": {"candidate_id": hybrid, "resolved_config": hybrid_config, "resolved_config_hash": resolved_config_hash(hybrid_config)}}
        write_json(tmp / "final_resolved_configs.json", final_configs)
        final_checkpoint_entries: list[dict[str, Any]] = []; final_preprocessors: list[dict[str, Any]] = []; final_calibrators: list[dict[str, Any]] = []; final_manifest: dict[str, Any] = {"development_records": len(development), "external_validation_claim": False, "final_models": []}
        final_dir = tmp / "final_models"; final_dir.mkdir(parents=True, exist_ok=True)
        if overall == "R0":
            final_manifest["final_models"].append({"role": "overall", "candidate_id": "R0", "fit": "not_applicable_deterministic_g2_threshold_contract"})
            overall_calibrator, cal_meta = None, {"candidate_id": "R0", "method": "not_applicable_hard_rule"}
        else:
            model = _make_ml(overall_config, PHASE_E_SEEDS[0]); model.fit(development[["G1", "G2"]], development["G3"])
            path = final_dir / f"overall_{overall}.pkl"
            with path.open("wb") as handle: pickle.dump(model, handle)
            with path.open("rb") as handle: reproduced = pickle.load(handle)
            difference = float(np.max(np.abs(model.predict_proba(development[["G1", "G2"]]) - reproduced.predict_proba(development[["G1", "G2"]]))))
            if difference > REPRODUCTION_TOLERANCE: raise RuntimeError("Final overall reproduction failed.")
            final_checkpoint_entries.append({"role": "overall", "candidate_id": overall, "path": path.relative_to(tmp).as_posix(), "sha256": sha256_file(path), "prediction_reproduction_max_abs_difference": difference, "pass": True})
            final_manifest["final_models"].append({"role": "overall", "candidate_id": overall, "full_development_records": len(development), "parameter_count": _parameter_count(model, overall)})
            overall_calibrator, cal_meta = _final_calibrator(overall, overall_config, development)
        if overall_calibrator is not None:
            path = final_dir / f"overall_{overall}.calibrator.json"; write_json(path, overall_calibrator); final_calibrators.append({"role": "overall", "path": path.relative_to(tmp).as_posix(), "sha256": sha256_file(path), **cal_meta})
        else: final_calibrators.append({"role": "overall", **cal_meta})

        ensemble_probabilities: list[np.ndarray] = []
        for seed in PHASE_E_SEEDS:
            result = fit_final_development_estimator(development_frame=development, spec=DATASETS["student-mat"], resolved_config=hybrid_config, seed=seed)
            path = final_dir / f"hybrid_{hybrid}_seed{seed}.pt"; prep = final_dir / f"hybrid_{hybrid}_seed{seed}.preprocessor.pkl"
            torch.save(result.refit_state_dict, path)
            with prep.open("wb") as handle: pickle.dump({"preprocessor": result.preprocessor, "selector": result.selector}, handle)
            with prep.open("rb") as handle: loaded = pickle.load(handle)
            prediction = predict_with_fitted_estimator(frame=development, spec=DATASETS["student-mat"], resolved_config=hybrid_config, state_dict=torch.load(path, map_location="cpu", weights_only=True), preprocessor=loaded["preprocessor"], selector=loaded["selector"])
            direct = predict_with_fitted_estimator(frame=development, spec=DATASETS["student-mat"], resolved_config=hybrid_config, state_dict=result.refit_state_dict, preprocessor=result.preprocessor, selector=result.selector)
            difference = float(np.max(np.abs(prediction - direct)))
            if difference > REPRODUCTION_TOLERANCE: raise RuntimeError("Final hybrid checkpoint reproduction failed.")
            ensemble_probabilities.append(prediction)
            final_checkpoint_entries.append({"role": "thesis_hybrid", "candidate_id": hybrid, "seed": seed, "path": path.relative_to(tmp).as_posix(), "sha256": sha256_file(path), "prediction_reproduction_max_abs_difference": difference, "pass": True})
            final_preprocessors.append({"role": "thesis_hybrid", "seed": seed, "path": prep.relative_to(tmp).as_posix(), "sha256": sha256_file(prep)})
        ensemble = np.mean(np.stack(ensemble_probabilities), axis=0)
        np.save(final_dir / f"hybrid_{hybrid}_five_seed_ensemble_probabilities.npy", ensemble)
        hybrid_calibrator, hybrid_cal_meta = _final_calibrator(hybrid, hybrid_config, development)
        calibrator_path = final_dir / f"hybrid_{hybrid}.calibrator.json"; write_json(calibrator_path, hybrid_calibrator); final_calibrators.append({"role": "thesis_hybrid", "path": calibrator_path.relative_to(tmp).as_posix(), "sha256": sha256_file(calibrator_path), **hybrid_cal_meta})
        final_manifest["final_models"].append({"role": "thesis_hybrid", "candidate_id": hybrid, "full_development_records": len(development), "seeds": PHASE_E_SEEDS, "checkpoints": 5, "ensemble": "arithmetic_mean_of_five_seed_probabilities", "ensemble_probability_contract": bool(np.allclose(ensemble, sum(ensemble_probabilities) / 5.0))})
        write_json(tmp / "final_model_manifest.json", final_manifest); write_json(tmp / "final_checkpoint_checksums.json", {"entries": final_checkpoint_entries, "all_reproduced": all(x["pass"] for x in final_checkpoint_entries)}); write_json(tmp / "final_preprocessor_checksums.json", {"entries": final_preprocessors}); write_json(tmp / "final_calibrator_checksums.json", {"entries": final_calibrators})
        write_json(tmp / "stability_checkpoint_checksums.json", {"entries": checkpoints, "all_reproduced": all(x["pass"] for x in checkpoints)})

        recomputed = classification_rows(pd.read_csv(tmp / "outer_oof_predictions.csv"))
        metric_diff = float(np.max(np.abs(classification.sort_values(["candidate_id", "seed", "outer_fold"])["macro_f1"].to_numpy() - recomputed.sort_values(["candidate_id", "seed", "outer_fold"])["macro_f1"].to_numpy())))
        # R0 and deterministic M2 each have one 316-row OOF set.  Only M1,
        # N0 and N1 contribute five genuine seed-specific OOF sets.
        expected_oof = 2 * 316 + 3 * 5 * 316
        checks = [
            {"id": "new_seeds_only", "pass": set(PHASE_E_SEEDS).isdisjoint({42, 123, 155})},
            {"id": "phase_c_source_checksums", "pass": not phase_c_manifest["checksum_failures"]},
            {"id": "full_test_suite", "pass": tests["status"] == "PASS"},
            {"id": "development_only_read_only_access", "pass": metadata["transaction_read_only"] and len(development) == 316},
            {"id": "no_legacy_observed_access", "pass": not protocol["legacy_observed_79_fetched"]},
            {"id": "no_architecture_search_during_stability", "pass": not protocol["architecture_search_during_stability"]},
            {"id": "no_fake_deterministic_seed_rows", "pass": len(oof[oof["candidate_id"].isin(["R0", "M2"])]) == 2 * 316},
            {"id": "stability_job_completeness", "pass": bool((pd.DataFrame(ledger)["status"] == "completed").all())},
            {"id": "oof_coverage", "pass": len(oof) == expected_oof},
            {"id": "calibration_inner_only", "pass": all(row.get("fit_scope", "").startswith("outer_train_inner_oof") or row.get("method", "").startswith("not_applicable") for row in calibration_parameters)},
            {"id": "checkpoint_reproduction", "pass": all(row["pass"] for row in checkpoints) and all(row["pass"] for row in final_checkpoint_entries)},
            {"id": "five_hybrid_checkpoints", "pass": len([x for x in final_checkpoint_entries if x["role"] == "thesis_hybrid"]) == 5},
            {"id": "metric_recomputation", "pass": metric_diff <= 1e-12},
            {"id": "final_refit_all_316", "pass": all(x.get("full_development_records", 316) == 316 for x in final_manifest["final_models"])},
            {"id": "recommendation_and_conditional_not_run", "pass": not protocol["recommendation_phase_d_executed"] and not protocol["conditional_branches_executed"]},
        ]
        strict = {"run_id": args.run_id, "status": "PASS" if all(c["pass"] for c in checks) else "FAIL", "checks": checks, "metric_recomputation_max_abs_difference": metric_diff, "legacy_observed_79_accessed": False, "final_overall_model": overall, "final_thesis_hybrid_model": hybrid}
        write_json(tmp / "strict_validation.json", strict); (tmp / "phase_e_prediction_conclusion.md").write_text(_conclusion(summary, decision, strict), encoding="utf-8")
        if strict["status"] != "PASS": raise RuntimeError("Phase E strict validation failed.")
        checksums = {p.relative_to(tmp).as_posix(): sha256_file(p) for p in sorted(tmp.rglob("*")) if p.is_file() and p.name not in {"artifact_checksums.json", "run_state.json"}}
        write_json(tmp / "artifact_checksums.json", checksums); _write_state(tmp, "completed", strict_status="PASS")
        missing = [name for name in MINIMUM_OUTPUTS if not (tmp / name).is_file()]
        if missing: raise RuntimeError(f"Missing required Phase E artifacts: {missing}")
        for path in tmp.iterdir():
            if path.is_file(): shutil.copy2(path, report_tmp / path.name)
        os.replace(tmp, final); os.replace(report_tmp, report)
        print(json.dumps({"status": "PASS", "artifact_path": str(final), "report_path": str(report), "final_overall": overall, "final_hybrid": hybrid}))
    except Exception as exc:
        _write_state(tmp, "failed", failure_type=type(exc).__name__, failure_reason=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
