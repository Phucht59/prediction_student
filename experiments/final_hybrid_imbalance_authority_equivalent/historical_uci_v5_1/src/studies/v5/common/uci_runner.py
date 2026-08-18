from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .artifacts import (
    atomic_write_json,
    build_checksum_manifest,
    result_fingerprint,
    safe_v5_root,
    verify_checksum_manifest,
)
from .metrics import multiclass_metrics
from .protocol import ROOT, load_study_protocol, protocol_fingerprint, sha256_file, verify_declared_sources
from .uci_data import UCIData, context_preprocessor, load_uci
from .uci_training import UCIInputs, fit_uci_model


IMBALANCE_STRATEGIES = ["none", "class_weight", "random_oversampling", "smote", "adasyn"]
BASELINES = ["logistic_regression", "decision_tree", "random_forest", "svm", "hist_gradient_boosting"]


def _prepared(data: UCIData, train_indices: np.ndarray, evaluation_indices: np.ndarray) -> tuple[UCIInputs, UCIInputs]:
    transformer = context_preprocessor()
    transformer.fit(data.context.iloc[train_indices])
    context_train = transformer.transform(data.context.iloc[train_indices]).astype(np.float32)
    context_evaluation = transformer.transform(data.context.iloc[evaluation_indices]).astype(np.float32)
    mean = data.sequence[train_indices].mean(axis=(0, 1), keepdims=True)
    std = data.sequence[train_indices].std(axis=(0, 1), keepdims=True).clip(1e-6)
    sequence_train = ((data.sequence[train_indices] - mean) / std).astype(np.float32)
    sequence_evaluation = ((data.sequence[evaluation_indices] - mean) / std).astype(np.float32)
    train = UCIInputs(sequence_train, context_train, data.target[train_indices], data.raw_g3[train_indices])
    evaluation = UCIInputs(
        sequence_evaluation,
        context_evaluation,
        data.target[evaluation_indices],
        data.raw_g3[evaluation_indices],
    )
    return train, evaluation


def _matrix(inputs: UCIInputs) -> np.ndarray:
    return np.concatenate([inputs.sequence.reshape(len(inputs.target), 2), inputs.context], axis=1)


def _sample_matrix(x: np.ndarray, y: np.ndarray, strategy: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if strategy in {"none", "class_weight"}:
        return x, y
    minimum = int(np.bincount(y, minlength=3).min())
    samplers = {
        "random_oversampling": RandomOverSampler(random_state=seed),
        "smote": SMOTE(random_state=seed, k_neighbors=min(5, max(1, minimum - 1))),
        "adasyn": ADASYN(random_state=seed, n_neighbors=min(5, max(1, minimum - 1))),
    }
    result_x, result_y = samplers[strategy].fit_resample(x, y)
    if len(result_y) <= len(y):
        raise RuntimeError(f"{strategy} was a no-op")
    return result_x, result_y


def _baseline(name: str, parameters: dict[str, Any], seed: int, class_weight: bool = False):
    weight = "balanced" if class_weight else None
    if name == "logistic_regression":
        return LogisticRegression(
            C=float(parameters.get("C", 1.0)),
            max_iter=3000,
            class_weight=weight,
            random_state=seed,
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=parameters.get("max_depth"),
            min_samples_leaf=int(parameters.get("min_samples_leaf", 2)),
            class_weight=weight,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(parameters.get("n_estimators", 300)),
            max_depth=parameters.get("max_depth"),
            min_samples_leaf=int(parameters.get("min_samples_leaf", 1)),
            max_features=parameters.get("max_features", "sqrt"),
            class_weight=weight,
            random_state=seed,
            n_jobs=6,
        )
    if name == "svm":
        return SVC(
            C=float(parameters.get("C", 1.0)),
            gamma=parameters.get("gamma", "scale"),
            kernel="rbf",
            class_weight=weight,
            probability=True,
            random_state=seed,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=float(parameters.get("learning_rate", 0.08)),
            max_leaf_nodes=int(parameters.get("max_leaf_nodes", 15)),
            l2_regularization=float(parameters.get("l2_regularization", 0.1)),
            max_iter=int(parameters.get("max_iter", 200)),
            random_state=seed,
        )
    raise KeyError(name)


def _sample_baseline(trial: optuna.Trial, name: str) -> dict[str, Any]:
    if name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-2, 100, log=True)}
    if name == "decision_tree":
        return {
            "max_depth": trial.suggest_categorical("max_depth", [2, 3, 4, 5, 7, None]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [200, 300, 500]),
            "max_depth": trial.suggest_categorical("max_depth", [4, 6, 10, None]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.7]),
        }
    if name == "svm":
        return {
            "C": trial.suggest_float("C", 1e-2, 100, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto", 0.01, 0.1, 1.0]),
        }
    if name == "hist_gradient_boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 31),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10, log=True),
            "max_iter": trial.suggest_categorical("max_iter", [100, 200, 300]),
        }
    raise KeyError(name)


def _sample_neural(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "cnn_channels": trial.suggest_categorical("cnn_channels", [8, 16, 24, 32]),
        "kernel_size": trial.suggest_categorical("kernel_size", [1, 2]),
        "lstm_hidden": trial.suggest_categorical("lstm_hidden", [8, 16, 24, 32]),
        "context_hidden": trial.suggest_categorical("context_hidden", [16, 24, 32, 48]),
        "fusion_hidden": trial.suggest_categorical("fusion_hidden", [16, 24, 32, 48]),
        "fusion": trial.suggest_categorical("fusion", ["gated", "concatenation"]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "multitask_alpha": trial.suggest_categorical("multitask_alpha", [0.0, 0.05, 0.1, 0.2]),
        "max_epochs": 100,
        "patience": 12,
        "gradient_clip": 1.0,
        "parameter_limit": 1_500_000,
    }


def _anchor_config() -> dict[str, Any]:
    return {
        "cnn_channels": 16,
        "kernel_size": 1,
        "lstm_hidden": 16,
        "context_hidden": 24,
        "fusion_hidden": 24,
        "fusion": "gated",
        "dropout": 0.2,
        "learning_rate": 8e-4,
        "weight_decay": 1e-5,
        "batch_size": 64,
        "multitask_alpha": 0.0,
        "max_epochs": 60,
        "patience": 8,
        "gradient_clip": 1.0,
        "parameter_limit": 1_500_000,
    }


def _fold_cache(root: Path, fold: int, fingerprint: str) -> Path:
    return root / "runtime_cache" / f"outer_{fold}_{fingerprint[:16]}.joblib"


def _load_cache(path: Path, fingerprint: str) -> dict[str, Any] | None:
    metadata = path.with_suffix(".json")
    if not path.is_file() or not metadata.is_file():
        return None
    value = json.loads(metadata.read_text(encoding="utf-8"))
    if value.get("fingerprint") != fingerprint or value.get("sha256") != sha256_file(path):
        return None
    return joblib.load(path)


def _save_cache(path: Path, fingerprint: str, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)
    atomic_write_json(path.with_suffix(".json"), {"fingerprint": fingerprint, "sha256": sha256_file(path)})


def _screen_imbalance(
    data: UCIData,
    outer_train: np.ndarray,
    outer_fold: int,
    split_seed: int,
    device: str,
) -> tuple[str, list[dict[str, Any]]]:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=split_seed + outer_fold)
    rows: list[dict[str, Any]] = []
    for strategy in IMBALANCE_STRATEGIES:
        for inner_fold, (relative_train, relative_validation) in enumerate(
            splitter.split(outer_train, data.target[outer_train])
        ):
            train_index = outer_train[relative_train]
            validation_index = outer_train[relative_validation]
            train, validation = _prepared(data, train_index, validation_index)
            fit = fit_uci_model(
                train,
                validation,
                config=_anchor_config(),
                seed=split_seed + outer_fold * 100 + inner_fold,
                imbalance_strategy=strategy,
                device_name=device,
            )
            metric = multiclass_metrics(validation.target, fit.probability)
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "strategy": strategy,
                    "macro_f1": metric["macro_f1"],
                    "before": json.dumps(fit.before_class_counts, sort_keys=True),
                    "after": json.dumps(fit.after_class_counts, sort_keys=True),
                }
            )
    summary = pd.DataFrame(rows).groupby("strategy").macro_f1.mean().sort_values(ascending=False)
    return str(summary.index[0]), rows


def _search_neural(
    data: UCIData,
    outer_train: np.ndarray,
    outer_fold: int,
    protocol: dict[str, Any],
    artifact: Path,
    imbalance: str,
    device: str,
) -> tuple[dict[str, Any], int, list[dict[str, Any]]]:
    database = artifact / "optuna.db"
    study = optuna.create_study(
        study_name=f"{data.dataset_id}_cnn_bilstm_outer_{outer_fold}",
        storage=f"sqlite:///{database.resolve().as_posix()}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(protocol["search"]["sampler_seed"]) + outer_fold),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
    )
    if not study.trials:
        study.enqueue_trial({key: value for key, value in _anchor_config().items() if key not in {"max_epochs", "patience", "gradient_clip", "parameter_limit"}})
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(protocol["splits"]["fixed_split_seed"]) + outer_fold)

    def objective(trial: optuna.Trial) -> float:
        config = _sample_neural(trial)
        scores: list[float] = []
        epochs: list[int] = []
        for inner_fold, (relative_train, relative_validation) in enumerate(
            splitter.split(outer_train, data.target[outer_train])
        ):
            train_index = outer_train[relative_train]
            validation_index = outer_train[relative_validation]
            train, validation = _prepared(data, train_index, validation_index)
            fit = fit_uci_model(
                train,
                validation,
                config=config,
                seed=int(protocol["search"]["sampler_seed"]) + outer_fold * 100 + inner_fold,
                imbalance_strategy=imbalance,
                device_name=device,
            )
            scores.append(float(f1_score(validation.target, fit.probability.argmax(1), average="macro")))
            epochs.append(fit.selected_epoch)
            trial.report(float(np.mean(scores)), inner_fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("config", json.dumps(config, sort_keys=True))
        trial.set_user_attr("epochs", epochs)
        return float(np.mean(scores))

    target_total = int(protocol["search"]["cnn_bilstm_trials"])
    per_outer = max(1, int(np.ceil(target_total / int(protocol["splits"]["outer"].rsplit("_", 1)[-1]))))
    complete = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    if complete < per_outer:
        study.optimize(objective, n_trials=per_outer - complete, catch=(RuntimeError,), show_progress_bar=False)
    best = study.best_trial
    config = json.loads(best.user_attrs["config"])
    epochs = max(1, int(round(np.median(best.user_attrs["epochs"]))))
    rows = [
        {
            "study": study.study_name,
            "outer_fold": outer_fold,
            "trial": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "parameters": json.dumps(trial.params, sort_keys=True),
        }
        for trial in study.trials
    ]
    return config, epochs, rows


def _search_baseline(
    data: UCIData,
    outer_train: np.ndarray,
    outer_fold: int,
    name: str,
    protocol: dict[str, Any],
    artifact: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    study = optuna.create_study(
        study_name=f"{data.dataset_id}_{name}_outer_{outer_fold}",
        storage=f"sqlite:///{(artifact / 'optuna.db').resolve().as_posix()}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(protocol["search"]["sampler_seed"]) + outer_fold),
    )
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(protocol["splits"]["fixed_split_seed"]) + outer_fold)

    def objective(trial: optuna.Trial) -> float:
        parameters = _sample_baseline(trial, name)
        scores = []
        for relative_train, relative_validation in splitter.split(outer_train, data.target[outer_train]):
            train, validation = _prepared(data, outer_train[relative_train], outer_train[relative_validation])
            model = _baseline(name, parameters, int(protocol["search"]["sampler_seed"]))
            model.fit(_matrix(train), train.target)
            scores.append(float(f1_score(validation.target, model.predict(_matrix(validation)), average="macro")))
        return float(np.mean(scores))

    total = int(protocol["search"]["ml_trials_per_tuned_family"])
    per_outer = max(1, int(np.ceil(total / 5)))
    complete = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    if complete < per_outer:
        study.optimize(objective, n_trials=per_outer - complete, catch=(RuntimeError, ValueError), show_progress_bar=False)
    rows = [
        {
            "study": study.study_name,
            "outer_fold": outer_fold,
            "trial": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "parameters": json.dumps(trial.params, sort_keys=True),
        }
        for trial in study.trials
    ]
    return dict(study.best_trial.params), rows


def _run_outer_fold(
    data: UCIData,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    fold: int,
    protocol: dict[str, Any],
    artifact: Path,
    device: str,
) -> dict[str, Any]:
    selected_imbalance, imbalance_rows = _screen_imbalance(
        data, train_indices, fold, int(protocol["splits"]["fixed_split_seed"]), device
    )
    neural_config, fixed_epochs, neural_trials = _search_neural(
        data, train_indices, fold, protocol, artifact, selected_imbalance, device
    )
    train, validation = _prepared(data, train_indices, validation_indices)
    predictions: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for seed in protocol["final_evaluation"]["seeds"]:
        fit = fit_uci_model(
            train,
            validation,
            config=neural_config,
            seed=int(seed),
            imbalance_strategy=selected_imbalance,
            fixed_epochs=fixed_epochs,
            device_name=device,
        )
        checkpoint_path = artifact / "checkpoints" / f"cnn_bilstm_outer_{fold}_seed_{seed}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fit.state_dict, checkpoint_path)
        checkpoints.append(
            {
                "outer_fold": fold,
                "seed": int(seed),
                "path": checkpoint_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(checkpoint_path),
                "state_dict_sha256": fit.checkpoint_sha256,
                "replay_max_abs_difference": fit.replay_max_abs_difference,
                "parameters": fit.parameter_count,
                "runtime_seconds": fit.runtime_seconds,
            }
        )
        for epoch in fit.history:
            curves.append({"outer_fold": fold, "seed": int(seed), **epoch})
        for row, probability, regression in zip(validation_indices, fit.probability, fit.regression):
            predictions.append(
                {
                    "record_id": data.record_ids[row],
                    "source_row": int(row),
                    "outer_fold": fold,
                    "candidate": "cnn_bilstm_v5",
                    "seed": int(seed),
                    "target": int(data.target[row]),
                    "raw_g3": float(data.raw_g3[row]),
                    "p_low": float(probability[0]),
                    "p_medium": float(probability[1]),
                    "p_high": float(probability[2]),
                    "regression_g3": float(regression) if float(neural_config["multitask_alpha"]) > 0 else None,
                }
            )
    baseline_trials: list[dict[str, Any]] = []
    selected_baselines: dict[str, Any] = {}
    for name in BASELINES:
        parameters, rows = _search_baseline(data, train_indices, fold, name, protocol, artifact)
        baseline_trials.extend(rows)
        selected_baselines[name] = parameters
        model = _baseline(name, parameters, int(protocol["splits"]["fixed_split_seed"]))
        model.fit(_matrix(train), train.target)
        probability = model.predict_proba(_matrix(validation))
        for row, values in zip(validation_indices, probability):
            predictions.append(
                {
                    "record_id": data.record_ids[row],
                    "source_row": int(row),
                    "outer_fold": fold,
                    "candidate": name,
                    "seed": 42,
                    "target": int(data.target[row]),
                    "raw_g3": float(data.raw_g3[row]),
                    "p_low": float(values[0]),
                    "p_medium": float(values[1]),
                    "p_high": float(values[2]),
                    "regression_g3": None,
                }
            )
    return {
        "predictions": predictions,
        "curves": curves,
        "checkpoints": checkpoints,
        "imbalance_rows": imbalance_rows,
        "neural_trials": neural_trials,
        "baseline_trials": baseline_trials,
        "selected": {
            "outer_fold": fold,
            "imbalance": selected_imbalance,
            "cnn_bilstm": neural_config,
            "fixed_epochs": fixed_epochs,
            "baselines": selected_baselines,
        },
    }


def _summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    for (candidate, seed), frame in predictions.groupby(["candidate", "seed"], dropna=False):
        probability = frame[["p_low", "p_medium", "p_high"]].to_numpy()
        regression = frame.regression_g3.to_numpy(dtype=float)
        has_regression = np.isfinite(regression).all()
        metric = multiclass_metrics(
            frame.target.to_numpy(),
            probability,
            regression_target=frame.raw_g3.to_numpy() if has_regression else None,
            regression_prediction=regression if has_regression else None,
        )
        metric_rows.append({"candidate": candidate, "seed": seed, **{k: v for k, v in metric.items() if k not in {"per_class", "confusion_matrix"}}})
    deep = predictions[predictions.candidate == "cnn_bilstm_v5"]
    ensemble = (
        deep.groupby(["record_id", "source_row", "outer_fold", "target", "raw_g3"], as_index=False)
        .agg({"p_low": "mean", "p_medium": "mean", "p_high": "mean", "regression_g3": "mean"})
    )
    ensemble["candidate"] = "cnn_bilstm_v5_ensemble"
    ensemble["seed"] = -1
    predictions = pd.concat([predictions, ensemble[predictions.columns]], ignore_index=True)
    ensemble_regression = ensemble.regression_g3.to_numpy(dtype=float)
    ensemble_has_regression = np.isfinite(ensemble_regression).all()
    metric = multiclass_metrics(
        ensemble.target.to_numpy(),
        ensemble[["p_low", "p_medium", "p_high"]].to_numpy(),
        regression_target=ensemble.raw_g3.to_numpy() if ensemble_has_regression else None,
        regression_prediction=ensemble_regression if ensemble_has_regression else None,
    )
    metric_rows.append({"candidate": "cnn_bilstm_v5_ensemble", "seed": -1, **{k: v for k, v in metric.items() if k not in {"per_class", "confusion_matrix"}}})
    return predictions, pd.DataFrame(metric_rows).sort_values("macro_f1", ascending=False)


def run_uci_study(study: str, *, device: str = "cuda", force: bool = False) -> dict[str, Any]:
    if study not in {"student-mat", "student-por"}:
        raise KeyError(study)
    started = time.perf_counter()
    protocol = load_study_protocol(study)
    source_audit = verify_declared_sources(protocol)
    if any(row["status"] != "PASS" for row in source_audit):
        raise RuntimeError(f"Source audit failed for {study}")
    data = load_uci(ROOT / protocol["source"]["path"], study)
    artifact = safe_v5_root(ROOT / "artifacts" / "v5" / study.replace("-", "_"))
    fingerprint = result_fingerprint(
        protocol_hash=protocol_fingerprint(study),
        source_hashes={study: protocol["source"]["sha256"]},
        config={"device": device},
    )
    checksum_path = artifact / "artifact_checksums.json"
    run_state_path = artifact / "run_state.json"
    if not force and run_state_path.is_file() and checksum_path.is_file():
        state = json.loads(run_state_path.read_text(encoding="utf-8"))
        manifest = json.loads(checksum_path.read_text(encoding="utf-8"))
        if state.get("status") == "COMPLETE" and state.get("fingerprint") == fingerprint and verify_checksum_manifest(artifact, manifest):
            return {"study": study, "status": "SKIPPED_VALID_CACHE", "artifact": str(artifact)}
    atomic_write_json(run_state_path, {"study": study, "status": "RUNNING", "fingerprint": fingerprint})
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(protocol["splits"]["fixed_split_seed"]))
    fold_results: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for fold, (train_indices, validation_indices) in enumerate(splitter.split(np.zeros(len(data.target)), data.target)):
        for role, indices in (("outer_train", train_indices), ("outer_validation", validation_indices)):
            split_rows.extend(
                {"outer_fold": fold, "role": role, "record_id": data.record_ids[index], "source_row": int(index)}
                for index in indices
            )
        cache = _fold_cache(artifact, fold, fingerprint)
        result = None if force else _load_cache(cache, fingerprint)
        if result is None:
            result = _run_outer_fold(data, train_indices, validation_indices, fold, protocol, artifact, device)
            _save_cache(cache, fingerprint, result)
        fold_results.append(result)
        atomic_write_json(
            run_state_path,
            {"study": study, "status": "RUNNING", "fingerprint": fingerprint, "completed_outer_folds": fold + 1},
        )
    predictions = pd.DataFrame([row for result in fold_results for row in result["predictions"]])
    predictions, metrics = _summarize(predictions)
    pd.DataFrame(split_rows).to_csv(artifact / "split_manifest.csv", index=False)
    predictions.to_csv(artifact / "oof_predictions.csv", index=False)
    metrics.to_csv(artifact / "final_metrics.csv", index=False)
    pd.DataFrame([row for result in fold_results for row in result["curves"]]).to_csv(artifact / "learning_curves.csv", index=False)
    pd.DataFrame([row for result in fold_results for row in result["imbalance_rows"]]).to_csv(artifact / "imbalance_comparison.csv", index=False)
    pd.DataFrame([row for result in fold_results for row in result["neural_trials"] + result["baseline_trials"]]).to_csv(artifact / "search_trials.csv", index=False)
    selected = [result["selected"] for result in fold_results]
    atomic_write_json(artifact / "selected_configs.json", selected)
    atomic_write_json(artifact / "checkpoint_metadata.json", [row for result in fold_results for row in result["checkpoints"]])
    atomic_write_json(artifact / "source_manifest.json", {"sources": source_audit, "records": len(data.target)})
    atomic_write_json(artifact / "protocol_snapshot.json", protocol)
    best = metrics.iloc[0]
    thesis = metrics[metrics.candidate == "cnn_bilstm_v5_ensemble"].iloc[0]
    registry = {
        "final_thesis_model": "CNN-BiLSTM V5 Ensemble",
        "final_overall_model": str(best.candidate),
        "reason": "Highest pooled development OOF Macro-F1; thesis and operational roles remain separate.",
        "metrics": {"thesis_macro_f1": float(thesis.macro_f1), "overall_macro_f1": float(best.macro_f1)},
        "limitations": ["Nested development OOF only", "G1/G2 sequence has two time steps", "No external causal recommendation validation"],
        "artifact_paths": ["final_metrics.csv", "oof_predictions.csv", "selected_configs.json"],
        "checkpoint_hashes": [row["sha256"] for result in fold_results for row in result["checkpoints"]],
    }
    atomic_write_json(artifact / "model_registry.json", registry)
    atomic_write_json(
        run_state_path,
        {
            "study": study,
            "status": "COMPLETE",
            "fingerprint": fingerprint,
            "runtime_seconds": time.perf_counter() - started,
            "future_accessed": False,
        },
    )
    manifest = build_checksum_manifest(artifact)
    atomic_write_json(checksum_path, manifest)
    return {"study": study, "status": "COMPLETE", "artifact": str(artifact), "registry": registry}


__all__ = ["BASELINES", "IMBALANCE_STRATEGIES", "run_uci_study"]
