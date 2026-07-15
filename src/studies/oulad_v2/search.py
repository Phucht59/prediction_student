from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import optuna

from .data import OULADV2Data, manifest_indices
from .metrics import choose_thresholds
from .training import fit_candidate


@dataclass
class SearchResult:
    candidate_id: str
    outer_fold: int
    config: dict[str, Any]
    thresholds: dict[str, Any]
    refit_epochs: int
    inner_selected_epochs: list[int]
    parameter_count: int
    trial_rows: list[dict[str, Any]]
    learning_curves: list[dict[str, Any]]
    runtime_seconds: float


def _base_training_config(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        **parameters,
        "max_epochs": 40,
        "patience": 6,
        "gradient_clip": 1.0,
    }


def suggest_h2t(trial: optuna.Trial) -> dict[str, Any]:
    return _base_training_config(
        {
            "conv_channels": trial.suggest_categorical("conv_channels", [24, 32, 48]),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
            "lstm_hidden": trial.suggest_categorical("lstm_hidden", [32, 48, 64]),
            "lstm_layers": trial.suggest_categorical("lstm_layers", [1, 2]),
            "static_hidden": 32,
            "fusion_hidden": 32,
            "dropout": trial.suggest_float("dropout", 0.10, 0.35),
            "learning_rate": trial.suggest_float("learning_rate", 2e-4, 2e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
            "scheduler": trial.suggest_categorical("scheduler", ["fixed_lr", "deterministic_cosine"]),
            "positive_weight": trial.suggest_categorical("positive_weight", ["none", "sqrt_balanced", "fully_balanced"]),
        }
    )


def suggest_a0(trial: optuna.Trial) -> dict[str, Any]:
    return _base_training_config(
        {
            "aggregate_hidden_1": trial.suggest_categorical("aggregate_hidden_1", [64, 96, 128]),
            "aggregate_hidden_2": trial.suggest_categorical("aggregate_hidden_2", [0, 32, 64]),
            "dropout": trial.suggest_float("dropout", 0.10, 0.35),
            "learning_rate": trial.suggest_float("learning_rate", 2e-4, 2e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
            "scheduler": trial.suggest_categorical("scheduler", ["fixed_lr", "deterministic_cosine"]),
            "positive_weight": trial.suggest_categorical("positive_weight", ["none", "sqrt_balanced", "fully_balanced"]),
        }
    )


def run_nested_search(
    data: OULADV2Data,
    candidate_id: str,
    outer_fold: int,
    inner_manifest,
    *,
    trials: int,
    device: str,
    search_seed: int,
) -> SearchResult:
    if candidate_id not in {"V2-H2T", "V2-A0"}:
        raise KeyError(candidate_id)
    started = time.perf_counter()
    trial_rows: list[dict[str, Any]] = []
    learning_curves: list[dict[str, Any]] = []
    suggest = suggest_h2t if candidate_id == "V2-H2T" else suggest_a0

    def objective(trial: optuna.Trial) -> float:
        config = suggest(trial)
        probabilities: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        selected_epochs: list[int] = []
        parameter_counts: list[int] = []
        fit_runtime = 0.0
        fit_count = 0
        trial_started = time.perf_counter()
        try:
            for inner_fold in sorted(inner_manifest["inner_fold"].unique()):
                train_indices, validation_indices = manifest_indices(data, inner_manifest, int(inner_fold))
                result = fit_candidate(
                    data,
                    candidate_id,
                    train_indices,
                    validation_indices,
                    temporal_config=config if candidate_id == "V2-H2T" else None,
                    aggregate_config=config if candidate_id == "V2-A0" else None,
                    seed=search_seed + int(inner_fold),
                    device_name=device,
                )
                fit_count += 1
                fit_runtime += result.runtime_seconds
                probabilities.append(result.probabilities)
                targets.append(data.y[validation_indices])
                selected_epochs.append(result.selected_epoch)
                parameter_counts.append(result.parameter_count)
                for row in result.history:
                    learning_curves.append(
                        {
                            "candidate_id": candidate_id,
                            "outer_fold": outer_fold,
                            "trial_id": trial.number,
                            "inner_fold": int(inner_fold),
                            **row,
                        }
                    )
            pooled_y = np.concatenate(targets)
            pooled_probability = np.concatenate(probabilities)
            thresholds = choose_thresholds(pooled_y, pooled_probability)
            score = float(thresholds["inner_macro_f1"])
            trial.set_user_attr("resolved_config", json.dumps(config, sort_keys=True))
            trial.set_user_attr("thresholds", json.dumps(thresholds, sort_keys=True))
            trial.set_user_attr("selected_epochs", json.dumps(selected_epochs))
            trial.set_user_attr("parameter_count", max(parameter_counts))
            trial.set_user_attr("fit_count", fit_count)
            trial.set_user_attr("fit_runtime_seconds", fit_runtime)
            return score
        except RuntimeError as error:
            trial.set_user_attr("failure_reason", str(error))
            if "out of memory" in str(error).lower():
                trial.set_user_attr("prune_reason", "cuda_oom")
                raise optuna.TrialPruned("CUDA OOM") from error
            raise
        finally:
            trial.set_user_attr("wall_runtime_seconds", time.perf_counter() - trial_started)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=search_seed))
    study.optimize(objective, n_trials=trials, catch=(RuntimeError,), show_progress_bar=False)
    for trial in study.trials:
        trial_rows.append(
            {
                "candidate_id": candidate_id,
                "outer_fold": outer_fold,
                "trial_id": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "resolved_config": trial.user_attrs.get("resolved_config"),
                "thresholds": trial.user_attrs.get("thresholds"),
                "selected_epochs": trial.user_attrs.get("selected_epochs"),
                "parameter_count": trial.user_attrs.get("parameter_count"),
                "fit_count": trial.user_attrs.get("fit_count", 0),
                "fit_runtime_seconds": trial.user_attrs.get("fit_runtime_seconds", 0.0),
                "wall_runtime_seconds": trial.user_attrs.get("wall_runtime_seconds", 0.0),
                "failure_reason": trial.user_attrs.get("failure_reason"),
                "prune_reason": trial.user_attrs.get("prune_reason"),
            }
        )
    if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
        raise RuntimeError(f"No completed trials for {candidate_id} outer fold {outer_fold}")
    best = study.best_trial
    config = json.loads(best.user_attrs["resolved_config"])
    thresholds = json.loads(best.user_attrs["thresholds"])
    selected_epochs = json.loads(best.user_attrs["selected_epochs"])
    refit_epochs = max(1, int(round(float(np.median(selected_epochs)))))
    return SearchResult(
        candidate_id,
        outer_fold,
        config,
        thresholds,
        refit_epochs,
        selected_epochs,
        int(best.user_attrs["parameter_count"]),
        trial_rows,
        learning_curves,
        time.perf_counter() - started,
    )


def fit_frozen_inner_threshold(
    data: OULADV2Data,
    candidate_id: str,
    inner_manifest,
    *,
    temporal_config: dict[str, Any] | None,
    aggregate_config: dict[str, Any] | None,
    fixed_epochs: int,
    device: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, float]:
    started = time.perf_counter()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    histories: list[dict[str, Any]] = []
    parameter_count = 0
    for inner_fold in sorted(inner_manifest["inner_fold"].unique()):
        train_indices, validation_indices = manifest_indices(data, inner_manifest, int(inner_fold))
        result = fit_candidate(
            data,
            candidate_id,
            train_indices,
            validation_indices,
            temporal_config=temporal_config,
            aggregate_config=aggregate_config,
            seed=seed + int(inner_fold),
            fixed_epochs=fixed_epochs,
            device_name=device,
        )
        probabilities.append(result.probabilities)
        targets.append(data.y[validation_indices])
        parameter_count = result.parameter_count
        for row in result.history:
            histories.append({"candidate_id": candidate_id, "inner_fold": int(inner_fold), "trial_id": "frozen_threshold", **row})
    thresholds = choose_thresholds(np.concatenate(targets), np.concatenate(probabilities))
    return thresholds, histories, parameter_count, time.perf_counter() - started


def result_to_json(result: SearchResult) -> dict[str, Any]:
    return asdict(result)
