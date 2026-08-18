from __future__ import annotations

import json
import time
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src.studies.v5.common.metrics import multiclass_metrics

from ..common.artifacts import atomic_write_json, build_checksum_manifest, safe_v5_1_root
from ..common.protocol import ROOT, load_protocol, protocol_hash, sha256_file, verify_source
from ..common.uci_data import UCIDataV51, load_uci_v5_1
from ..common.uci_training import UCIInputsV51, fit_uci_model_v5_1, prepare_partition
from ..common.uci_transfer import (
    SharedSubjectInputs,
    combine_subject_inputs,
    fit_shared_subject_model,
    overlap_safe_source_indices,
    pretrain_then_finetune,
)


SCREENING_SEEDS = (42, 2026, 3407)
FINAL_SEEDS = (42, 1201, 2026, 3407, 7319)


def anchor_config() -> dict[str, Any]:
    return {
        "input_projection": 24,
        "cnn_channels": 16,
        "lstm_hidden": 24,
        "lstm_layers": 1,
        "context_hidden": 24,
        "context_layers": 1,
        "fusion_hidden": 32,
        "fusion": "gated",
        "dropout": 0.15,
        "activation": "gelu",
        "objective": "classification_only",
        "regression_weight": 0.0,
        "ordinal_weight": 0.0,
        "classification_loss": "standard",
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "gradient_clip": 1.0,
        "max_epochs": 50,
        "patience": 8,
        "parameter_limit": 1_500_000,
    }


def _outer_indices(protocol: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    path = ROOT / protocol["splits"]["outer_manifest"]
    if sha256_file(path) != protocol["splits"]["outer_manifest_sha256"]:
        raise RuntimeError(f"Outer manifest hash mismatch: {path}")
    manifest = pd.read_csv(path)
    results = []
    for fold in range(int(protocol["splits"]["outer_folds"])):
        selected = manifest[manifest.outer_fold == fold]
        train = selected.loc[selected.role == "outer_train", "source_row"].to_numpy(dtype=int)
        validation = selected.loc[selected.role == "outer_validation", "source_row"].to_numpy(dtype=int)
        if len(set(train) & set(validation)) or not len(validation):
            raise RuntimeError(f"Invalid outer fold {fold}")
        results.append((train, validation))
    return results


def _inner_splits(
    data: UCIDataV51, outer_train: np.ndarray, *, split_seed: int, outer_fold: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=split_seed + outer_fold)
    return [
        (outer_train[relative_train], outer_train[relative_validation])
        for relative_train, relative_validation in splitter.split(
            outer_train, data.target[outer_train], data.quasi_groups[outer_train]
        )
    ]


def _score_standalone(
    data: UCIDataV51,
    splits: list[tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    seeds: tuple[int, ...],
    imbalance: str,
    device: str,
    *,
    max_epochs: int = 35,
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for inner_fold, (train_index, validation_index) in enumerate(splits):
        train, transformer = prepare_partition(data, train_index, train_index)
        validation, _ = prepare_partition(data, train_index, validation_index, fitted=transformer)
        for seed in seeds:
            local = {**config, "max_epochs": max_epochs, "patience": min(6, int(config.get("patience", 8)))}
            fit = fit_uci_model_v5_1(
                train,
                validation,
                config=local,
                seed=seed,
                imbalance_strategy=imbalance,
                device_name=device,
            )
            score = float(
                f1_score(validation.target, fit.probability.argmax(axis=1), average="macro", zero_division=0)
            )
            rows.append(
                {
                    "inner_fold": inner_fold,
                    "seed": seed,
                    "macro_f1": score,
                    "selected_epoch": fit.selected_epoch,
                    "runtime_seconds": fit.runtime_seconds,
                    "parameter_count": fit.parameter_count,
                    "gate_mean": fit.gate_stats["mean"],
                    "gate_variance": fit.gate_stats["variance"],
                    "gate_saturation_fraction": fit.gate_stats["saturation_fraction"],
                    "gate_collapsed": fit.gate_stats["collapsed"],
                }
            )
    return float(np.mean([row["macro_f1"] for row in rows])), rows


def _source_inputs(
    source: UCIDataV51,
    source_indices: np.ndarray,
    transformer,
) -> UCIInputsV51:
    return UCIInputsV51(
        temporal=source.temporal[source_indices].astype(np.float32),
        context=transformer.transform(source.context.iloc[source_indices]).astype(np.float32),
        target=source.target[source_indices].astype(np.int64),
        raw_g3=source.raw_g3[source_indices].astype(np.float32),
    )


def _target_shared(inputs: UCIInputsV51) -> SharedSubjectInputs:
    return SharedSubjectInputs(
        inputs.temporal,
        inputs.context,
        inputs.target,
        inputs.raw_g3,
        np.zeros(len(inputs.target), dtype=np.int64),
    )


def _score_transfer(
    target: UCIDataV51,
    source: UCIDataV51,
    splits: list[tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    method: str,
    device: str,
) -> tuple[float, list[dict[str, Any]]]:
    if method == "standalone":
        return _score_standalone(target, splits, config, SCREENING_SEEDS, "none", device, max_epochs=20)
    rows: list[dict[str, Any]] = []
    for inner_fold, (train_index, validation_index) in enumerate(splits):
        target_train, transformer = prepare_partition(target, train_index, train_index)
        target_validation, _ = prepare_partition(
            target, train_index, validation_index, fitted=transformer
        )
        safe_source = overlap_safe_source_indices(
            source.quasi_groups, target.quasi_groups[validation_index]
        )
        source_splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=3407 + inner_fold)
        relative_train, relative_validation = next(
            source_splitter.split(
                safe_source, source.target[safe_source], source.quasi_groups[safe_source]
            )
        )
        source_train = _source_inputs(source, safe_source[relative_train], transformer)
        source_validation = _source_inputs(source, safe_source[relative_validation], transformer)
        for seed in SCREENING_SEEDS:
            if method == "por_pretrain_mat_freeze_unfreeze":
                _, fit = pretrain_then_finetune(
                    source_train,
                    source_validation,
                    target_train,
                    target_validation,
                    config={**config, "unfreeze_learning_rate_fraction": 0.35},
                    seed=seed,
                    source_epochs=8,
                    target_epochs=8,
                    freeze_epochs=3,
                    device_name=device,
                )
                probability = fit.probability
                runtime = fit.runtime_seconds
            elif method == "shared_trunk_subject_specific_heads":
                combined = combine_subject_inputs(target_train, source_train)
                fit = fit_shared_subject_model(
                    combined,
                    _target_shared(target_validation),
                    config={**config, "subject_embedding_dim": 4},
                    seed=seed,
                    fixed_epochs=10,
                    device_name=device,
                )
                probability = fit.probability
                runtime = fit.runtime_seconds
            else:
                raise ValueError(method)
            score = float(
                f1_score(target_validation.target, probability.argmax(axis=1), average="macro", zero_division=0)
            )
            rows.append(
                {
                    "inner_fold": inner_fold,
                    "seed": seed,
                    "macro_f1": score,
                    "runtime_seconds": runtime,
                    "source_records": int(len(source_train.target)),
                    "excluded_overlap_records": int(len(source.target) - len(safe_source)),
                }
            )
    return float(np.mean([row["macro_f1"] for row in rows])), rows


def _screen_group(
    data: UCIDataV51,
    splits: list[tuple[np.ndarray, np.ndarray]],
    group: str,
    candidates: list[tuple[str, dict[str, Any], str]],
    device: str,
) -> dict[str, Any]:
    rows = []
    for name, changes, imbalance in candidates:
        score, detail = _score_standalone(
            data, splits, {**anchor_config(), **changes}, SCREENING_SEEDS, imbalance, device, max_epochs=18
        )
        fold_scores = [
            float(np.mean([row["macro_f1"] for row in detail if row["inner_fold"] == fold]))
            for fold in range(3)
        ]
        rows.append(
            {
                "candidate": name,
                "score": score,
                "positive_inner_folds_vs_anchor": None,
                "worst_inner_fold": min(fold_scores),
                "detail": detail,
                "changes": changes,
                "imbalance": imbalance,
            }
        )
    anchor = next(row for row in rows if row["candidate"] in {"gated", "classification_only", "none"})
    anchor_folds = [
        float(np.mean([row["macro_f1"] for row in anchor["detail"] if row["inner_fold"] == fold]))
        for fold in range(3)
    ]
    for row in rows:
        folds = [
            float(np.mean([item["macro_f1"] for item in row["detail"] if item["inner_fold"] == fold]))
            for fold in range(3)
        ]
        row["delta_vs_anchor"] = float(row["score"] - anchor["score"])
        row["positive_inner_folds_vs_anchor"] = int(sum(a > b for a, b in zip(folds, anchor_folds)))
    best = max(rows, key=lambda row: (row["score"], row["worst_inner_fold"]))
    return {
        "group": group,
        "selected": best["candidate"],
        "selection_score": best["score"],
        "stop_reason": "all_registered_component_candidates_evaluated",
        "candidates": rows,
    }


def screen_components(dataset: str, device: str = "cuda", force: bool = False) -> dict[str, Any]:
    protocol = load_protocol(dataset)
    verify_source(protocol)
    data = load_uci_v5_1(ROOT / protocol["source"]["path"], dataset)
    source = None
    if dataset == "student-mat":
        source_protocol = load_protocol("student-por")
        verify_source(source_protocol)
        source = load_uci_v5_1(ROOT / source_protocol["source"]["path"], "student-por")
    artifact = safe_v5_1_root(ROOT / "artifacts" / "v5_1" / dataset.replace("-", "_"))
    output_path = artifact / "component_screening.json"
    fingerprint = protocol_hash(dataset)
    previous_results: list[dict[str, Any]] = []
    if not force and output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") == fingerprint and previous.get("status") == "COMPLETE":
            return previous
        if previous.get("fingerprint") == fingerprint and previous.get("status") == "RUNNING":
            previous_results = list(previous.get("results", []))
    results = previous_results
    for outer_fold, (outer_train, _) in enumerate(_outer_indices(protocol)):
        if outer_fold < len(results):
            continue
        splits = _inner_splits(
            data,
            outer_train,
            split_seed=int(protocol["splits"]["split_seed"]),
            outer_fold=outer_fold,
        )
        groups = [
            _screen_group(
                data,
                splits,
                "fusion",
                [
                    ("gated", {"fusion": "gated"}, "none"),
                    ("concatenation", {"fusion": "concatenation"}, "none"),
                    ("film_residual", {"fusion": "film_residual"}, "none"),
                ],
                device,
            ),
            _screen_group(
                data,
                splits,
                "objective",
                [
                    ("classification_only", {"objective": "classification_only"}, "none"),
                    (
                        "classification_plus_huber_regression",
                        {"objective": "classification_plus_huber_regression", "regression_weight": 0.1},
                        "none",
                    ),
                    (
                        "classification_plus_huber_regression_plus_ordinal",
                        {
                            "objective": "classification_plus_huber_regression_plus_ordinal",
                            "regression_weight": 0.1,
                            "ordinal_weight": 0.05,
                        },
                        "none",
                    ),
                ],
                device,
            ),
            _screen_group(
                data,
                splits,
                "imbalance",
                [
                    ("none", {}, "none"),
                    ("class_weight", {}, "class_weight"),
                    ("random_sample_duplication", {}, "random_sample_duplication"),
                    ("focal", {"classification_loss": "focal", "focal_gamma": 2.0}, "focal"),
                ],
                device,
            ),
        ]
        transfer = None
        if source is not None:
            transfer_rows = []
            for method in [
                "standalone",
                "por_pretrain_mat_freeze_unfreeze",
                "shared_trunk_subject_specific_heads",
            ]:
                score, detail = _score_transfer(data, source, splits, anchor_config(), method, device)
                transfer_rows.append({"candidate": method, "score": score, "detail": detail})
            anchor_score = transfer_rows[0]["score"]
            for row in transfer_rows:
                row["delta_vs_standalone"] = float(row["score"] - anchor_score)
                row["positive_seed_count"] = int(
                    sum(
                        np.mean([item["macro_f1"] for item in row["detail"] if item["seed"] == seed])
                        > np.mean(
                            [item["macro_f1"] for item in transfer_rows[0]["detail"] if item["seed"] == seed]
                        )
                        for seed in SCREENING_SEEDS
                    )
                )
            best_transfer = max(transfer_rows, key=lambda row: row["score"])
            selected = (
                best_transfer["candidate"]
                if best_transfer["delta_vs_standalone"] > 0 and best_transfer["positive_seed_count"] >= 2
                else "standalone"
            )
            transfer = {
                "group": "transfer",
                "selected": selected,
                "candidates": transfer_rows,
                "stop_reason": "all_two_registered_transfer_methods_evaluated",
            }
        results.append({"outer_fold": outer_fold, "groups": groups, "transfer": transfer})
        atomic_write_json(
            output_path,
            {
                "status": "RUNNING",
                "fingerprint": fingerprint,
                "dataset": dataset,
                "completed_outer_folds": outer_fold + 1,
                "outer_results_used": False,
                "results": results,
            },
        )
    result = {
        "status": "COMPLETE",
        "fingerprint": fingerprint,
        "dataset": dataset,
        "screening_seeds": list(SCREENING_SEEDS),
        "outer_results_used": False,
        "results": results,
    }
    atomic_write_json(output_path, result)
    return result


def _retained(screen: dict[str, Any], outer_fold: int, group: str) -> list[str]:
    result = screen["results"][outer_fold]
    selected_group = next(item for item in result["groups"] if item["group"] == group)
    anchor_names = {"fusion": "gated", "objective": "classification_only", "imbalance": "none"}
    retained = [anchor_names[group]]
    for candidate in selected_group["candidates"]:
        if candidate["delta_vs_anchor"] > 0 and candidate["positive_inner_folds_vs_anchor"] >= 2:
            retained.append(candidate["candidate"])
    return sorted(set(retained))


def _sample_focused(trial: optuna.Trial, screen: dict[str, Any], outer_fold: int) -> tuple[dict[str, Any], str]:
    fusion = trial.suggest_categorical("fusion", _retained(screen, outer_fold, "fusion"))
    objective = trial.suggest_categorical("objective", _retained(screen, outer_fold, "objective"))
    imbalance = trial.suggest_categorical("imbalance", _retained(screen, outer_fold, "imbalance"))
    config = {
        "input_projection": trial.suggest_categorical("input_projection", [16, 24, 32]),
        "cnn_channels": trial.suggest_categorical("cnn_channels", [8, 16, 24, 32, 48]),
        "lstm_hidden": trial.suggest_categorical("lstm_hidden", [16, 24, 32, 48, 64]),
        "lstm_layers": trial.suggest_categorical("lstm_layers", [1, 2]),
        "context_hidden": trial.suggest_categorical("context_hidden", [16, 24, 32, 48, 64]),
        "context_layers": trial.suggest_categorical("context_layers", [1, 2]),
        "fusion_hidden": trial.suggest_categorical("fusion_hidden", [16, 24, 32, 48, 64]),
        "fusion": fusion,
        "dropout": trial.suggest_float("dropout", 0.05, 0.40),
        "activation": trial.suggest_categorical("activation", ["gelu", "relu"]),
        "objective": objective,
        "regression_weight": (
            trial.suggest_categorical("regression_weight", [0.05, 0.10, 0.20])
            if objective != "classification_only"
            else 0.0
        ),
        "ordinal_weight": (
            trial.suggest_categorical("ordinal_weight", [0.025, 0.05, 0.10])
            if objective == "classification_plus_huber_regression_plus_ordinal"
            else 0.0
        ),
        "classification_loss": "focal" if imbalance == "focal" else "standard",
        "focal_gamma": 2.0,
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "gradient_clip": 1.0,
        "max_epochs": 70,
        "patience": 10,
        "parameter_limit": 1_500_000,
    }
    return config, imbalance


def focused_search(dataset: str, device: str = "cuda", force: bool = False) -> dict[str, Any]:
    protocol = load_protocol(dataset)
    data = load_uci_v5_1(ROOT / protocol["source"]["path"], dataset)
    artifact = safe_v5_1_root(ROOT / "artifacts" / "v5_1" / dataset.replace("-", "_"))
    screen = screen_components(dataset, device, force=False)
    output_path = artifact / "focused_search.json"
    fingerprint = protocol_hash(dataset)
    previous_results: list[dict[str, Any]] = []
    if not force and output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") == fingerprint and previous.get("status") == "COMPLETE":
            return previous
        if previous.get("fingerprint") == fingerprint and previous.get("status") == "RUNNING":
            previous_results = list(previous.get("results", []))
    results = previous_results
    for outer_fold, (outer_train, _) in enumerate(_outer_indices(protocol)):
        if outer_fold < len(results):
            continue
        splits = _inner_splits(
            data,
            outer_train,
            split_seed=int(protocol["splits"]["split_seed"]),
            outer_fold=outer_fold,
        )
        storage = f"sqlite:///{(artifact / 'optuna.db').as_posix()}"
        study = optuna.create_study(
            study_name=f"{dataset}-outer-{outer_fold}-v5.1",
            direction="maximize",
            storage=storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=int(protocol["search"]["sampler_seed"])),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
        )

        def objective(trial: optuna.Trial) -> float:
            config, imbalance = _sample_focused(trial, screen, outer_fold)
            score, rows = _score_standalone(
                data, splits, config, (3407,), imbalance, device, max_epochs=70
            )
            trial.set_user_attr("config", config)
            trial.set_user_attr("imbalance", imbalance)
            trial.set_user_attr("selected_epoch_median", int(np.median([row["selected_epoch"] for row in rows])))
            trial.set_user_attr("runtime_seconds", float(sum(row["runtime_seconds"] for row in rows)))
            trial.set_user_attr("gpu_peak_memory_bytes", int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0)
            for step, row in enumerate(rows):
                trial.report(float(row["macro_f1"]), step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return score

        maximum = int(protocol["search"]["round_b_trials_max"])
        minimum = int(protocol["search"]["round_b_trials_min"])
        plateau = int(protocol["search"]["plateau_non_improving_trials"])
        completed_before = len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE])
        stop_reason = "maximum_budget"

        def callback(current_study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            nonlocal stop_reason
            completed = [item for item in current_study.trials if item.state == optuna.trial.TrialState.COMPLETE]
            if len(completed) < minimum:
                return
            best_number = current_study.best_trial.number
            later = [item for item in completed if item.number > best_number]
            if len(later) >= plateau:
                stop_reason = f"plateau_{plateau}_completed_trials_without_improvement"
                current_study.stop()

        if force or completed_before < maximum:
            study.optimize(
                objective,
                n_trials=max(0, maximum - completed_before),
                callbacks=[callback],
                catch=(RuntimeError, ValueError),
            )
        top = sorted(
            [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE],
            key=lambda trial: float(trial.value),
            reverse=True,
        )[:3]
        confirmations = []
        for trial in top:
            score, rows = _score_standalone(
                data,
                splits,
                dict(trial.user_attrs["config"]),
                SCREENING_SEEDS,
                str(trial.user_attrs["imbalance"]),
                device,
                max_epochs=70,
            )
            confirmations.append(
                {
                    "trial": trial.number,
                    "screening_seed_mean_macro_f1": score,
                    "screening_seed_std_macro_f1": float(np.std([row["macro_f1"] for row in rows])),
                    "worst_score": float(min(row["macro_f1"] for row in rows)),
                    "config": trial.user_attrs["config"],
                    "imbalance": trial.user_attrs["imbalance"],
                    "fixed_epochs": int(np.median([row["selected_epoch"] for row in rows])),
                }
            )
        selected = max(
            confirmations,
            key=lambda row: (row["screening_seed_mean_macro_f1"], row["worst_score"]),
        )
        results.append(
            {
                "outer_fold": outer_fold,
                "completed_trials": len(
                    [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
                ),
                "stop_reason": stop_reason,
                "top_confirmations": confirmations,
                "selected": selected,
            }
        )
        atomic_write_json(
            output_path,
            {
                "status": "RUNNING",
                "fingerprint": fingerprint,
                "dataset": dataset,
                "completed_outer_folds": outer_fold + 1,
                "results": results,
            },
        )
    result = {
        "status": "COMPLETE",
        "fingerprint": fingerprint,
        "dataset": dataset,
        "outer_results_used": False,
        "results": results,
    }
    atomic_write_json(output_path, result)
    return result


def final_evaluation(dataset: str, device: str = "cuda", force: bool = False) -> dict[str, Any]:
    protocol = load_protocol(dataset)
    data = load_uci_v5_1(ROOT / protocol["source"]["path"], dataset)
    artifact = safe_v5_1_root(ROOT / "artifacts" / "v5_1" / dataset.replace("-", "_"))
    screen = screen_components(dataset, device, force=False)
    search = focused_search(dataset, device, force=False)
    transfer_source = None
    if dataset == "student-mat":
        source_protocol = load_protocol("student-por")
        transfer_source = load_uci_v5_1(
            ROOT / source_protocol["source"]["path"], "student-por"
        )
    checkpoints = artifact / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    learning_curves: list[dict[str, Any]] = []
    for outer_fold, (outer_train, outer_validation) in enumerate(_outer_indices(protocol)):
        selected = search["results"][outer_fold]["selected"]
        config = dict(selected["config"])
        fixed_epochs = max(1, int(selected["fixed_epochs"]))
        train, transformer = prepare_partition(data, outer_train, outer_train)
        validation, _ = prepare_partition(data, outer_train, outer_validation, fitted=transformer)
        variants = {
            "cnn_bilstm_v5_1": ("cnn_bilstm", FINAL_SEEDS),
            "cnn_only_v5_1": ("cnn_only", SCREENING_SEEDS),
            "bilstm_only_v5_1": ("bilstm_only", SCREENING_SEEDS),
        }
        for candidate, (variant, seeds) in variants.items():
            for seed in seeds:
                variant_config = {**config, "temporal_variant": variant}
                fit = fit_uci_model_v5_1(
                    train,
                    validation,
                    config=variant_config,
                    seed=seed,
                    imbalance_strategy=str(selected["imbalance"]),
                    fixed_epochs=fixed_epochs,
                    device_name=device,
                )
                checkpoint = checkpoints / f"{candidate}_outer_{outer_fold}_seed_{seed}.pt"
                torch.save(fit.state_dict, checkpoint)
                metadata.append(
                    {
                        "candidate": candidate,
                        "outer_fold": outer_fold,
                        "seed": seed,
                        "path": checkpoint.relative_to(ROOT).as_posix(),
                        "sha256": sha256_file(checkpoint),
                        "state_dict_sha256": fit.checkpoint_sha256,
                        "replay_max_abs_difference": fit.replay_max_abs_difference,
                        "parameter_count": fit.parameter_count,
                        "runtime_seconds": fit.runtime_seconds,
                        "gate_statistics": fit.gate_stats,
                        "temporal_norm_mean": fit.temporal_norm_mean,
                        "context_norm_mean": fit.context_norm_mean,
                    }
                )
                for index, record_index in enumerate(outer_validation):
                    rows.append(
                        {
                            "record_id": data.record_ids[record_index],
                            "source_row": int(record_index),
                            "outer_fold": outer_fold,
                            "candidate": candidate,
                            "seed": seed,
                            "target": int(validation.target[index]),
                            "raw_g3": float(validation.raw_g3[index]),
                            "p_low": float(fit.probability[index, 0]),
                            "p_medium": float(fit.probability[index, 1]),
                            "p_high": float(fit.probability[index, 2]),
                            "regression_g3": float(fit.regression[index]),
                        }
                    )
                learning_curves.extend(
                    {"candidate": candidate, "outer_fold": outer_fold, "seed": seed, **history}
                    for history in fit.history
                )
        if transfer_source is not None:
            method = str(screen["results"][outer_fold]["transfer"]["selected"])
            safe_source = overlap_safe_source_indices(
                transfer_source.quasi_groups, data.quasi_groups[outer_validation]
            )
            source_splitter = StratifiedGroupKFold(
                n_splits=3, shuffle=True, random_state=3407 + outer_fold
            )
            relative_train, relative_validation = next(
                source_splitter.split(
                    safe_source,
                    transfer_source.target[safe_source],
                    transfer_source.quasi_groups[safe_source],
                )
            )
            source_train = _source_inputs(
                transfer_source, safe_source[relative_train], transformer
            )
            source_validation = _source_inputs(
                transfer_source, safe_source[relative_validation], transformer
            )
            for seed in FINAL_SEEDS:
                candidate = "cnn_bilstm_v5_1_transfer_selected"
                if method == "standalone":
                    source_rows = [
                        row
                        for row in rows
                        if row["outer_fold"] == outer_fold
                        and row["seed"] == seed
                        and row["candidate"] == "cnn_bilstm_v5_1"
                    ]
                    rows.extend(
                        {**row, "candidate": candidate, "transfer_method": method}
                        for row in source_rows
                    )
                    continue
                if method == "por_pretrain_mat_freeze_unfreeze":
                    pretrained, transfer_fit = pretrain_then_finetune(
                        source_train,
                        source_validation,
                        train,
                        validation,
                        config={**config, "unfreeze_learning_rate_fraction": 0.35},
                        seed=seed,
                        source_epochs=fixed_epochs,
                        target_epochs=fixed_epochs,
                        freeze_epochs=3,
                        device_name=device,
                        target_imbalance=str(selected["imbalance"]),
                    )
                    probability = transfer_fit.probability
                    regression = transfer_fit.regression
                    state_dict = transfer_fit.state_dict
                    checkpoint_sha = transfer_fit.checkpoint_sha256
                    replay_difference = transfer_fit.replay_max_abs_difference
                    parameter_count = transfer_fit.parameter_count
                    runtime_seconds = pretrained.runtime_seconds + transfer_fit.runtime_seconds
                    history = transfer_fit.history
                elif method == "shared_trunk_subject_specific_heads":
                    shared_fit = fit_shared_subject_model(
                        combine_subject_inputs(train, source_train),
                        _target_shared(validation),
                        config={**config, "subject_embedding_dim": 4},
                        seed=seed,
                        fixed_epochs=fixed_epochs,
                        device_name=device,
                    )
                    probability = shared_fit.probability
                    regression = np.full(len(validation.target), np.nan)
                    state_dict = shared_fit.state_dict
                    checkpoint_sha = shared_fit.checkpoint_sha256
                    replay_difference = shared_fit.replay_max_abs_difference
                    parameter_count = shared_fit.parameter_count
                    runtime_seconds = shared_fit.runtime_seconds
                    history = shared_fit.history
                else:
                    raise ValueError(method)
                checkpoint = checkpoints / f"{candidate}_outer_{outer_fold}_seed_{seed}.pt"
                torch.save(state_dict, checkpoint)
                metadata.append(
                    {
                        "candidate": candidate,
                        "transfer_method": method,
                        "outer_fold": outer_fold,
                        "seed": seed,
                        "path": checkpoint.relative_to(ROOT).as_posix(),
                        "sha256": sha256_file(checkpoint),
                        "state_dict_sha256": checkpoint_sha,
                        "replay_max_abs_difference": replay_difference,
                        "parameter_count": parameter_count,
                        "runtime_seconds": runtime_seconds,
                        "source_records": int(len(source_train.target)),
                        "excluded_overlap_records": int(len(transfer_source.target) - len(safe_source)),
                    }
                )
                for index, record_index in enumerate(outer_validation):
                    rows.append(
                        {
                            "record_id": data.record_ids[record_index],
                            "source_row": int(record_index),
                            "outer_fold": outer_fold,
                            "candidate": candidate,
                            "transfer_method": method,
                            "seed": seed,
                            "target": int(validation.target[index]),
                            "raw_g3": float(validation.raw_g3[index]),
                            "p_low": float(probability[index, 0]),
                            "p_medium": float(probability[index, 1]),
                            "p_high": float(probability[index, 2]),
                            "regression_g3": float(regression[index]),
                        }
                    )
                learning_curves.extend(
                    {
                        "candidate": candidate,
                        "transfer_method": method,
                        "outer_fold": outer_fold,
                        "seed": seed,
                        **row,
                    }
                    for row in history
                )
    predictions = pd.DataFrame(rows)
    predictions.to_parquet(artifact / "oof_predictions.parquet", index=False)
    pd.DataFrame(learning_curves).to_csv(artifact / "learning_curves.csv", index=False)
    seed_metrics = []
    for (candidate, seed), selected in predictions.groupby(["candidate", "seed"]):
        selected = selected.sort_values("source_row")
        regression_available = np.isfinite(selected.regression_g3.to_numpy()).all()
        metrics = multiclass_metrics(
            selected.target.to_numpy(),
            selected[["p_low", "p_medium", "p_high"]].to_numpy(),
            regression_target=selected.raw_g3.to_numpy() if regression_available else None,
            regression_prediction=selected.regression_g3.to_numpy() if regression_available else None,
        )
        seed_metrics.append({"candidate": candidate, "seed": int(seed), **metrics})
    ensemble_metrics_by_candidate = []
    for candidate, candidate_predictions in predictions.groupby("candidate"):
        ensemble = (
            candidate_predictions.groupby(
                ["record_id", "source_row", "outer_fold", "target", "raw_g3"], as_index=False
            )
            .agg(
                p_low=("p_low", "mean"),
                p_medium=("p_medium", "mean"),
                p_high=("p_high", "mean"),
                regression_g3=("regression_g3", "mean"),
            )
            .sort_values("source_row")
        )
        if len(ensemble) != len(data.target) or not np.array_equal(
            ensemble.source_row.to_numpy(), np.arange(len(data.target))
        ):
            raise RuntimeError(f"UCI V5.1 OOF record alignment failed for {candidate}")
        regression_available = np.isfinite(ensemble.regression_g3.to_numpy()).all()
        metrics = multiclass_metrics(
            ensemble.target.to_numpy(),
            ensemble[["p_low", "p_medium", "p_high"]].to_numpy(),
            regression_target=ensemble.raw_g3.to_numpy() if regression_available else None,
            regression_prediction=ensemble.regression_g3.to_numpy() if regression_available else None,
        )
        ensemble_metrics_by_candidate.append({"candidate": f"{candidate}_ensemble", **metrics})
    primary_base = (
        "cnn_bilstm_v5_1_transfer_selected" if dataset == "student-mat" else "cnn_bilstm_v5_1"
    )
    ensemble_metrics = next(
        row
        for row in ensemble_metrics_by_candidate
        if row["candidate"] == f"{primary_base}_ensemble"
    )
    full_seed_metrics = [row for row in seed_metrics if row["candidate"] == primary_base]
    result = {
        "status": "COMPLETE",
        "dataset": dataset,
        "candidate": f"{primary_base}_ensemble",
        "metrics": ensemble_metrics,
        "seed_metrics": seed_metrics,
        "ablation_metrics": ensemble_metrics_by_candidate,
        "seed_stability": {
            "mean": float(np.mean([row["macro_f1"] for row in full_seed_metrics])),
            "std": float(np.std([row["macro_f1"] for row in full_seed_metrics])),
            "min": float(min(row["macro_f1"] for row in full_seed_metrics)),
            "max": float(max(row["macro_f1"] for row in full_seed_metrics)),
        },
        "future_accessed": False,
    }
    atomic_write_json(artifact / "final_metrics.json", result)
    atomic_write_json(artifact / "checkpoint_metadata.json", metadata)
    atomic_write_json(artifact / "selected_configs.json", [row["selected"] for row in search["results"]])
    atomic_write_json(artifact / "protocol_snapshot.json", protocol)
    from .baselines import tune_and_evaluate_baselines

    result["ml"] = tune_and_evaluate_baselines(dataset, data, protocol, force=force)
    atomic_write_json(artifact / "artifact_checksums.json", build_checksum_manifest(artifact))
    return result


def run_uci_v5_1(
    dataset: str,
    *,
    phase: str = "all",
    device: str = "cuda",
    force: bool = False,
) -> dict[str, Any]:
    if dataset not in {"student-mat", "student-por"}:
        raise ValueError(dataset)
    phases: dict[str, Callable[[], dict[str, Any]]] = {
        "screen": lambda: screen_components(dataset, device, force),
        "search": lambda: focused_search(dataset, device, force),
        "final": lambda: final_evaluation(dataset, device, force),
    }
    if phase in phases:
        return phases[phase]()
    if phase != "all":
        raise ValueError(phase)
    started = time.perf_counter()
    screen_components(dataset, device, force)
    focused_search(dataset, device, force)
    result = final_evaluation(dataset, device, force)
    result["total_runtime_seconds"] = time.perf_counter() - started
    return result


__all__ = [
    "anchor_config",
    "final_evaluation",
    "focused_search",
    "run_uci_v5_1",
    "screen_components",
]
