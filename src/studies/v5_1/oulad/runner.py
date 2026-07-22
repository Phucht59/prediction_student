from __future__ import annotations

import json
import time
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score

from src.studies.oulad_v4.data import build_v4_inner_manifest, load_v4_data, manifest_indices
from src.studies.v5.common.metrics import binary_metrics_per_record_threshold

from ..common.artifacts import atomic_write_json, build_checksum_manifest, safe_v5_1_root
from ..common.protocol import ROOT, load_protocol, protocol_hash, sha256_file
from .data import prepare_oulad_inputs
from .pretraining import fit_masked_week_pretraining
from .training import choose_threshold, fit_oulad_model_v5_1, fit_prepared_oulad_model


SCREENING_SEEDS = (42, 2026, 3407)
FINAL_SEEDS = (42, 1201, 2026, 3407, 7319)


def anchor_config() -> dict[str, Any]:
    return {
        "input_projection": 48,
        "conv_channels": 32,
        "kernels": [2, 3, 5],
        "dilation": 1,
        "lstm_hidden": 64,
        "lstm_layers": 1,
        "pooling": "masked_attention",
        "pooling_projection": 64,
        "aggregate_hidden": 64,
        "static_hidden": 32,
        "fusion_hidden": 64,
        "fusion": "gated_residual",
        "branch_dropout": 0.1,
        "dropout": 0.2,
        "loss": "standard_bce",
        "focal_gamma": 2.0,
        "learning_rate": 5e-4,
        "pretraining_learning_rate": 5e-4,
        "weight_decay": 1e-5,
        "batch_size": 256,
        "gradient_clip": 1.0,
        "max_epochs": 50,
        "patience": 7,
        "parameter_limit": 1_500_000,
        "temporal_order": "original",
    }


def _load():
    protocol = load_protocol("oulad")
    for source in protocol["sources"].values():
        path = ROOT / source["path"]
        if sha256_file(path) != source["sha256"]:
            raise RuntimeError(f"OULAD source hash mismatch: {path}")
    v4_protocol = yaml.safe_load((ROOT / "configs/oulad_v4_protocol.yaml").read_text(encoding="utf-8"))
    data = load_v4_data(ROOT / "data/processed/study_c_oulad", v4_protocol)
    if data.dynamic_sequence.shape[2] != int(protocol["data_contract"]["sequence_channels"]):
        raise RuntimeError("OULAD sequence contract changed")
    return protocol, v4_protocol, data


def _inner_splits(data, outer_fold: int, v4_protocol: dict[str, Any]):
    manifest = build_v4_inner_manifest(data, outer_fold, v4_protocol)
    return [
        manifest_indices(data.v2, manifest, int(inner_fold))
        for inner_fold in sorted(manifest.inner_fold.unique())
    ]


def _score_config(
    data,
    splits,
    config: dict[str, Any],
    seeds: tuple[int, ...],
    device: str,
    *,
    augmentation: str = "none",
    variant: str = "cnn_bilstm",
    max_epochs: int = 35,
    initial_states: dict[tuple[int, int], dict[str, torch.Tensor]] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for inner_fold, (train_index, validation_index) in enumerate(splits):
        for seed in seeds:
            local = {**config, "max_epochs": max_epochs, "patience": min(6, int(config.get("patience", 7)))}
            fit = fit_oulad_model_v5_1(
                data,
                train_index,
                validation_index,
                config=local,
                seed=seed,
                augmentation=augmentation,
                variant=variant,
                device_name=device,
                initial_temporal_state=(initial_states or {}).get((inner_fold, seed)),
            )
            score = float(
                f1_score(data.y[validation_index], fit.probability >= 0.5, average="macro", zero_division=0)
            )
            rows.append(
                {
                    "inner_fold": inner_fold,
                    "seed": seed,
                    "macro_f1": score,
                    "selected_epoch": fit.selected_epoch,
                    "runtime_seconds": fit.runtime_seconds,
                    "parameter_count": fit.parameter_count,
                    "gpu_peak_memory_bytes": fit.gpu_peak_memory_bytes,
                    "gate_statistics": fit.gate_statistics,
                    "attention_entropy_mean": fit.attention_entropy_mean,
                    "attention_padding_max": fit.attention_padding_max,
                    "augmentation_audit": fit.augmentation_audit,
                }
            )
    return float(np.mean([row["macro_f1"] for row in rows])), rows


def _sample_architecture(trial: optuna.Trial) -> dict[str, Any]:
    kernels = trial.suggest_categorical("kernels", ["2_3", "3_5", "2_3_5"])
    return {
        **anchor_config(),
        "input_projection": trial.suggest_categorical("input_projection", [32, 48, 64]),
        "conv_channels": trial.suggest_categorical("conv_channels", [24, 32, 48, 64]),
        "kernels": {"2_3": [2, 3], "3_5": [3, 5], "2_3_5": [2, 3, 5]}[kernels],
        "dilation": trial.suggest_categorical("dilation", [1, 2]),
        "lstm_hidden": trial.suggest_categorical("lstm_hidden", [32, 48, 64, 96]),
        "lstm_layers": trial.suggest_categorical("lstm_layers", [1, 2]),
        "pooling": trial.suggest_categorical("pooling", ["masked_mean_max", "masked_attention"]),
        "pooling_projection": trial.suggest_categorical("pooling_projection", [32, 48, 64, 96]),
        "aggregate_hidden": trial.suggest_categorical("aggregate_hidden", [48, 64, 96]),
        "static_hidden": trial.suggest_categorical("static_hidden", [16, 24, 32, 48]),
        "fusion_hidden": trial.suggest_categorical("fusion_hidden", [32, 48, 64, 96]),
        "fusion": trial.suggest_categorical("fusion", ["concatenation", "gated_residual"]),
        "branch_dropout": trial.suggest_categorical("branch_dropout", [0.0, 0.1, 0.2]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
    }


def _completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]


def _trial_signature(trial: optuna.trial.FrozenTrial) -> str:
    return json.dumps(trial.params, sort_keys=True, separators=(",", ":"))


def _unique_completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    unique = []
    seen = set()
    for trial in _completed_trials(study):
        signature = _trial_signature(trial)
        if signature not in seen:
            seen.add(signature)
            unique.append(trial)
    return unique


def _unique_trial_count(study: optuna.Study) -> int:
    return len({_trial_signature(trial) for trial in study.trials if trial.params})


def _fully_evaluated_score(
    trial: optuna.trial.FrozenTrial, expected_steps: int
) -> float | None:
    """Return the full-fold mean for a trial that already paid the full compute cost."""
    values = [trial.intermediate_values.get(step) for step in range(expected_steps)]
    if all(value is not None for value in values):
        return float(np.mean(values))
    if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
        return float(trial.value)
    return None


def _unique_fully_evaluated_trials(
    study: optuna.Study, expected_steps: int
) -> list[tuple[optuna.trial.FrozenTrial, float]]:
    """Keep one immutable row per config, including legacy late-pruned full evaluations."""
    unique: dict[str, tuple[optuna.trial.FrozenTrial, float]] = {}
    for trial in study.trials:
        if not trial.params:
            continue
        score = _fully_evaluated_score(trial, expected_steps)
        if score is None:
            continue
        signature = _trial_signature(trial)
        previous = unique.get(signature)
        if previous is None or (
            previous[0].state != optuna.trial.TrialState.COMPLETE
            and trial.state == optuna.trial.TrialState.COMPLETE
        ):
            unique[signature] = (trial, score)
    return list(unique.values())


def _best_evaluated_is_recent(
    evaluated: list[tuple[optuna.trial.FrozenTrial, float]], window: int
) -> bool:
    if not evaluated:
        return False
    best_trial = max(evaluated, key=lambda item: item[1])[0]
    recent_numbers = {
        trial.number for trial, _ in evaluated[-max(1, int(window)) :]
    }
    return best_trial.number in recent_numbers


def _best_trial_is_recent(study: optuna.Study, window: int) -> bool:
    completed = _unique_completed_trials(study)
    if not completed:
        return False
    recent_numbers = {trial.number for trial in completed[-max(1, int(window)) :]}
    return study.best_trial.number in recent_numbers


def _architecture_screen(
    data,
    splits,
    artifact,
    outer_fold: int,
    protocol: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    storage = f"sqlite:///{(artifact / 'optuna.db').as_posix()}"
    study_name = f"oulad-v5.1-screen-outer-{outer_fold}"
    existing_names = {
        summary.study_name for summary in optuna.study.get_all_study_summaries(storage=storage)
    }
    existing_count = (
        len(optuna.load_study(study_name=study_name, storage=storage).trials)
        if study_name in existing_names
        else 0
    )
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=int(protocol["search"]["sampler_seed"]) + existing_count
        ),
        # Each architecture objective already evaluates every inner fold before
        # reporting. Late median pruning saves no compute and can reject a strong
        # full-fold mean because one individual fold is below its step median.
        pruner=optuna.pruners.NopPruner(),
    )

    def objective(trial: optuna.Trial) -> float:
        config = _sample_architecture(trial)
        duplicate = next(
            (
                earlier
                for earlier in study.trials
                if earlier.number < trial.number
                and earlier.state == optuna.trial.TrialState.COMPLETE
                and earlier.params == trial.params
            ),
            None,
        )
        if duplicate is not None:
            trial.set_user_attr("duplicate_of_trial", duplicate.number)
            raise optuna.TrialPruned()
        score, rows = _score_config(data, splits, config, (3407,), device, max_epochs=35)
        trial.set_user_attr("config", config)
        trial.set_user_attr("selected_epoch_median", int(np.median([row["selected_epoch"] for row in rows])))
        trial.set_user_attr("runtime_seconds", float(sum(row["runtime_seconds"] for row in rows)))
        for step, row in enumerate(rows):
            trial.report(float(row["macro_f1"]), step)
        return score

    initial = int(protocol["search"]["round_a_architecture_evaluated_trials_initial"])
    maximum = int(protocol["search"]["round_a_architecture_evaluated_trials_max"])
    extension_threshold = float(protocol["search"]["round_a_extension_threshold_macro_f1"])
    recent_window = int(protocol["search"]["round_a_recent_improvement_window"])
    expected_steps = len(splits)

    def run_until_evaluated(target: int) -> None:
        while len(_unique_fully_evaluated_trials(study, expected_steps)) < target:
            study.optimize(objective, n_trials=1, catch=(RuntimeError, ValueError))

    run_until_evaluated(initial)
    initial_evaluated = _unique_fully_evaluated_trials(study, expected_steps)
    best_initial_score = max(score for _, score in initial_evaluated)
    extend = best_initial_score > extension_threshold and _best_evaluated_is_recent(
        initial_evaluated, recent_window
    )
    if extend:
        run_until_evaluated(maximum)
    evaluated = _unique_fully_evaluated_trials(study, expected_steps)
    completed = _unique_completed_trials(study)
    stop_reason = (
        "stage_gate_extended_to_maximum_recent_improvement"
        if extend
        else "stage_gate_stopped_at_initial_budget"
    )
    top = sorted(evaluated, key=lambda item: item[1], reverse=True)[
        : int(protocol["search"]["round_a_top_configs_to_confirm"])
    ]
    confirmations = []
    for trial, screening_score in top:
        score, rows = _score_config(
            data, splits, dict(trial.user_attrs["config"]), SCREENING_SEEDS, device, max_epochs=40
        )
        confirmations.append(
            {
                "trial": trial.number,
                "screening_macro_f1_mean": screening_score,
                "macro_f1_mean": score,
                "macro_f1_std": float(np.std([row["macro_f1"] for row in rows])),
                "worst_score": float(min(row["macro_f1"] for row in rows)),
                "fixed_epochs": int(np.median([row["selected_epoch"] for row in rows])),
                "config": trial.user_attrs["config"],
            }
        )
    selected = max(confirmations, key=lambda row: (row["macro_f1_mean"], row["worst_score"]))
    return {
        "screening_outer_fold": outer_fold,
        "completed_unique_trials": len(completed),
        "fully_evaluated_unique_trials": len(evaluated),
        "budget_unit": "unique_fully_evaluated_configs",
        "recovered_fully_evaluated_pruned_trials": [
            trial.number
            for trial, _ in evaluated
            if trial.state == optuna.trial.TrialState.PRUNED
        ],
        "completed_trial_rows": len(_completed_trials(study)),
        "preserved_pruned_trials": len(
            [trial for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED]
        ),
        "stop_reason": stop_reason,
        "extended_after_initial_gate": extend,
        "top_confirmations": confirmations,
        "selected": selected,
    }


def architecture_screening(device: str = "cuda", force: bool = False) -> dict[str, Any]:
    """Run and checkpoint the architecture-only stage before optional component gates."""
    protocol, v4_protocol, data = _load()
    artifact = safe_v5_1_root(ROOT / "artifacts/v5_1/oulad")
    output_path = artifact / "architecture_screening.json"
    fingerprint = protocol_hash("oulad")
    if not force and output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") == fingerprint and previous.get("status") == "COMPLETE":
            return previous

    outer_fold = int(protocol["search"]["component_screening_outer_fold"])
    splits = _inner_splits(data, outer_fold, v4_protocol)
    architecture = _architecture_screen(
        data, splits, artifact, outer_fold, protocol, device
    )
    result = {
        "status": "COMPLETE",
        "fingerprint": fingerprint,
        "screening_outer_fold": outer_fold,
        "screening_seeds": list(SCREENING_SEEDS),
        "architecture": architecture,
        "outer_results_used": False,
        "future_accessed": False,
    }
    atomic_write_json(output_path, result)
    return result


def _pretraining_screen(
    data,
    splits,
    selected_architecture: dict[str, Any],
    protocol: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    configurations = [
        None,
        (0.10, 5),
        (0.20, 5),
        (0.20, 10),
    ][: int(protocol["pretraining"]["controlled_configs_max"])]
    rows = []
    for candidate in configurations:
        fold_scores = []
        detail = []
        for inner_fold, (train_index, validation_index) in enumerate(splits):
            train = prepare_oulad_inputs(data, train_index, train_index)
            validation = prepare_oulad_inputs(
                data, train_index, validation_index, fitted=train.preprocessors
            )
            initial_state = None
            pretrain_metadata = None
            if candidate is not None:
                fraction, epochs = candidate
                pretrain = fit_masked_week_pretraining(
                    train,
                    dynamic_channel_order=data.dynamic_channel_order,
                    config=selected_architecture,
                    seed=42,
                    epochs=epochs,
                    mask_fraction=fraction,
                    device_name=device,
                )
                initial_state = pretrain.temporal_state_dict
                pretrain_metadata = {
                    "checkpoint_sha256": pretrain.checkpoint_sha256,
                    "masked_values": pretrain.masked_values,
                    "runtime_seconds": pretrain.runtime_seconds,
                }
            fit = fit_prepared_oulad_model(
                train,
                validation,
                config={**selected_architecture, "max_epochs": 40, "patience": 6},
                seed=42,
                device_name=device,
                initial_temporal_state=initial_state,
            )
            score = float(
                f1_score(validation.target, fit.probability >= 0.5, average="macro", zero_division=0)
            )
            fold_scores.append(score)
            detail.append(
                {
                    "inner_fold": inner_fold,
                    "macro_f1": score,
                    "selected_epoch": fit.selected_epoch,
                    "pretraining": pretrain_metadata,
                }
            )
        name = "none" if candidate is None else f"mask_{candidate[0]:.2f}_epochs_{candidate[1]}"
        rows.append(
            {
                "candidate": name,
                "mask_fraction": None if candidate is None else candidate[0],
                "epochs": 0 if candidate is None else candidate[1],
                "macro_f1_mean": float(np.mean(fold_scores)),
                "worst_fold": float(min(fold_scores)),
                "detail": detail,
            }
        )
    anchor = rows[0]
    for row in rows:
        row["delta_vs_none"] = float(row["macro_f1_mean"] - anchor["macro_f1_mean"])
        row["positive_inner_folds"] = int(
            sum(
                item["macro_f1"] > anchor["detail"][index]["macro_f1"]
                for index, item in enumerate(row["detail"])
            )
        )
    best = max(rows, key=lambda row: (row["macro_f1_mean"], row["worst_fold"]))
    selected = (
        best
        if best["delta_vs_none"]
        >= float(protocol["pretraining"]["retain_requires_mean_inner_delta_at_least"])
        and best["positive_inner_folds"] >= 2
        and best["worst_fold"] >= anchor["worst_fold"] - 0.01
        else anchor
    )
    return {
        "selected": selected["candidate"],
        "selected_config": selected,
        "candidates": rows,
        "stop_reason": "stage_gate_all_registered_pretraining_configs_evaluated",
    }


def _controlled_screen(
    data,
    splits,
    config: dict[str, Any],
    protocol: dict[str, Any],
    device: str,
    group: str,
) -> dict[str, Any]:
    if group == "augmentation":
        candidates = [
            ("none", {}, "none"),
            ("event_thinning", {}, "event_thinning"),
            ("short_span_masking", {}, "short_span_masking"),
            ("channel_group_dropout", {}, "channel_group_dropout"),
        ]
    elif group == "loss":
        candidates = [
            ("standard_bce", {"loss": "standard_bce"}, "none"),
            ("weighted_bce", {"loss": "weighted_bce"}, "none"),
            ("focal", {"loss": "focal", "focal_gamma": 2.0}, "none"),
        ]
    else:
        raise ValueError(group)
    rows = []
    for name, changes, augmentation in candidates:
        score, detail = _score_config(
            data,
            splits,
            {**config, **changes},
            SCREENING_SEEDS,
            device,
            augmentation=augmentation,
            max_epochs=35,
        )
        fold_scores = [
            float(np.mean([row["macro_f1"] for row in detail if row["inner_fold"] == fold]))
            for fold in range(3)
        ]
        rows.append(
            {
                "candidate": name,
                "macro_f1_mean": score,
                "worst_fold": min(fold_scores),
                "fold_scores": fold_scores,
                "detail": detail,
            }
        )
    anchor = rows[0]
    for row in rows:
        row["delta_vs_anchor"] = float(row["macro_f1_mean"] - anchor["macro_f1_mean"])
        row["positive_inner_folds"] = int(
            sum(value > anchor["fold_scores"][index] for index, value in enumerate(row["fold_scores"]))
        )
    best = max(rows, key=lambda row: (row["macro_f1_mean"], row["worst_fold"]))
    minimum_delta = float(protocol[group]["retain_requires_mean_inner_delta_at_least"])
    selected = (
        best
        if best["delta_vs_anchor"] >= minimum_delta and best["positive_inner_folds"] >= 2
        else anchor
    )
    return {
        "group": group,
        "selected": selected["candidate"],
        "candidates": rows,
        "stop_reason": "all_registered_controlled_candidates_evaluated",
    }


def _skipped_component_group(group: str, reason: str, selected: str) -> dict[str, Any]:
    return {
        "group": group,
        "selected": selected,
        "candidates": [],
        "stop_reason": reason,
        "executed": False,
    }


def screen_components(device: str = "cuda", force: bool = False) -> dict[str, Any]:
    protocol, v4_protocol, data = _load()
    artifact = safe_v5_1_root(ROOT / "artifacts/v5_1/oulad")
    output_path = artifact / "component_screening.json"
    fingerprint = protocol_hash("oulad")
    previous_results: list[dict[str, Any]] = []
    if not force and output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") == fingerprint and previous.get("status") == "COMPLETE":
            return previous
        if previous.get("fingerprint") == fingerprint and previous.get("status") == "RUNNING":
            previous_results = list(previous.get("results", []))
    results = previous_results
    outer_fold = int(protocol["search"]["component_screening_outer_fold"])
    if not results:
        splits = _inner_splits(data, outer_fold, v4_protocol)
        architecture = architecture_screening(device, force=force)["architecture"]
        selected_config = dict(architecture["selected"]["config"])
        architecture_score = float(architecture["selected"]["macro_f1_mean"])
        pretraining_gate = float(
            protocol["pretraining"][
                "open_only_if_selected_architecture_inner_macro_f1_at_least"
            ]
        )
        if architecture_score >= pretraining_gate:
            pretraining = _pretraining_screen(
                data, splits, selected_config, protocol, device
            )
            pretraining["executed"] = True
        else:
            pretraining = {
                "selected": "none",
                "selected_config": {
                    "candidate": "none",
                    "mask_fraction": None,
                    "epochs": 0,
                    "macro_f1_mean": architecture_score,
                    "delta_vs_none": 0.0,
                },
                "candidates": [],
                "stop_reason": "stage_gate_architecture_below_0.8305",
                "executed": False,
            }

        outer_train, _ = data.v2.outer_indices(outer_fold)
        positives = int(data.y[outer_train].sum())
        negatives = int(len(outer_train) - positives)
        minority_majority_ratio = float(min(positives, negatives) / max(positives, negatives))
        fixed_epochs = int(architecture["selected"]["fixed_epochs"])
        overfit_limit = float(
            protocol["augmentation"]["overfit_evidence_selected_epoch_fraction_at_most"]
        )
        overfit_evidence = fixed_epochs <= int(40 * overfit_limit)
        imbalance_limit = float(
            protocol["loss"]["imbalance_evidence_minority_majority_ratio_below"]
        )
        imbalance_evidence = minority_majority_ratio < imbalance_limit
        evidence = {
            "outer_fold": outer_fold,
            "selected_fixed_epochs": fixed_epochs,
            "overfit_epoch_limit": int(40 * overfit_limit),
            "overfit_evidence": overfit_evidence,
            "positive_records": positives,
            "negative_records": negatives,
            "minority_majority_ratio": minority_majority_ratio,
            "imbalance_ratio_limit": imbalance_limit,
            "imbalance_evidence": imbalance_evidence,
        }
        augmentation = (
            _controlled_screen(
                data, splits, selected_config, protocol, device, "augmentation"
            )
            if overfit_evidence
            else _skipped_component_group(
                "augmentation", "stage_gate_no_documented_overfit_evidence", "none"
            )
        )
        augmentation["executed"] = overfit_evidence
        loss = (
            _controlled_screen(data, splits, selected_config, protocol, device, "loss")
            if imbalance_evidence
            else _skipped_component_group(
                "loss", "stage_gate_no_documented_imbalance_evidence", "standard_bce"
            )
        )
        loss["executed"] = imbalance_evidence
        results.append(
            {
                "outer_fold": outer_fold,
                "architecture": architecture,
                "pretraining": pretraining,
                "augmentation": augmentation,
                "loss": loss,
                "training_side_evidence": evidence,
            }
        )
        atomic_write_json(
            output_path,
            {
                "status": "RUNNING",
                "fingerprint": fingerprint,
                "completed_outer_folds": 1,
                "results": results,
                "future_accessed": False,
            },
        )
    result = {
        "status": "COMPLETE",
        "fingerprint": fingerprint,
        "screening_seeds": list(SCREENING_SEEDS),
        "screening_outer_folds": [outer_fold],
        "outer_results_used": False,
        "results": results,
        "future_accessed": False,
    }
    atomic_write_json(output_path, result)
    return result


def _pretraining_spec(screen: dict[str, Any], outer_fold: int) -> tuple[float, int] | None:
    selected = screen["results"][outer_fold]["pretraining"]["selected_config"]
    if selected["candidate"] == "none":
        return None
    return float(selected["mask_fraction"]), int(selected["epochs"])


def _build_inner_pretraining_states(
    data,
    splits,
    config: dict[str, Any],
    spec: tuple[float, int] | None,
    seeds: tuple[int, ...],
    device: str,
) -> dict[tuple[int, int], dict[str, torch.Tensor]]:
    if spec is None:
        return {}
    fraction, epochs = spec
    states: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    for inner_fold, (train_index, _) in enumerate(splits):
        train = prepare_oulad_inputs(data, train_index, train_index)
        for seed in seeds:
            result = fit_masked_week_pretraining(
                train,
                dynamic_channel_order=data.dynamic_channel_order,
                config=config,
                seed=seed,
                epochs=epochs,
                mask_fraction=fraction,
                device_name=device,
            )
            states[(inner_fold, seed)] = result.temporal_state_dict
    return states


def _sample_focused(
    trial: optuna.Trial,
    base: dict[str, Any],
    selected_loss: str,
) -> dict[str, Any]:
    loss_candidates = ["standard_bce", selected_loss] if selected_loss != "standard_bce" else ["standard_bce"]
    fusion_candidates = sorted({str(base["fusion"]), "gated_residual"})
    return {
        **base,
        "fusion": trial.suggest_categorical("fusion", fusion_candidates),
        "branch_dropout": trial.suggest_categorical("branch_dropout", [0.0, 0.1, 0.2]),
        "dropout": trial.suggest_float("dropout", max(0.1, float(base["dropout"]) - 0.1), min(0.4, float(base["dropout"]) + 0.1)),
        "loss": trial.suggest_categorical("loss", loss_candidates),
        "focal_gamma": trial.suggest_float("focal_gamma", 1.0, 3.0) if selected_loss == "focal" else 2.0,
        "learning_rate": trial.suggest_float("learning_rate", max(1e-4, float(base["learning_rate"]) / 3), min(2e-3, float(base["learning_rate"]) * 3), log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "max_epochs": 70,
        "patience": 9,
    }


def focused_search(device: str = "cuda", force: bool = False) -> dict[str, Any]:
    protocol, v4_protocol, data = _load()
    artifact = safe_v5_1_root(ROOT / "artifacts/v5_1/oulad")
    screen = screen_components(device, force=False)
    output_path = artifact / "focused_search.json"
    fingerprint = protocol_hash("oulad")
    if not force and output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") == fingerprint and previous.get("status") == "COMPLETE":
            return previous
    outer_fold = int(protocol["search"]["round_b_outer_fold"])
    splits = _inner_splits(data, outer_fold, v4_protocol)
    screen_result = screen["results"][0]
    base = dict(screen_result["architecture"]["selected"]["config"])
    augmentation = str(screen_result["augmentation"]["selected"])
    selected_loss = str(screen_result["loss"]["selected"])
    spec = _pretraining_spec(screen, 0)
    states = _build_inner_pretraining_states(data, splits, base, spec, (3407,), device)
    storage = f"sqlite:///{(artifact / 'optuna.db').as_posix()}"
    study_name = "oulad-v5.1-focused-outer-0"
    existing_names = {
        summary.study_name for summary in optuna.study.get_all_study_summaries(storage=storage)
    }
    existing_count = (
        len(optuna.load_study(study_name=study_name, storage=storage).trials)
        if study_name in existing_names
        else 0
    )
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=int(protocol["search"]["sampler_seed"]) + existing_count
        ),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )

    def objective(trial: optuna.Trial) -> float:
        config = _sample_focused(trial, base, selected_loss)
        duplicate = next(
            (
                earlier
                for earlier in study.trials
                if earlier.number < trial.number
                and earlier.state == optuna.trial.TrialState.COMPLETE
                and earlier.params == trial.params
            ),
            None,
        )
        if duplicate is not None:
            trial.set_user_attr("duplicate_of_trial", duplicate.number)
            raise optuna.TrialPruned()
        score, rows = _score_config(
            data,
            splits,
            config,
            (3407,),
            device,
            augmentation=augmentation,
            max_epochs=70,
            initial_states=states,
        )
        trial.set_user_attr("config", config)
        trial.set_user_attr("augmentation", augmentation)
        trial.set_user_attr(
            "pretraining",
            None if spec is None else {"mask_fraction": spec[0], "epochs": spec[1]},
        )
        trial.set_user_attr(
            "selected_epoch_median", int(np.median([row["selected_epoch"] for row in rows]))
        )
        trial.set_user_attr("runtime_seconds", float(sum(row["runtime_seconds"] for row in rows)))
        trial.set_user_attr(
            "gpu_peak_memory_bytes", int(max(row["gpu_peak_memory_bytes"] for row in rows))
        )
        for step, row in enumerate(rows):
            trial.report(float(row["macro_f1"]), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return score

    initial = int(protocol["search"]["round_b_trials_initial_total"])
    maximum = int(protocol["search"]["round_b_trials_max_total"])
    while _unique_trial_count(study) < initial:
        study.optimize(objective, n_trials=1, catch=(RuntimeError, ValueError))
    extend = float(study.best_value) > 0.8300 and _best_trial_is_recent(
        study, int(protocol["search"]["round_b_recent_improvement_window"])
    )
    while extend and _unique_trial_count(study) < maximum:
        study.optimize(objective, n_trials=1, catch=(RuntimeError, ValueError))
    stop_reason = (
        "stage_gate_extended_to_24_total_trials"
        if extend
        else "stage_gate_stopped_at_16_total_trials"
    )
    top = sorted(
        _unique_completed_trials(study), key=lambda trial: float(trial.value), reverse=True
    )[: int(protocol["search"]["round_b_top_configs_to_confirm"])]
    confirmation_states = _build_inner_pretraining_states(
        data, splits, base, spec, SCREENING_SEEDS, device
    )
    confirmations = []
    for trial in top:
        score, rows = _score_config(
            data,
            splits,
            dict(trial.user_attrs["config"]),
            SCREENING_SEEDS,
            device,
            augmentation=augmentation,
            max_epochs=70,
            initial_states=confirmation_states,
        )
        confirmations.append(
            {
                "trial": trial.number,
                "macro_f1_mean": score,
                "macro_f1_std": float(np.std([row["macro_f1"] for row in rows])),
                "worst_score": float(min(row["macro_f1"] for row in rows)),
                "fixed_epochs": int(np.median([row["selected_epoch"] for row in rows])),
                "config": trial.user_attrs["config"],
                "augmentation": augmentation,
                "pretraining": trial.user_attrs["pretraining"],
            }
        )
    selected = max(confirmations, key=lambda row: (row["macro_f1_mean"], row["worst_score"]))
    fold_result = {
        "outer_fold": outer_fold,
        "unique_trials": _unique_trial_count(study),
        "trial_rows": len(study.trials),
        "completed_unique_trials": len(_unique_completed_trials(study)),
        "stop_reason": stop_reason,
        "top_confirmations": confirmations,
        "selected": selected,
    }
    result = {
        "status": "COMPLETE",
        "fingerprint": fingerprint,
        "outer_results_used": False,
        "selection_outer_fold": outer_fold,
        "results": [fold_result],
        "selected": selected,
        "selected_configuration_reused_across_all_outer_folds": True,
        "future_accessed": False,
    }
    atomic_write_json(output_path, result)
    return result


def _threshold_and_epochs(
    data,
    splits,
    config: dict[str, Any],
    augmentation: str,
    states: dict[tuple[int, int], dict[str, torch.Tensor]],
    device: str,
) -> tuple[dict[str, float], int]:
    probabilities = []
    targets = []
    epochs = []
    for inner_fold, (train_index, validation_index) in enumerate(splits):
        fit = fit_oulad_model_v5_1(
            data,
            train_index,
            validation_index,
            config=config,
            seed=3407,
            augmentation=augmentation,
            device_name=device,
            initial_temporal_state=states.get((inner_fold, 3407)),
        )
        probabilities.append(fit.probability)
        targets.append(data.y[validation_index])
        epochs.append(fit.selected_epoch)
    return choose_threshold(np.concatenate(targets), np.concatenate(probabilities)), max(1, int(np.median(epochs)))


def _full_pretraining_state(
    data,
    train_index: np.ndarray,
    config: dict[str, Any],
    spec: dict[str, Any] | None,
    seed: int,
    device: str,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any] | None]:
    if spec is None:
        return None, None
    train = prepare_oulad_inputs(data, train_index, train_index)
    result = fit_masked_week_pretraining(
        train,
        dynamic_channel_order=data.dynamic_channel_order,
        config=config,
        seed=seed,
        epochs=int(spec["epochs"]),
        mask_fraction=float(spec["mask_fraction"]),
        device_name=device,
    )
    return result.temporal_state_dict, {
        "checkpoint_sha256": result.checkpoint_sha256,
        "runtime_seconds": result.runtime_seconds,
        "masked_values": result.masked_values,
        "replay_max_abs_difference": result.replay_max_abs_difference,
    }


def _reuse_v5_oulad_baselines(protocol: dict[str, Any], artifact) -> dict[str, Any]:
    v5_protocol = yaml.safe_load((ROOT / "configs/oulad_v5.yaml").read_text(encoding="utf-8"))
    source_pairs = {
        "processed_manifest": "processed_manifest",
        "source_split_manifest": "split_manifest",
        "sequence": "sequence",
        "full_aggregate_oracle": "aggregate",
        "target_table": "target_table",
    }
    checks = {}
    for current_name, v5_name in source_pairs.items():
        current = protocol["sources"][current_name]
        frozen = v5_protocol["sources"][v5_name]
        matches = current["path"] == frozen["path"] and current["sha256"] == frozen["sha256"]
        checks[current_name] = matches
    split_path = ROOT / protocol["splits"]["outer_manifest"]
    checks["outer_manifest"] = (
        sha256_file(split_path) == protocol["splits"]["outer_manifest_sha256"]
    )
    if not all(checks.values()):
        raise RuntimeError("V5 OULAD baseline reuse contract or checksum changed")
    metrics_path = ROOT / "artifacts/v5/oulad/final_metrics.csv"
    rows = pd.read_csv(metrics_path)
    reused = rows[rows.candidate.isin(["xgboost", "mlp", "logistic_regression", "hist_gradient_boosting"])]
    result = {
        "status": "REUSED_IMMUTABLE_V5_EVIDENCE",
        "contract_checks": checks,
        "source_metrics": metrics_path.relative_to(ROOT).as_posix(),
        "source_metrics_sha256": sha256_file(metrics_path),
        "rows": reused.to_dict(orient="records"),
    }
    atomic_write_json(artifact / "reused_v5_baselines.json", result)
    return result


def final_evaluation(device: str = "cuda", force: bool = False) -> dict[str, Any]:
    protocol, v4_protocol, data = _load()
    if set(data.development_manifest.role) != {"historical_development"}:
        raise RuntimeError("A future OULAD role entered V5.1")
    artifact = safe_v5_1_root(ROOT / "artifacts/v5_1/oulad")
    search = focused_search(device, force=False)
    checkpoint_root = artifact / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    ablation_seed = int(protocol["evaluation"]["architecture_ablation_seed"])
    variants = {
        "cnn_bilstm_full": ("cnn_bilstm", FINAL_SEEDS, "original", True, None),
        "cnn_only": ("cnn_only", (ablation_seed,), "original", False, None),
        "bilstm_only": ("bilstm_only", (ablation_seed,), "original", False, None),
    }
    for outer_fold in range(int(protocol["splits"]["outer_folds"])):
        train_index, validation_index = data.v2.outer_indices(outer_fold)
        splits = _inner_splits(data, outer_fold, v4_protocol)
        selected = search["selected"]
        base_config = dict(selected["config"])
        augmentation = str(selected["augmentation"])
        pretraining_spec = selected["pretraining"]
        threshold_states = _build_inner_pretraining_states(
            data,
            splits,
            base_config,
            None
            if pretraining_spec is None
            else (float(pretraining_spec["mask_fraction"]), int(pretraining_spec["epochs"])),
            (3407,),
            device,
        )
        threshold, fixed_epochs = _threshold_and_epochs(
            data, splits, base_config, augmentation, threshold_states, device
        )
        selected_rows.append(
            {
                "outer_fold": outer_fold,
                "config": base_config,
                "augmentation": augmentation,
                "pretraining": pretraining_spec,
                "threshold": threshold,
                "fixed_epochs": fixed_epochs,
            }
        )
        pretraining_cache: dict[int, tuple[dict[str, torch.Tensor] | None, dict[str, Any] | None]] = {}
        for candidate, (variant, seeds, order, use_pretraining, fusion_override) in variants.items():
            for seed in seeds:
                config = {**base_config, "temporal_order": order}
                if fusion_override is not None:
                    config["fusion"] = fusion_override
                state = None
                pretraining_metadata = None
                if use_pretraining and pretraining_spec is not None:
                    if seed not in pretraining_cache:
                        pretraining_cache[seed] = _full_pretraining_state(
                            data, train_index, base_config, pretraining_spec, seed, device
                        )
                    state, pretraining_metadata = pretraining_cache[seed]
                fit = fit_oulad_model_v5_1(
                    data,
                    train_index,
                    validation_index,
                    config=config,
                    seed=seed,
                    augmentation=augmentation,
                    variant=variant,
                    fixed_epochs=fixed_epochs,
                    device_name=device,
                    initial_temporal_state=state,
                )
                checkpoint = checkpoint_root / f"{candidate}_outer_{outer_fold}_seed_{seed}.pt"
                torch.save(fit.state_dict, checkpoint)
                checkpoint_rows.append(
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
                        "gpu_peak_memory_bytes": fit.gpu_peak_memory_bytes,
                        "gate_statistics": fit.gate_statistics,
                        "attention_entropy_mean": fit.attention_entropy_mean,
                        "attention_padding_max": fit.attention_padding_max,
                        "branch_norm_means": fit.branch_norm_means,
                        "pretraining": pretraining_metadata,
                    }
                )
                curve_rows.extend(
                    {"candidate": candidate, "outer_fold": outer_fold, "seed": seed, **row}
                    for row in fit.history
                )
                for index, probability in zip(validation_index, fit.probability):
                    prediction_rows.append(
                        {
                            "record_id": str(data.base.record_ids[index]),
                            "id_student": int(data.groups[index]),
                            "code_module": str(data.base.cohort.iloc[index].code_module),
                            "outer_fold": outer_fold,
                            "candidate": candidate,
                            "seed": seed,
                            "target": int(data.y[index]),
                            "probability": float(probability),
                            "threshold": float(threshold["threshold"]),
                        }
                    )
    predictions = pd.DataFrame(prediction_rows)
    ensemble_rows = []
    metric_rows = []
    for (candidate, seed), frame in predictions.groupby(["candidate", "seed"]):
        metrics = binary_metrics_per_record_threshold(
            frame.target.to_numpy(), frame.probability.to_numpy(), frame.threshold.to_numpy()
        )
        metric_rows.append({"candidate": candidate, "seed": int(seed), **metrics})
    for candidate, frame in predictions.groupby("candidate"):
        ensemble = (
            frame.groupby(
                ["record_id", "id_student", "code_module", "outer_fold", "target", "threshold"],
                as_index=False,
            )
            .probability.mean()
        )
        ensemble["candidate"] = f"{candidate}_ensemble"
        ensemble["seed"] = -1
        ensemble_rows.append(ensemble[predictions.columns])
        metrics = binary_metrics_per_record_threshold(
            ensemble.target.to_numpy(), ensemble.probability.to_numpy(), ensemble.threshold.to_numpy()
        )
        metric_rows.append({"candidate": f"{candidate}_ensemble", "seed": -1, **metrics})
    predictions = pd.concat([predictions, *ensemble_rows], ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values("macro_f1", ascending=False)
    predictions.to_parquet(artifact / "oof_predictions.parquet", index=False)
    metrics.to_json(artifact / "final_metrics.json", orient="records", indent=2)
    pd.DataFrame(curve_rows).to_csv(artifact / "learning_curves.csv", index=False)
    atomic_write_json(artifact / "checkpoint_metadata.json", checkpoint_rows)
    atomic_write_json(artifact / "selected_configs.json", selected_rows)
    atomic_write_json(artifact / "protocol_snapshot.json", protocol)
    atomic_write_json(
        artifact / "run_state.json",
        {
            "status": "COMPLETE",
            "future_accessed": False,
            "records": int(len(data.y)),
            "fixed_seeds": list(FINAL_SEEDS),
        },
    )
    atomic_write_json(artifact / "artifact_checksums.json", build_checksum_manifest(artifact))
    full = metrics[metrics.candidate == "cnn_bilstm_full_ensemble"].iloc[0].to_dict()
    ml = _reuse_v5_oulad_baselines(protocol, artifact)
    atomic_write_json(artifact / "artifact_checksums.json", build_checksum_manifest(artifact))
    return {
        "status": "COMPLETE",
        "candidate": "cnn_bilstm_full_ensemble",
        "metrics": full,
        "ml": ml,
        "future_accessed": False,
    }


def run_oulad_v5_1(
    *, phase: str = "all", device: str = "cuda", force: bool = False
) -> dict[str, Any]:
    phases: dict[str, Callable[[], dict[str, Any]]] = {
        "architecture": lambda: architecture_screening(device, force),
        "screen": lambda: screen_components(device, force),
        "search": lambda: focused_search(device, force),
        "final": lambda: final_evaluation(device, force),
    }
    if phase in phases:
        return phases[phase]()
    if phase != "all":
        raise ValueError(phase)
    started = time.perf_counter()
    screen_components(device, force)
    focused_search(device, force)
    result = final_evaluation(device, force)
    result["total_runtime_seconds"] = time.perf_counter() - started
    return result


__all__ = [
    "anchor_config",
    "architecture_screening",
    "final_evaluation",
    "focused_search",
    "run_oulad_v5_1",
    "screen_components",
]
