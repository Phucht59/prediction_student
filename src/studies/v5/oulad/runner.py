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

from src.studies.oulad_v4.data import build_v4_inner_manifest, load_v4_data, manifest_indices

from ..common.artifacts import (
    atomic_write_json,
    build_checksum_manifest,
    result_fingerprint,
    safe_v5_root,
    verify_checksum_manifest,
)
from ..common.metrics import binary_metrics, binary_metrics_per_record_threshold
from ..common.protocol import (
    ROOT,
    load_json_yaml,
    load_study_protocol,
    protocol_fingerprint,
    sha256_file,
    verify_declared_sources,
)
from .training import choose_threshold, fit_oulad_model


AUGMENTATIONS = ["none", "event_thinning", "short_span_masking", "channel_dropout"]
VARIANTS = ["cnn_bilstm", "cnn_only", "bilstm_only"]


def _anchor_config() -> dict[str, Any]:
    return {
        "conv_channels": 24,
        "kernels": [3, 5],
        "lstm_hidden": 40,
        "lstm_layers": 1,
        "pooling": "masked_mean_max",
        "pooling_projection": 48,
        "aggregate_hidden": 80,
        "static_hidden": 24,
        "fusion_hidden": 48,
        "dropout": 0.25,
        "learning_rate": 5e-4,
        "weight_decay": 1e-5,
        "batch_size": 256,
        "positive_weight": "none",
        "loss": "weighted_bce",
        "focal_gamma": 2.0,
        "max_epochs": 60,
        "patience": 8,
        "gradient_clip": 1.0,
        "parameter_limit": 1_500_000,
    }


def _sample_config(trial: optuna.Trial) -> dict[str, Any]:
    loss = trial.suggest_categorical("loss", ["weighted_bce", "focal"])
    kernels_name = trial.suggest_categorical("kernels", ["3", "5", "3_5"])
    return {
        "conv_channels": trial.suggest_categorical("conv_channels", [16, 24, 32, 48]),
        "kernels": {"3": [3], "5": [5], "3_5": [3, 5]}[kernels_name],
        "lstm_hidden": trial.suggest_categorical("lstm_hidden", [24, 32, 40, 48, 64]),
        "lstm_layers": trial.suggest_categorical("lstm_layers", [1, 2]),
        "pooling": trial.suggest_categorical("pooling", ["masked_mean_max", "masked_attention"]),
        "pooling_projection": trial.suggest_categorical("pooling_projection", [32, 48, 64, 96]),
        "aggregate_hidden": trial.suggest_categorical("aggregate_hidden", [48, 64, 80, 96, 128]),
        "static_hidden": trial.suggest_categorical("static_hidden", [16, 24, 32, 48]),
        "fusion_hidden": trial.suggest_categorical("fusion_hidden", [32, 48, 64, 96]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "positive_weight": trial.suggest_categorical("positive_weight", ["none", "balanced"]),
        "loss": loss,
        "focal_gamma": trial.suggest_float("focal_gamma", 1.0, 3.0) if loss == "focal" else 2.0,
        "max_epochs": 80,
        "patience": 10,
        "gradient_clip": 1.0,
        "parameter_limit": 1_500_000,
    }


def _screen_augmentation(data, outer_fold: int, v4_protocol: dict[str, Any], device: str):
    inner = build_v4_inner_manifest(data, outer_fold, v4_protocol)
    rows: list[dict[str, Any]] = []
    seeds = load_study_protocol("oulad")["augmentation"]["screening_seeds"]
    for augmentation in AUGMENTATIONS:
        for seed in seeds:
            probabilities: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            audits: list[dict[str, object]] = []
            for inner_fold in sorted(inner.inner_fold.unique()):
                train, validation = manifest_indices(data.v2, inner, int(inner_fold))
                fit = fit_oulad_model(
                    data,
                    train,
                    validation,
                    config=_anchor_config(),
                    seed=int(seed) + int(inner_fold),
                    augmentation=augmentation,
                    device_name=device,
                )
                probabilities.append(fit.probability)
                targets.append(data.y[validation])
                audits.append(fit.augmentation_audit)
            threshold = choose_threshold(np.concatenate(targets), np.concatenate(probabilities))
            metric = binary_metrics(np.concatenate(targets), np.concatenate(probabilities), threshold["threshold"])
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "augmentation": augmentation,
                    "seed": int(seed),
                    "macro_f1": metric["macro_f1"],
                    "threshold": threshold["threshold"],
                    "changed_values": sum(int(audit["changed_values"]) for audit in audits),
                }
            )
    summary = pd.DataFrame(rows).groupby("augmentation").macro_f1.agg(["mean", "std"]).sort_values(
        ["mean", "std"], ascending=[False, True]
    )
    return str(summary.index[0]), rows


def _search(data, outer_fold: int, augmentation: str, protocol: dict[str, Any], v4_protocol: dict[str, Any], artifact: Path, device: str):
    study = optuna.create_study(
        study_name=f"oulad_v5_outer_{outer_fold}",
        storage=f"sqlite:///{(artifact / 'optuna.db').resolve().as_posix()}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(protocol["search"]["sampler_seed"]) + outer_fold),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1),
    )
    if not study.trials:
        anchor = _anchor_config()
        study.enqueue_trial(
            {
                "conv_channels": anchor["conv_channels"],
                "kernels": "3_5",
                "lstm_hidden": anchor["lstm_hidden"],
                "lstm_layers": anchor["lstm_layers"],
                "pooling": anchor["pooling"],
                "pooling_projection": anchor["pooling_projection"],
                "aggregate_hidden": anchor["aggregate_hidden"],
                "static_hidden": anchor["static_hidden"],
                "fusion_hidden": anchor["fusion_hidden"],
                "dropout": anchor["dropout"],
                "learning_rate": anchor["learning_rate"],
                "weight_decay": anchor["weight_decay"],
                "batch_size": anchor["batch_size"],
                "positive_weight": anchor["positive_weight"],
                "loss": anchor["loss"],
            }
        )
    inner = build_v4_inner_manifest(data, outer_fold, v4_protocol)

    def objective(trial: optuna.Trial) -> float:
        config = _sample_config(trial)
        probabilities: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        epochs: list[int] = []
        parameter_count = 0
        for step, inner_fold in enumerate(sorted(inner.inner_fold.unique())):
            train, validation = manifest_indices(data.v2, inner, int(inner_fold))
            fit = fit_oulad_model(
                data,
                train,
                validation,
                config=config,
                seed=int(protocol["search"]["sampler_seed"]) + outer_fold * 100 + int(inner_fold),
                augmentation=augmentation,
                device_name=device,
            )
            probabilities.append(fit.probability)
            targets.append(data.y[validation])
            epochs.append(fit.selected_epoch)
            parameter_count = fit.parameter_count
            partial = choose_threshold(np.concatenate(targets), np.concatenate(probabilities))["inner_macro_f1"]
            trial.report(float(partial), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        threshold = choose_threshold(np.concatenate(targets), np.concatenate(probabilities))
        trial.set_user_attr("config", json.dumps(config, sort_keys=True))
        trial.set_user_attr("threshold", threshold)
        trial.set_user_attr("epochs", epochs)
        trial.set_user_attr("parameter_count", parameter_count)
        return float(threshold["inner_macro_f1"])

    target = int(protocol["search"]["cnn_bilstm_trials_per_outer_fold"])
    complete = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    if complete < target:
        study.optimize(objective, n_trials=target - complete, catch=(RuntimeError,), show_progress_bar=False)
    best = study.best_trial
    rows = [
        {
            "outer_fold": outer_fold,
            "trial": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "parameters": json.dumps(trial.params, sort_keys=True),
            "parameter_count": trial.user_attrs.get("parameter_count"),
        }
        for trial in study.trials
    ]
    return (
        json.loads(best.user_attrs["config"]),
        dict(best.user_attrs["threshold"]),
        max(1, int(round(np.median(best.user_attrs["epochs"])))),
        rows,
    )


def _cache_path(artifact: Path, fold: int, fingerprint: str) -> Path:
    return artifact / "runtime_cache" / f"outer_{fold}_{fingerprint[:16]}.joblib"


def _load_cache(path: Path, fingerprint: str):
    meta = path.with_suffix(".json")
    if not path.is_file() or not meta.is_file():
        return None
    value = json.loads(meta.read_text(encoding="utf-8"))
    if value.get("fingerprint") != fingerprint or value.get("sha256") != sha256_file(path):
        return None
    return joblib.load(path)


def _save_cache(path: Path, fingerprint: str, value: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)
    atomic_write_json(path.with_suffix(".json"), {"fingerprint": fingerprint, "sha256": sha256_file(path)})


def _outer_fold(data, fold: int, protocol: dict[str, Any], v4_protocol: dict[str, Any], artifact: Path, device: str):
    augmentation, augmentation_rows = _screen_augmentation(data, fold, v4_protocol, device)
    config, threshold, epochs, trials = _search(
        data, fold, augmentation, protocol, v4_protocol, artifact, device
    )
    train, validation = data.v2.outer_indices(fold)
    prediction_rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in protocol["evaluation"]["seeds"]:
            fit = fit_oulad_model(
                data,
                train,
                validation,
                config=config,
                seed=int(seed),
                augmentation=augmentation,
                variant=variant,
                fixed_epochs=epochs,
                device_name=device,
            )
            checkpoint_path = artifact / "checkpoints" / f"{variant}_outer_{fold}_seed_{seed}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(fit.state_dict, checkpoint_path)
            checkpoints.append(
                {
                    "variant": variant,
                    "outer_fold": fold,
                    "seed": int(seed),
                    "path": checkpoint_path.relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(checkpoint_path),
                    "state_dict_sha256": fit.checkpoint_sha256,
                    "replay_max_abs_difference": fit.replay_max_abs_difference,
                    "parameter_count": fit.parameter_count,
                    "runtime_seconds": fit.runtime_seconds,
                    "attention_padding_max": fit.attention_padding_max,
                    "gate_means": fit.gate_means,
                }
            )
            curves.extend(
                {"variant": variant, "outer_fold": fold, "seed": int(seed), **row} for row in fit.history
            )
            for index, probability in zip(validation, fit.probability):
                prediction_rows.append(
                    {
                        "record_id": str(data.base.record_ids[index]),
                        "id_student": int(data.groups[index]),
                        "code_module": str(data.base.cohort.iloc[index].code_module),
                        "outer_fold": fold,
                        "candidate": variant,
                        "seed": int(seed),
                        "target": int(data.y[index]),
                        "probability": float(probability),
                        "threshold": float(threshold["threshold"]),
                    }
                )
    return {
        "predictions": prediction_rows,
        "curves": curves,
        "checkpoints": checkpoints,
        "augmentation_rows": augmentation_rows,
        "trials": trials,
        "selected": {
            "outer_fold": fold,
            "augmentation": augmentation,
            "config": config,
            "threshold": threshold,
            "fixed_epochs": epochs,
        },
    }


def _summaries(predictions: pd.DataFrame):
    rows: list[dict[str, Any]] = []
    for (candidate, seed), frame in predictions.groupby(["candidate", "seed"]):
        metric = binary_metrics_per_record_threshold(
            frame.target.to_numpy(),
            frame.probability.to_numpy(),
            frame.threshold.to_numpy(),
        )
        rows.append({"candidate": candidate, "seed": int(seed), "evidence_source": "V5", **{k: v for k, v in metric.items() if k != "confusion_matrix"}})
    deep = predictions[predictions.candidate == "cnn_bilstm"]
    ensemble = (
        deep.groupby(["record_id", "id_student", "code_module", "outer_fold", "target", "threshold"], as_index=False)
        .probability.mean()
    )
    ensemble["candidate"] = "cnn_bilstm_ensemble"
    ensemble["seed"] = -1
    predictions = pd.concat([predictions, ensemble[predictions.columns]], ignore_index=True)
    metric = binary_metrics_per_record_threshold(
        ensemble.target.to_numpy(),
        ensemble.probability.to_numpy(),
        ensemble.threshold.to_numpy(),
    )
    rows.append({"candidate": "cnn_bilstm_ensemble", "seed": -1, "evidence_source": "V5", **{k: v for k, v in metric.items() if k != "confusion_matrix"}})
    return predictions, pd.DataFrame(rows)


def _v4_comparators() -> pd.DataFrame:
    path = ROOT / "artifacts" / "oulad" / "v4" / "oulad-v4-f2-scientific-20260716-v1" / "metrics_summary.csv"
    source = pd.read_csv(path).set_index("candidate_id")
    mapping = {
        "logistic_regression": "V4-LR",
        "hist_gradient_boosting": "V4-HGB",
        "xgboost": "V4-XGB-ENS",
        "mlp": "V4-A0-ENS",
    }
    rows = []
    for name, candidate in mapping.items():
        value = source.loc[candidate]
        rows.append(
            {
                "candidate": name,
                "seed": -1,
                "evidence_source": f"immutable V4 comparator {candidate}; same F2 data contract",
                "records": np.nan,
                "threshold": np.nan,
                "macro_f1": float(value.macro_f1),
                "balanced_accuracy": float(value.balanced_accuracy),
                "at_risk_precision": float(value.at_risk_precision),
                "at_risk_recall": float(value.at_risk_recall),
                "at_risk_f1": float(value.at_risk_f1),
                "pr_auc": float(value.pr_auc),
                "brier": float(value.brier),
                "nll": float(value.nll),
                "ece": float(value.ece),
            }
        )
    return pd.DataFrame(rows)


def run(*, device: str = "cuda", force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    protocol = load_study_protocol("oulad")
    source_audit = verify_declared_sources(protocol)
    if any(row["status"] != "PASS" for row in source_audit):
        raise RuntimeError("OULAD V5 source audit failed")
    v4_protocol = load_json_yaml(ROOT / "configs" / "oulad_v4_protocol.yaml")
    data = load_v4_data(ROOT / "data" / "processed" / "study_c_oulad", v4_protocol)
    if set(data.development_manifest.role) != {"historical_development"}:
        raise RuntimeError("A future role entered OULAD V5")
    artifact = safe_v5_root(ROOT / "artifacts" / "v5" / "oulad")
    fingerprint = result_fingerprint(
        protocol_hash=protocol_fingerprint("oulad"),
        source_hashes={row["source"]: row["expected_sha256"] for row in source_audit},
        config={"device": device},
    )
    state_path = artifact / "run_state.json"
    checksum_path = artifact / "artifact_checksums.json"
    if not force and state_path.is_file() and checksum_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        manifest = json.loads(checksum_path.read_text(encoding="utf-8"))
        if state.get("status") == "COMPLETE" and state.get("fingerprint") == fingerprint and verify_checksum_manifest(artifact, manifest):
            return {"study": "oulad", "status": "SKIPPED_VALID_CACHE", "artifact": str(artifact)}
    atomic_write_json(state_path, {"study": "oulad", "status": "RUNNING", "fingerprint": fingerprint})
    fold_results = []
    split_rows = []
    for fold in range(int(protocol["data_contract"]["outer_folds"])):
        train, validation = data.v2.outer_indices(fold)
        split_rows.extend(
            {"outer_fold": fold, "role": role, "record_id": str(data.base.record_ids[index]), "id_student": int(data.groups[index])}
            for role, indices in (("outer_train", train), ("outer_validation", validation))
            for index in indices
        )
        cache = _cache_path(artifact, fold, fingerprint)
        result = None if force else _load_cache(cache, fingerprint)
        if result is None:
            result = _outer_fold(data, fold, protocol, v4_protocol, artifact, device)
            _save_cache(cache, fingerprint, result)
        fold_results.append(result)
        atomic_write_json(state_path, {"study": "oulad", "status": "RUNNING", "fingerprint": fingerprint, "completed_outer_folds": fold + 1})
    predictions = pd.DataFrame([row for result in fold_results for row in result["predictions"]])
    predictions, v5_metrics = _summaries(predictions)
    final_metrics = pd.concat([v5_metrics, _v4_comparators()], ignore_index=True).sort_values("macro_f1", ascending=False)
    predictions.to_parquet(artifact / "oof_predictions.parquet", index=False)
    final_metrics.to_csv(artifact / "final_metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(artifact / "split_manifest.csv", index=False)
    pd.DataFrame([row for result in fold_results for row in result["curves"]]).to_csv(artifact / "learning_curves.csv", index=False)
    pd.DataFrame([row for result in fold_results for row in result["augmentation_rows"]]).to_csv(artifact / "augmentation_comparison.csv", index=False)
    pd.DataFrame([row for result in fold_results for row in result["trials"]]).to_csv(artifact / "search_trials.csv", index=False)
    atomic_write_json(artifact / "selected_configs.json", [result["selected"] for result in fold_results])
    atomic_write_json(artifact / "checkpoint_metadata.json", [row for result in fold_results for row in result["checkpoints"]])
    atomic_write_json(artifact / "source_manifest.json", source_audit)
    atomic_write_json(artifact / "protocol_snapshot.json", protocol)
    best = final_metrics.iloc[0]
    thesis = final_metrics[final_metrics.candidate == "cnn_bilstm_ensemble"].iloc[0]
    registry = {
        "final_thesis_model": "CNN-BiLSTM V5 Ensemble",
        "final_overall_model": str(best.candidate),
        "reason": "Highest valid grouped historical-development OOF Macro-F1; no future benchmark was opened.",
        "metrics": {"thesis_macro_f1": float(thesis.macro_f1), "overall_macro_f1": float(best.macro_f1)},
        "limitations": ["Historical grouped development OOF only", "Future benchmark remains locked", "Recommendation effectiveness not established"],
        "artifact_paths": ["final_metrics.csv", "oof_predictions.parquet", "selected_configs.json"],
        "checkpoint_hashes": [row["sha256"] for result in fold_results for row in result["checkpoints"]],
    }
    atomic_write_json(artifact / "model_registry.json", registry)
    atomic_write_json(state_path, {"study": "oulad", "status": "COMPLETE", "fingerprint": fingerprint, "runtime_seconds": time.perf_counter() - started, "future_accessed": False})
    manifest = build_checksum_manifest(artifact)
    atomic_write_json(checksum_path, manifest)
    return {"study": "oulad", "status": "COMPLETE", "artifact": str(artifact), "registry": registry}


__all__ = ["AUGMENTATIONS", "VARIANTS", "run"]
