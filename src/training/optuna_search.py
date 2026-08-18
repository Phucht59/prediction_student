"""Resumable, inner-only Phase 3 Optuna VNext execution."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.importance import get_param_importances
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.pipelines import oulad
from src.training.config_authority import (
    architecture_metadata,
    load_config_authority,
    resolved_deep_config,
)
from src.training.control import (
    canonical_json,
    select_refit_epoch,
    select_research_threshold,
    stable_hash,
)


ROOT = Path(__file__).resolve().parents[2]
PHASE2_COMMIT = "14b3df97b16aa7abc2d2231594dd8145fe014894"
AUTHORITY_PATH = ROOT / "configs" / "registry" / "oulad_unified_stage_aware_v2.yaml"
OUT = ROOT / "artifacts" / "audit" / "phase3"
RUNTIME = OUT / "runtime"
LOGS = OUT / "logs"
OPTUNA_DIR = OUT / "optuna"
STATUS_PATH = RUNTIME / "phase3_status.json"
RUNNING = RUNTIME / "PHASE3_RUNNING"
COMPLETE = RUNTIME / "PHASE3_COMPLETE"
FAILED = RUNTIME / "PHASE3_FAILED"
SEARCH_SEED = 42
STABILITY_SEEDS = (1201, 2026)
TRIALS_PER_OUTER_FOLD = 24
N_STARTUP_TRIALS = 6
PRUNING_WARMUP_EPOCHS = 3
MAX_EPOCHS = 15
INNER_FOLDS = 2
ARCHITECTURE_PARAMETER_COUNT = 150202
STAGES = tuple(oulad.STAGES)
STAGE_LABEL = {
    STAGES[0]: "20",
    STAGES[1]: "35",
    STAGES[2]: "50",
    STAGES[3]: "75",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, default=str) + "\n")


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def phase2_is_ancestor() -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PHASE2_COMMIT, "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def prepare_directories() -> None:
    for path in (OUT, RUNTIME, LOGS, OPTUNA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def status_payload(**updates: Any) -> dict[str, Any]:
    if STATUS_PATH.is_file():
        current = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    else:
        current = {
            "state": "PENDING",
            "started_at": None,
            "finished_at": None,
            "current_stage": "preconditions",
            "completed_trials": 0,
            "pruned_trials": 0,
            "failed_trials": 0,
            "oom_trials": 0,
            "last_successful_outer_fold": None,
            "exit_code": None,
            "pid": os.getpid(),
        }
    current.update(updates)
    write_json(STATUS_PATH, current)
    return current


def set_sentinel(state: str, detail: dict[str, Any] | None = None) -> None:
    prepare_directories()
    for path in (RUNNING, COMPLETE, FAILED):
        if path.exists():
            path.unlink()
    target = {"RUNNING": RUNNING, "COMPLETE": COMPLETE, "FAILED": FAILED}[state]
    target.write_text(
        json.dumps({"state": state, "at": utc_now(), **(detail or {})}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def control_config() -> dict[str, Any]:
    authority = load_config_authority(AUTHORITY_PATH)
    training = authority["training"]
    loss = authority["loss"]
    return {
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "dropout": float(training["dropout"]),
        "batch_size": int(training["batch_size"]),
        "loss_policy": "weighted_bce",
        "pos_weight_strategy": "full_ratio",
        "survival_weight": float(loss["survival_weight"]),
        "outcome_weight": float(loss["outcome_weight"]),
    }


def sample_trial_config(trial: optuna.Trial) -> dict[str, Any]:
    loss_policy = trial.suggest_categorical(
        "loss_policy", ["standard_bce", "weighted_bce"]
    )
    strategy = (
        trial.suggest_categorical(
            "pos_weight_strategy", ["sqrt_ratio", "full_ratio"]
        )
        if loss_policy == "weighted_bce"
        else "not_applicable"
    )
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-4, 2e-3, log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-8, 5e-4, log=True
        ),
        "dropout": trial.suggest_float("dropout", 0.10, 0.35),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "loss_policy": loss_policy,
        "pos_weight_strategy": strategy,
        "survival_weight": trial.suggest_categorical(
            "survival_weight", [0.0, 0.10, 0.15, 0.20]
        ),
        "outcome_weight": trial.suggest_categorical(
            "outcome_weight", [0.0, 0.10, 0.15, 0.20]
        ),
    }


def search_space_manifest() -> dict[str, Any]:
    return {
        "learning_rate": {"type": "float", "low": 1e-4, "high": 2e-3, "log": True},
        "weight_decay": {"type": "float", "low": 1e-8, "high": 5e-4, "log": True},
        "dropout": {"type": "float", "low": 0.10, "high": 0.35},
        "batch_size": {"type": "categorical", "choices": [128, 256]},
        "loss_policy": {
            "type": "categorical",
            "choices": ["standard_bce", "weighted_bce"],
        },
        "pos_weight_strategy": {
            "type": "conditional_categorical",
            "when": "loss_policy == weighted_bce",
            "choices": ["sqrt_ratio", "full_ratio"],
        },
        "survival_weight": {
            "type": "categorical",
            "choices": [0.0, 0.10, 0.15, 0.20],
        },
        "outcome_weight": {
            "type": "categorical",
            "choices": [0.0, 0.10, 0.15, 0.20],
        },
        "frozen": {
            "architecture": "oulad_unified_stage_aware_v2",
            "branch_dropout": 0.1,
            "optimizer": "AdamW",
            "scheduler": None,
            "max_epochs": MAX_EPOCHS,
            "checkpoint_policy": "minimize_mean_stage_validation_nll",
            "pretraining_executed": False,
            "amp": False,
        },
    }


def architecture_contract() -> dict[str, Any]:
    authority = load_config_authority(AUTHORITY_PATH)
    model = oulad._deep_model(
        "cnn_bilstm", 165, 13, resolved_deep_config(authority)
    )
    metadata = architecture_metadata(
        model, authority=authority, aggregate_dim=165, static_dim=13
    )
    if metadata["parameter_count"] != ARCHITECTURE_PARAMETER_COUNT:
        raise RuntimeError("authoritative parameter count changed")
    if authority["pretraining"] != {
        "requested": False,
        "executed": False,
        "checkpoint": None,
        "strategy": None,
    }:
        raise RuntimeError("pretraining provenance is not frozen off")
    return metadata


def validate_preconditions() -> dict[str, Any]:
    prepare_directories()
    if not phase2_is_ancestor():
        raise RuntimeError("Phase 2 commit is not an ancestor of HEAD")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Phase 3 requires exactly one available CUDA training GPU")
    authority = load_config_authority(AUTHORITY_PATH)
    if int(authority["training"]["max_epochs"]) != MAX_EPOCHS:
        raise RuntimeError("authoritative epoch cap is not 15")
    if authority["training"]["monitor"] != "mean_stage_validation_nll":
        raise RuntimeError("authoritative checkpoint monitor changed")
    phase2_gate = json.loads(
        (ROOT / "artifacts" / "audit" / "phase2" / "phase2_gate.json").read_text(
            encoding="utf-8"
        )
    )
    if phase2_gate.get("status") != "PASS":
        raise RuntimeError("Phase 2 gate is not PASS")
    contract = architecture_contract()
    result = {
        "status": "PASS",
        "phase2_commit_is_ancestor": True,
        "commit_sha": git_commit(),
        "device": torch.cuda.get_device_name(0),
        "gpu_count": 1,
        "concurrency": 1,
        "precision": "FP32",
        "amp": False,
        "architecture_hash": contract["architecture_hash"],
        "parameter_count": contract["parameter_count"],
        "pretraining_executed": False,
        "outer_labels_available_to_runner": False,
        "max_epochs": MAX_EPOCHS,
        "checkpoint_policy": "minimize_mean_stage_validation_nll",
    }
    write_json(OUT / "precondition_validation.json", result)
    write_json(OUT / "search_space.json", search_space_manifest())
    return result


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        selected = (probabilities >= low) & (
            probabilities < (high if high < 1 else high + 1e-9)
        )
        if selected.any():
            result += selected.mean() * abs(
                probabilities[selected].mean() - labels[selected].mean()
            )
    return float(result)


def _risk_loss(
    labels: np.ndarray, config: dict[str, Any], device: torch.device
) -> tuple[nn.Module, float]:
    negatives = int((labels == 0).sum())
    positives = int((labels == 1).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("invalid training partition class prevalence")
    ratio = negatives / positives
    if config["loss_policy"] == "standard_bce":
        weight = 1.0
    elif config["pos_weight_strategy"] == "sqrt_ratio":
        weight = math.sqrt(ratio)
    elif config["pos_weight_strategy"] == "full_ratio":
        weight = ratio
    else:
        raise ValueError("invalid loss/positive-weight policy")
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("invalid positive weight")
    return (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(weight, device=device), reduction="none"
        ),
        float(weight),
    )


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    authority = load_config_authority(AUTHORITY_PATH)
    result = resolved_deep_config(authority)
    for field in (
        "learning_rate",
        "weight_decay",
        "dropout",
        "batch_size",
        "survival_weight",
        "outcome_weight",
    ):
        result[field] = config[field]
    result["max_epochs"] = MAX_EPOCHS
    result["patience"] = 5
    return result


@dataclass
class InnerResult:
    frame: pd.DataFrame
    probability: np.ndarray
    selected_epoch: int
    epochs_trained: int
    best_mean_stage_nll: float
    pos_weight: float
    architecture_hash: str
    parameter_count: int


def _train_inner(
    train: tuple,
    validation: tuple,
    *,
    config: dict[str, Any],
    training_seed: int,
    inner_fold: int,
    trial: optuna.Trial | None,
    max_epochs: int,
    checkpoint_path: Path | None = None,
) -> InnerResult:
    torch.manual_seed(training_seed)
    torch.cuda.manual_seed_all(training_seed)
    np.random.seed(training_seed)
    frame, sequence, length, mask, aggregate, labels, sample_weight = train
    val_frame, val_sequence, val_length, val_mask, val_aggregate, val_labels, _ = (
        validation
    )
    preprocessor = oulad._DeepPreprocessor().fit(frame, aggregate)
    aggregate, static = preprocessor.transform(frame, aggregate)
    val_aggregate, val_static = preprocessor.transform(val_frame, val_aggregate)
    model_config = _model_config(config)
    device = torch.device("cuda")
    model = oulad._deep_model(
        "cnn_bilstm", aggregate.shape[1], static.shape[1], model_config
    ).to(device)
    authority = load_config_authority(AUTHORITY_PATH)
    metadata = architecture_metadata(
        model,
        authority=authority,
        aggregate_dim=int(aggregate.shape[1]),
        static_dim=int(static.shape[1]),
    )
    if metadata["parameter_count"] != ARCHITECTURE_PARAMETER_COUNT:
        raise RuntimeError("architecture parameter count mismatch")
    expected_hash = architecture_contract()["architecture_hash"]
    if metadata["architecture_hash"] != expected_hash:
        raise RuntimeError("architecture hash mismatch")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    risk_loss, positive_weight = _risk_loss(labels, config, device)
    dataset = TensorDataset(
        torch.from_numpy(sequence),
        torch.from_numpy(length.astype(np.int64)),
        torch.from_numpy(mask.astype(np.float32)),
        torch.from_numpy(aggregate),
        torch.from_numpy(static),
        torch.from_numpy(labels.astype(np.float32)),
        torch.from_numpy(sample_weight.astype(np.float32)),
        torch.from_numpy(frame.outcome_aux.to_numpy(dtype=np.int64)),
        torch.from_numpy(frame.cutoff_day.to_numpy(dtype=np.int64)),
        torch.from_numpy(frame.module_presentation_length.to_numpy(dtype=np.int64)),
        torch.from_numpy(
            frame.date_unregistration.fillna(-1).to_numpy(dtype=np.int64)
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(training_seed),
    )
    best_nll = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    epochs_trained = 0
    for epoch in range(1, max_epochs + 1):
        epochs_trained = epoch
        model.train()
        for batch in loader:
            (
                batch_sequence,
                batch_length,
                batch_mask,
                batch_aggregate,
                batch_static,
                target,
                weight,
                outcome,
                cutoff,
                course_end,
                unregistration,
            ) = (value.to(device) for value in batch)
            optimizer.zero_grad()
            output = model(
                batch_sequence,
                batch_length,
                batch_mask,
                batch_aggregate,
                batch_static,
            )
            loss, _ = oulad._multitask_loss(
                output,
                target,
                weight,
                outcome,
                cutoff,
                course_end,
                unregistration,
                risk_loss,
                survival_weight=float(config["survival_weight"]),
                outcome_weight=float(config["outcome_weight"]),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        probability = oulad._predict_deep(
            model,
            val_sequence,
            val_length,
            val_mask,
            val_aggregate,
            val_static,
            "cnn_bilstm",
            device,
        )
        mean_stage_nll = oulad._mean_stage_nll(
            val_frame, val_labels, probability
        )
        if not math.isfinite(mean_stage_nll):
            raise FloatingPointError("non-finite validation NLL")
        if mean_stage_nll < best_nll - 1e-6:
            best_nll = mean_stage_nll
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
        if trial is not None:
            step = inner_fold * MAX_EPOCHS + epoch
            trial.report(-mean_stage_nll, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"median pruner at inner={inner_fold}, epoch={epoch}"
                )
        if wait >= 5:
            break
    if best_state is None or best_epoch == 0:
        raise RuntimeError("no valid checkpoint was selected")
    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "selected_epoch": best_epoch,
                "architecture_hash": metadata["architecture_hash"],
                "parameter_count": metadata["parameter_count"],
                "smoke_only": True,
            },
            checkpoint_path,
        )
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if loaded["selected_epoch"] != best_epoch:
            raise RuntimeError("smoke checkpoint identity validation failed")
    probability = oulad._predict_deep(
        model,
        val_sequence,
        val_length,
        val_mask,
        val_aggregate,
        val_static,
        "cnn_bilstm",
        device,
    )
    if not np.isfinite(probability).all():
        raise FloatingPointError("non-finite validation probabilities")
    if float(np.std(probability)) < 1e-8:
        raise ValueError("numerically collapsed constant predictions")
    result_frame = val_frame.loc[
        :, ["base_record_id", "id_student", "prediction_stage", "target"]
    ].copy()
    del model, optimizer, loader, dataset
    torch.cuda.empty_cache()
    return InnerResult(
        frame=result_frame,
        probability=probability,
        selected_epoch=best_epoch,
        epochs_trained=epochs_trained,
        best_mean_stage_nll=float(best_nll),
        pos_weight=positive_weight,
        architecture_hash=metadata["architecture_hash"],
        parameter_count=metadata["parameter_count"],
    )


def evaluate_oof(
    frames: list[pd.DataFrame], probabilities: list[np.ndarray]
) -> dict[str, Any]:
    frame = pd.concat(frames, ignore_index=True)
    probability = np.concatenate(probabilities)
    if len(frame) != len(probability):
        raise RuntimeError("OOF frame/probability alignment failure")
    frame["probability"] = probability
    stage_metrics: dict[str, dict[str, float]] = {}
    thresholds: dict[str, float] = {}
    for stage in STAGES:
        selected = frame.prediction_stage.eq(stage).to_numpy()
        labels = frame.loc[selected, "target"].to_numpy(dtype=int)
        values = np.clip(probability[selected], 1e-7, 1 - 1e-7)
        if len(labels) == 0 or len(np.unique(labels)) < 2:
            raise ValueError(f"invalid OOF labels for stage {stage}")
        threshold = select_research_threshold(labels, values)
        thresholds[stage] = float(threshold["threshold"])
        predicted = values >= threshold["threshold"]
        stage_metrics[stage] = {
            "macro_f1": float(f1_score(labels, predicted, average="macro")),
            "pr_auc": float(average_precision_score(labels, values)),
            "nll": float(log_loss(labels, values, labels=[0, 1])),
            "brier": float(np.mean((values - labels) ** 2)),
            "ece": _ece(labels, values),
            "roc_auc": float(roc_auc_score(labels, values)),
        }
    aggregate = {
        "mean_stage_macro_f1": float(
            np.mean([value["macro_f1"] for value in stage_metrics.values()])
        ),
        "worst_stage_macro_f1": float(
            np.min([value["macro_f1"] for value in stage_metrics.values()])
        ),
        "mean_stage_pr_auc": float(
            np.mean([value["pr_auc"] for value in stage_metrics.values()])
        ),
        "mean_stage_nll": float(
            np.mean([value["nll"] for value in stage_metrics.values()])
        ),
        "mean_stage_brier": float(
            np.mean([value["brier"] for value in stage_metrics.values()])
        ),
        "mean_stage_ece": float(
            np.mean([value["ece"] for value in stage_metrics.values()])
        ),
    }
    return {
        **aggregate,
        "stage_metrics": stage_metrics,
        "research_thresholds": thresholds,
        "oof_count": int(len(frame)),
    }


class OptunaSearchRunner:
    """Inner-CV runner whose public API intentionally has no outer labels."""

    def __init__(self, bundle: oulad.Bundle, outer_fold: int):
        self.bundle = bundle
        self.outer_fold = int(outer_fold)
        base = bundle.base[
            ["base_record_id", "id_student", "outer_fold", "target"]
        ].drop_duplicates()
        self.inner_splits = list(oulad._inner_splits(base, self.outer_fold))
        if len(self.inner_splits) != INNER_FOLDS:
            raise RuntimeError("authoritative inner-fold count changed")

    def evaluate(
        self,
        config: dict[str, Any],
        *,
        training_seed: int,
        trial: optuna.Trial | None = None,
        smoke_only: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        frames: list[pd.DataFrame] = []
        probabilities: list[np.ndarray] = []
        inner_results: list[InnerResult] = []
        splits = self.inner_splits[:1] if smoke_only else self.inner_splits
        for inner_fold, (fit_ids, validation_ids) in enumerate(splits):
            checkpoint_path = (
                RUNTIME / "smoke_checkpoint.pt" if smoke_only else None
            )
            result = _train_inner(
                oulad._stage_rows(self.bundle, fit_ids),
                oulad._stage_rows(self.bundle, validation_ids),
                config=config,
                training_seed=training_seed,
                inner_fold=inner_fold,
                trial=trial,
                max_epochs=1 if smoke_only else MAX_EPOCHS,
                checkpoint_path=checkpoint_path,
            )
            frames.append(result.frame)
            probabilities.append(result.probability)
            inner_results.append(result)
        metrics = evaluate_oof(frames, probabilities)
        architecture_hashes = {result.architecture_hash for result in inner_results}
        parameter_counts = {result.parameter_count for result in inner_results}
        if len(architecture_hashes) != 1 or len(parameter_counts) != 1:
            raise RuntimeError("architecture invariant failed within evaluation")
        inner_epochs = [result.selected_epoch for result in inner_results]
        metrics.update(
            {
                "outer_fold": self.outer_fold,
                "training_seed": int(training_seed),
                "architecture_hash": next(iter(architecture_hashes)),
                "parameter_count": next(iter(parameter_counts)),
                "config_hash": stable_hash(config),
                "config": copy.deepcopy(config),
                "inner_selected_epochs": inner_epochs,
                "inner_epochs_trained": [
                    result.epochs_trained for result in inner_results
                ],
                "aggregated_epoch": select_refit_epoch(inner_epochs),
                "inner_pos_weights": [
                    result.pos_weight for result in inner_results
                ],
                "runtime_seconds": time.perf_counter() - started,
                "device": torch.cuda.get_device_name(0),
                "commit_sha": git_commit(),
                "pretraining_executed": False,
                "outer_labels_used": False,
                "shared_estimator_all_stages": True,
                "smoke_only": bool(smoke_only),
            }
        )
        return metrics


def _trial_record(
    trial: optuna.trial.FrozenTrial, outer_fold: int, sampler_seed: int
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "trial_number": trial.number,
        "outer_fold": outer_fold,
        "sampler_seed": sampler_seed,
        "state": trial.state.name,
        "value": trial.value,
        "params": trial.params,
    }
    record.update(trial.user_attrs)
    return record


def objective_for(
    runner: OptunaSearchRunner, outer_fold: int, sampler_seed: int
):
    def objective(trial: optuna.Trial) -> float:
        config = sample_trial_config(trial)
        trial.set_user_attr("sampler_seed", sampler_seed)
        trial.set_user_attr("training_seed", SEARCH_SEED)
        trial.set_user_attr("outer_fold", outer_fold)
        trial.set_user_attr("config_hash", stable_hash(config))
        trial.set_user_attr("config", config)
        trial.set_user_attr("pretraining_executed", False)
        trial.set_user_attr("outer_labels_used", False)
        try:
            result = runner.evaluate(
                config, training_seed=SEARCH_SEED, trial=trial
            )
            for key, value in result.items():
                if key != "config":
                    trial.set_user_attr(key, value)
            append_jsonl(
                LOGS / f"outer{outer_fold}.log",
                {
                    "at": utc_now(),
                    "trial": trial.number,
                    "state": "COMPLETE",
                    "objective": result["mean_stage_macro_f1"],
                    "runtime_seconds": result["runtime_seconds"],
                },
            )
            return float(result["mean_stage_macro_f1"])
        except optuna.TrialPruned:
            trial.set_user_attr("failure_type", "PRUNED")
            append_jsonl(
                LOGS / f"outer{outer_fold}.log",
                {"at": utc_now(), "trial": trial.number, "state": "PRUNED"},
            )
            raise
        except torch.cuda.OutOfMemoryError as error:
            trial.set_user_attr("failure_type", "OOM")
            trial.set_user_attr("failure_reason", str(error))
            torch.cuda.empty_cache()
            raise
        except Exception as error:
            trial.set_user_attr("failure_type", "FAILED")
            trial.set_user_attr("failure_reason", repr(error))
            raise

    return objective


def create_study(outer_fold: int, *, smoke: bool = False) -> optuna.Study:
    suffix = "smoke" if smoke else f"outer{outer_fold}"
    database = OPTUNA_DIR / f"{suffix}.db"
    name = (
        "oulad_hybrid_vnext_smoke"
        if smoke
        else f"oulad_hybrid_vnext_outer{outer_fold}"
    )
    return optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{database.as_posix()}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=SEARCH_SEED + outer_fold,
            n_startup_trials=N_STARTUP_TRIALS,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=N_STARTUP_TRIALS,
            n_warmup_steps=PRUNING_WARMUP_EPOCHS,
            interval_steps=1,
        ),
        load_if_exists=True,
    )


def run_smoke() -> dict[str, Any]:
    validate_preconditions()
    bundle = oulad._build_bundle()
    runner = OptunaSearchRunner(bundle, outer_fold=0)
    study = create_study(0, smoke=True)
    if len(study.trials) < 1:
        study.enqueue_trial(control_config())

        def smoke_objective(trial: optuna.Trial) -> float:
            config = {
                **control_config(),
                **{
                    key: value
                    for key, value in trial.params.items()
                    if key in control_config()
                },
            }
            result = runner.evaluate(
                config, training_seed=SEARCH_SEED, trial=trial, smoke_only=True
            )
            trial.set_user_attr("smoke_only", True)
            trial.set_user_attr("architecture_hash", result["architecture_hash"])
            trial.set_user_attr("parameter_count", result["parameter_count"])
            trial.set_user_attr("outer_labels_used", False)
            return float(result["mean_stage_macro_f1"])

        study.optimize(smoke_objective, n_trials=1)
    before = len(study.trials)
    resumed = create_study(0, smoke=True)
    after = len(resumed.trials)
    checkpoint = RUNTIME / "smoke_checkpoint.pt"
    result = {
        "status": "PASS",
        "smoke_only": True,
        "trial_count": before,
        "resume_trial_count": after,
        "resume_did_not_duplicate": before == after == 1,
        "database": (OPTUNA_DIR / "smoke.db").relative_to(ROOT).as_posix(),
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_exists": checkpoint.is_file(),
        "status_and_sentinel_interface": True,
        "outer_labels_used": False,
    }
    if not all(
        (
            result["resume_did_not_duplicate"],
            result["checkpoint_exists"],
            result["outer_labels_used"] is False,
        )
    ):
        raise RuntimeError("Phase 3 smoke validation failed")
    write_json(OUT / "smoke_validation.json", result)
    status_payload(state="PENDING", current_stage="smoke_complete", exit_code=0)
    (RUNTIME / "PHASE3_SMOKE_COMPLETE").write_text(
        json.dumps({"status": "PASS", "at": utc_now()}, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _study_counts(studies: list[optuna.Study]) -> dict[str, int]:
    trials = [trial for study in studies for trial in study.trials]
    return {
        "completed_trials": sum(
            trial.state == optuna.trial.TrialState.COMPLETE for trial in trials
        ),
        "pruned_trials": sum(
            trial.state == optuna.trial.TrialState.PRUNED for trial in trials
        ),
        "failed_trials": sum(
            trial.state == optuna.trial.TrialState.FAIL for trial in trials
        ),
        "oom_trials": sum(
            trial.user_attrs.get("failure_type") == "OOM" for trial in trials
        ),
    }


def _select_trial(study: optuna.Study) -> optuna.trial.FrozenTrial:
    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
    ]
    if not complete:
        raise RuntimeError(f"study {study.study_name} has no complete trials")
    best_primary = max(float(trial.value) for trial in complete)
    candidates = [
        trial
        for trial in complete
        if float(trial.value) >= best_primary - 1e-4
    ]
    return sorted(
        candidates,
        key=lambda trial: (
            -float(trial.user_attrs["worst_stage_macro_f1"]),
            -float(trial.user_attrs["mean_stage_pr_auc"]),
            float(trial.user_attrs["mean_stage_nll"]),
            float(trial.user_attrs["mean_stage_brier"]),
            int(trial.number),
        ),
    )[0]


def _run_control(
    runner: OptunaSearchRunner, outer_fold: int
) -> dict[str, Any]:
    path = OUT / f"control_outer{outer_fold}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    result = runner.evaluate(control_config(), training_seed=SEARCH_SEED)
    result["record_type"] = "CONTROL_CURRENT"
    write_json(path, result)
    return result


def _run_stability(
    runners: dict[int, OptunaSearchRunner],
    controls: dict[int, dict[str, Any]],
    selected: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    path = OUT / "stability_results.json"
    rows: list[dict[str, Any]] = (
        json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    )
    completed_keys = {
        (int(row["outer_fold"]), row["configuration"], int(row["seed"]))
        for row in rows
    }
    for outer_fold, runner in runners.items():
        for label, config in (
            ("CONTROL_CURRENT", controls[outer_fold]["config"]),
            ("OPTUNA_SELECTED", selected[outer_fold]["config"]),
        ):
            for seed in STABILITY_SEEDS:
                key = (outer_fold, label, seed)
                if key in completed_keys:
                    continue
                result = runner.evaluate(config, training_seed=seed)
                rows.append(
                    {
                        "outer_fold": outer_fold,
                        "configuration": label,
                        "seed": seed,
                        **{
                            key: result[key]
                            for key in (
                                "mean_stage_macro_f1",
                                "worst_stage_macro_f1",
                                "mean_stage_pr_auc",
                                "mean_stage_nll",
                                "mean_stage_brier",
                                "mean_stage_ece",
                                "inner_selected_epochs",
                                "aggregated_epoch",
                                "research_thresholds",
                                "architecture_hash",
                                "parameter_count",
                                "config_hash",
                                "runtime_seconds",
                                "outer_labels_used",
                            )
                        },
                    }
                )
                write_json(path, rows)
                completed_keys.add(key)
                append_jsonl(
                    LOGS / "stability.log",
                    {
                        "at": utc_now(),
                        "outer_fold": outer_fold,
                        "configuration": label,
                        "seed": seed,
                        "state": "COMPLETE",
                    },
                )
    write_json(path, rows)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: canonical_json(value)
                    if isinstance(value, dict)
                    else json.dumps(value)
                    if isinstance(value, list)
                    else value
                    for field, value in row.items()
                }
            )


def _convergence(study: optuna.Study) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for budget in (6, 12, 18, 24):
        values = [
            float(trial.value)
            for trial in study.trials
            if trial.number < budget
            and trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
        ]
        result[f"best_after_{budget}"] = max(values) if values else None
    result["trial24_is_best"] = bool(
        study.trials
        and study.trials[-1].state == optuna.trial.TrialState.COMPLETE
        and study.trials[-1].value == max(
            (
                trial.value
                for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
                and trial.value is not None
            ),
            default=None,
        )
    )
    return result


def generate_machine_outputs(
    studies: dict[int, optuna.Study],
    controls: dict[int, dict[str, Any]],
    selected_trials: dict[int, optuna.trial.FrozenTrial],
    stability: list[dict[str, Any]],
) -> None:
    trial_records = [
        _trial_record(trial, outer_fold, SEARCH_SEED + outer_fold)
        for outer_fold, study in studies.items()
        for trial in study.trials
    ]
    write_json(OUT / "all_trials.json", trial_records)
    _write_csv(OUT / "all_trials.csv", trial_records)
    selected = {
        str(outer_fold): {
            "trial_number": trial.number,
            "objective": trial.value,
            "config": trial.user_attrs["config"],
            "config_hash": trial.user_attrs["config_hash"],
            "inner_selected_epochs": trial.user_attrs["inner_selected_epochs"],
            "aggregated_epoch": trial.user_attrs["aggregated_epoch"],
            "metrics": {
                key: trial.user_attrs[key]
                for key in (
                    "mean_stage_macro_f1",
                    "worst_stage_macro_f1",
                    "mean_stage_pr_auc",
                    "mean_stage_nll",
                    "mean_stage_brier",
                    "mean_stage_ece",
                )
            },
            "stage_metrics": trial.user_attrs["stage_metrics"],
            "research_thresholds": trial.user_attrs["research_thresholds"],
            "architecture_hash": trial.user_attrs["architecture_hash"],
            "parameter_count": trial.user_attrs["parameter_count"],
            "outer_labels_used": trial.user_attrs["outer_labels_used"],
            "pretraining_executed": trial.user_attrs["pretraining_executed"],
        }
        for outer_fold, trial in selected_trials.items()
    }
    write_json(OUT / "selected_configs.json", selected)
    _write_csv(
        OUT / "control_trials.csv",
        [
            {
                "outer_fold": outer_fold,
                **{
                    key: value
                    for key, value in result.items()
                    if key not in {"stage_metrics"}
                },
            }
            for outer_fold, result in controls.items()
        ],
    )
    write_json(OUT / "stability_results.json", stability)
    _write_csv(OUT / "stability_results.csv", stability)
    stage_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    for outer_fold, control in controls.items():
        selected_row = selected[str(outer_fold)]
        for label, metrics, thresholds in (
            (
                "CONTROL_CURRENT",
                control["stage_metrics"],
                control["research_thresholds"],
            ),
            (
                "OPTUNA_SELECTED",
                selected_row["stage_metrics"],
                selected_row["research_thresholds"],
            ),
        ):
            for stage in STAGES:
                stage_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "configuration": label,
                        "prediction_stage": stage,
                        **metrics[stage],
                    }
                )
                threshold_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "configuration": label,
                        "prediction_stage": stage,
                        "research_threshold": thresholds[stage],
                    }
                )
    _write_csv(OUT / "stage_metrics.csv", stage_rows)
    _write_csv(OUT / "threshold_summary.csv", threshold_rows)
    importance: dict[str, Any] = {}
    for outer_fold, study in studies.items():
        try:
            importance[str(outer_fold)] = {
                "status": "SEARCH_ASSOCIATION",
                "values": get_param_importances(study),
            }
        except Exception as error:
            importance[str(outer_fold)] = {
                "status": "UNAVAILABLE",
                "reason": repr(error),
            }
    write_json(OUT / "parameter_importance.json", importance)
    convergence = {
        str(outer_fold): _convergence(study)
        for outer_fold, study in studies.items()
    }
    write_json(OUT / "convergence_summary.json", convergence)
    failures = [
        record
        for record in trial_records
        if record["state"] in {"FAIL", "PRUNED"}
    ]
    write_json(
        OUT / "failure_summary.json",
        {
            "failed_or_pruned": failures,
            "counts": _study_counts(list(studies.values())),
        },
    )
    architecture_hashes = {
        record.get("architecture_hash")
        for record in trial_records
        if record["state"] == "COMPLETE"
    }
    parameter_counts = {
        record.get("parameter_count")
        for record in trial_records
        if record["state"] == "COMPLETE"
    }
    selected_epochs = [
        epoch
        for record in trial_records
        if record["state"] == "COMPLETE"
        for epoch in record.get("inner_selected_epochs", [])
    ]
    cap_fraction = (
        sum(epoch == MAX_EPOCHS for epoch in selected_epochs) / len(selected_epochs)
        if selected_epochs
        else 0.0
    )
    counts = _study_counts(list(studies.values()))
    gate = {
        "status": "PASS"
        if all(len(study.trials) >= TRIALS_PER_OUTER_FOLD for study in studies.values())
        and all(
            any(
                trial.state == optuna.trial.TrialState.COMPLETE
                for trial in study.trials
            )
            for study in studies.values()
        )
        and len(architecture_hashes) == 1
        and len(parameter_counts) == 1
        and not any(record.get("outer_labels_used") for record in trial_records)
        and not any(
            record.get("pretraining_executed") for record in trial_records
        )
        else "FAIL",
        "scheduled_trials": sum(len(study.trials) for study in studies.values()),
        **counts,
        "control_completed": len(controls) == 3,
        "selected_configs_produced": len(selected) == 3,
        "stability_completed": len(stability) == 12,
        "unique_architecture_hash_count": len(architecture_hashes),
        "unique_parameter_count": len(parameter_counts),
        "outer_labels_used": False,
        "pretraining_executed": False,
        "official_final_artifacts_modified": False,
        "epoch_cap_saturation_fraction": cap_fraction,
        "epoch_cap_saturation_warning": cap_fraction >= 0.25,
    }
    write_json(OUT / "phase3_gate.json", gate)
    write_json(
        OUT / "study_manifest.json",
        {
            "status": gate["status"],
            "studies": {
                str(outer_fold): {
                    "name": study.study_name,
                    "database": (
                        OPTUNA_DIR / f"outer{outer_fold}.db"
                    ).relative_to(ROOT).as_posix(),
                    "scheduled_trials": len(study.trials),
                }
                for outer_fold, study in studies.items()
            },
            "sampler": "TPESampler",
            "sampler_seed_by_fold": {
                str(fold): SEARCH_SEED + fold for fold in range(3)
            },
            "n_startup_trials": N_STARTUP_TRIALS,
            "pruner": "MedianPruner",
            "pruning_warmup_epochs": PRUNING_WARMUP_EPOCHS,
            "max_epochs": MAX_EPOCHS,
            "search_seed": SEARCH_SEED,
            "stability_seeds": list(STABILITY_SEEDS),
            "gpu_concurrency": 1,
            "commit_sha": git_commit(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "optuna_version": optuna.__version__,
        },
    )


def run_supervisor() -> int:
    prepare_directories()
    started_at = utc_now()
    set_sentinel("RUNNING", {"pid": os.getpid()})
    status_payload(
        state="RUNNING",
        started_at=started_at,
        finished_at=None,
        current_stage="preconditions",
        exit_code=None,
        pid=os.getpid(),
    )
    try:
        preconditions = validate_preconditions()
        bundle = oulad._build_bundle()
        runners = {
            outer_fold: OptunaSearchRunner(bundle, outer_fold)
            for outer_fold in range(3)
        }
        controls: dict[int, dict[str, Any]] = {}
        studies: dict[int, optuna.Study] = {}
        for outer_fold in range(3):
            status_payload(current_stage=f"control_outer{outer_fold}")
            controls[outer_fold] = _run_control(runners[outer_fold], outer_fold)
            status_payload(last_successful_outer_fold=outer_fold)
        for outer_fold in range(3):
            status_payload(current_stage=f"optuna_outer{outer_fold}")
            study = create_study(outer_fold)
            studies[outer_fold] = study
            remaining = max(0, TRIALS_PER_OUTER_FOLD - len(study.trials))
            if remaining:
                study.optimize(
                    objective_for(
                        runners[outer_fold],
                        outer_fold,
                        SEARCH_SEED + outer_fold,
                    ),
                    n_trials=remaining,
                    catch=(
                        RuntimeError,
                        ValueError,
                        FloatingPointError,
                        torch.cuda.OutOfMemoryError,
                    ),
                    gc_after_trial=True,
                    show_progress_bar=False,
                    callbacks=[
                        lambda _study, _trial: status_payload(
                            **_study_counts(list(studies.values()))
                        )
                    ],
                )
            status_payload(
                last_successful_outer_fold=outer_fold,
                **_study_counts(list(studies.values())),
            )
        selected_trials = {
            outer_fold: _select_trial(study)
            for outer_fold, study in studies.items()
        }
        selected = {
            outer_fold: {
                "config": trial.user_attrs["config"],
                "trial_number": trial.number,
            }
            for outer_fold, trial in selected_trials.items()
        }
        status_payload(current_stage="stability")
        stability = _run_stability(runners, controls, selected)
        status_payload(current_stage="machine_summaries")
        generate_machine_outputs(
            studies, controls, selected_trials, stability
        )
        gate = json.loads((OUT / "phase3_gate.json").read_text(encoding="utf-8"))
        if gate["status"] != "PASS":
            raise RuntimeError("Phase 3 machine gate failed")
        finished_at = utc_now()
        status_payload(
            state="COMPLETE",
            finished_at=finished_at,
            current_stage="complete",
            exit_code=0,
            **_study_counts(list(studies.values())),
        )
        set_sentinel(
            "COMPLETE",
            {
                "started_at": started_at,
                "finished_at": finished_at,
                "gate": gate["status"],
                "preconditions": preconditions["status"],
            },
        )
        return 0
    except Exception as error:
        finished_at = utc_now()
        failure = {
            "state": "FAILED",
            "started_at": started_at,
            "finished_at": finished_at,
            "failure_type": type(error).__name__,
            "failure_reason": repr(error),
            "exit_code": 1,
        }
        status_payload(
            **failure,
            current_stage="failed",
        )
        set_sentinel("FAILED", failure)
        append_jsonl(LOGS / "supervisor.log", failure)
        return 1


def reset_smoke_only() -> None:
    """Remove only disposable smoke evidence; never touch full studies."""
    for path in (
        OPTUNA_DIR / "smoke.db",
        RUNTIME / "smoke_checkpoint.pt",
        RUNTIME / "PHASE3_SMOKE_COMPLETE",
    ):
        if path.is_file():
            path.unlink()


def status() -> dict[str, Any]:
    if not STATUS_PATH.is_file():
        return {"state": "PENDING", "status_file": False}
    return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
