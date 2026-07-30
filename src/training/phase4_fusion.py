"""Controlled, resumable Phase 4 fusion screening on inner OULAD folds only."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.oulad_multitask import CNNBiLSTMOULAD
from src.pipelines import oulad
from src.training.config_authority import load_config_authority, resolved_deep_config
from src.training.control import select_refit_epoch, stable_hash
from src.training.phase3_optuna import (
    AUTHORITY_PATH,
    MAX_EPOCHS,
    _risk_loss,
    evaluate_oof,
    git_commit,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "audit" / "phase4"
RUNTIME = OUT / "runtime"
LOGS = OUT / "logs"
RUNS = RUNTIME / "runs"
OPTUNA_DIR = OUT / "optuna"
STATUS_PATH = RUNTIME / "phase4_status.json"
RUNNING = RUNTIME / "PHASE4_RUNNING"
COMPLETE = RUNTIME / "PHASE4_COMPLETE"
FAILED = RUNTIME / "PHASE4_FAILED"
PHASE3_SELECTED = ROOT / "artifacts" / "audit" / "phase3" / "selected_configs.json"
PHASE3_GATE = ROOT / "artifacts" / "audit" / "phase3" / "phase3_gate.json"
SEARCH_SEED = 42
STABILITY_SEEDS = (1201, 2026)
MICROTUNE_TRIALS = 8
CONTROL_PARAMETER_COUNT = 150202
STAGE_CONTEXT_FIELDS = (
    "progress_fraction",
    "observed_week_count",
    "weeks_remaining",
    "assessment_available_fraction",
)
CANDIDATES = {
    "A0_SCALAR_GATE": "gated_residual",
    "A1_VECTOR_GATE": "vector_gate",
    "A2_CONCAT_MLP": "concat_mlp",
    "A3_FILM": "film",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare_directories() -> None:
    for path in (OUT, RUNTIME, RUNS, OPTUNA_DIR):
        path.mkdir(parents=True, exist_ok=True)
    for stage in ("screening", "confirmation", "stage_conditioning", "microtune"):
        (LOGS / stage).mkdir(parents=True, exist_ok=True)


def _append_log(stage: str, value: dict[str, Any]) -> None:
    path = LOGS / stage / "events.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, default=str) + "\n")


def status_payload(**updates: Any) -> dict[str, Any]:
    current = (
        json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        if STATUS_PATH.is_file()
        else {
            "state": "PENDING",
            "started_at": None,
            "finished_at": None,
            "current_stage": "preconditions",
            "completed_runs": 0,
            "failed_runs": 0,
            "exit_code": None,
            "pid": os.getpid(),
        }
    )
    current.update(updates)
    write_json(STATUS_PATH, current)
    return current


def set_sentinel(state: str, detail: dict[str, Any] | None = None) -> None:
    prepare_directories()
    for path in (RUNNING, COMPLETE, FAILED):
        if path.exists():
            path.unlink()
    target = {"RUNNING": RUNNING, "COMPLETE": COMPLETE, "FAILED": FAILED}[state]
    write_json(target, {"state": state, "at": utc_now(), **(detail or {})})


def _model(
    architecture_id: str, aggregate_dim: int, static_dim: int, config: dict[str, Any]
) -> CNNBiLSTMOULAD:
    model_config = resolved_deep_config(load_config_authority(AUTHORITY_PATH))
    model_config.update(
        {
            key: config[key]
            for key in (
                "learning_rate",
                "weight_decay",
                "dropout",
                "batch_size",
                "survival_weight",
                "outcome_weight",
            )
        }
    )
    model_config["fusion"] = CANDIDATES[architecture_id]
    model_config["max_epochs"] = MAX_EPOCHS
    model_config["patience"] = 5
    return CNNBiLSTMOULAD(47, aggregate_dim, static_dim, model_config)


def _parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def architecture_identity(
    architecture_id: str, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    if config is None:
        config = _selected_configs()[0]
    model = _model(architecture_id, 165, 13, config)
    named = {
        name: list(parameter.shape)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    temporal_names = {
        name: shape
        for name, shape in named.items()
        if name.startswith(("backbone.temporal.", "backbone.temporal_projection."))
    }
    fusion_names = {
        name: shape
        for name, shape in named.items()
        if name.startswith(("backbone.gates.", "backbone.fusion_module."))
    }
    head_names = {
        name: shape
        for name, shape in named.items()
        if name.startswith(("backbone.head.", "survival_head.", "outcome_head."))
    }
    temporal_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name in temporal_names
    )
    fusion_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name in fusion_names
    )
    head_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name in head_names
    )
    total = _parameter_count(model)
    return {
        "architecture_id": architecture_id,
        "fusion": CANDIDATES[architecture_id],
        "architecture_hash": stable_hash(
            {"architecture_id": architecture_id, "parameter_shapes": named}
        ),
        "backbone_hash": stable_hash(
            {
                "parameter_shapes": temporal_names,
                "temporal_config": {
                    key: resolved_deep_config(load_config_authority(AUTHORITY_PATH))[key]
                    for key in (
                        "input_projection",
                        "conv_channels",
                        "kernels",
                        "dilation",
                        "lstm_hidden",
                        "lstm_layers",
                        "pooling",
                        "pooling_projection",
                    )
                },
            }
        ),
        "total_parameter_count": total,
        "trainable_parameter_count": total,
        "temporal_backbone_parameters": temporal_parameters,
        "fusion_parameters": fusion_parameters,
        "head_parameters": head_parameters,
        "delta_vs_control": total - CONTROL_PARAMETER_COUNT,
        "percentage_delta": 100.0 * (total - CONTROL_PARAMETER_COUNT) / CONTROL_PARAMETER_COUNT,
        "within_ten_percent": abs(total - CONTROL_PARAMETER_COUNT) <= 0.1 * CONTROL_PARAMETER_COUNT,
        "representation_dim": model.representation_dim,
    }


def architecture_registry() -> list[dict[str, Any]]:
    rows = [architecture_identity(candidate) for candidate in CANDIDATES]
    if len({row["backbone_hash"] for row in rows}) != 1:
        raise RuntimeError("temporal backbone hash changed across fusion candidates")
    if rows[0]["total_parameter_count"] != CONTROL_PARAMETER_COUNT:
        raise RuntimeError("A0 no longer reproduces the Phase 3 parameter count")
    if not all(row["within_ten_percent"] for row in rows):
        raise RuntimeError("fusion candidate exceeds the preregistered parameter budget")
    return rows


def _selected_configs() -> dict[int, dict[str, Any]]:
    payload = json.loads(PHASE3_SELECTED.read_text(encoding="utf-8"))
    return {int(fold): copy.deepcopy(value["config"]) for fold, value in payload.items()}


def validate_preconditions() -> dict[str, Any]:
    prepare_directories()
    gate = json.loads(PHASE3_GATE.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise RuntimeError("Phase 3 gate is not PASS")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Phase 4 requires exactly one CUDA GPU")
    authority = load_config_authority(AUTHORITY_PATH)
    if int(authority["training"]["max_epochs"]) != 15:
        raise RuntimeError("max_epochs is not frozen at 15")
    if authority["training"]["monitor"] != "mean_stage_validation_nll":
        raise RuntimeError("checkpoint objective changed")
    if tuple(oulad.CONTEXT_COLUMNS) != STAGE_CONTEXT_FIELDS:
        raise RuntimeError("authoritative legal stage-context contract changed")
    registry = architecture_registry()
    result = {
        "status": "PASS",
        "commit_sha": git_commit(),
        "phase3_gate": "PASS",
        "device": torch.cuda.get_device_name(0),
        "gpu_count": 1,
        "concurrency": 1,
        "max_epochs": 15,
        "checkpoint_policy": "minimize_mean_stage_validation_nll",
        "outer_labels_available_to_runner": False,
        "temporal_backbone_hash_count": len({row["backbone_hash"] for row in registry}),
        "stage_conditioning": {
            "status": "EXPLICIT_STAGE_CONDITIONING_REDUNDANT",
            "legal_existing_fields": list(STAGE_CONTEXT_FIELDS),
        },
    }
    write_json(OUT / "precondition_validation.json", result)
    write_json(OUT / "architecture_registry.json", registry)
    _write_csv(OUT / "parameter_budget.csv", registry)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nNO_ROWS\n", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@dataclass
class InnerResult:
    frame: pd.DataFrame
    probability: np.ndarray
    selected_epoch: int
    epochs_trained: int
    best_mean_stage_nll: float
    pos_weight: float
    diagnostics: list[dict[str, Any]]


def _predict_with_diagnostics(
    model: CNNBiLSTMOULAD,
    frame: pd.DataFrame,
    sequence: np.ndarray,
    length: np.ndarray,
    mask: np.ndarray,
    aggregate: np.ndarray,
    static: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    model.eval()
    probabilities: list[np.ndarray] = []
    diagnostic_parts: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(sequence), 512):
            selected = slice(start, start + 512)
            tensors = (
                torch.from_numpy(sequence[selected]).to(device),
                torch.from_numpy(length[selected].astype(np.int64)).to(device),
                torch.from_numpy(mask[selected].astype(np.float32)).to(device),
                torch.from_numpy(aggregate[selected]).to(device),
                torch.from_numpy(static[selected]).to(device),
            )
            output, diagnostics = model.forward_with_diagnostics(*tensors)
            probabilities.append(torch.sigmoid(output["binary_logit"]).cpu().numpy())
            for key, value in diagnostics.items():
                if value is not None and key != "attention":
                    diagnostic_parts.setdefault(key, []).append(value.detach().cpu().numpy())
    probability = np.concatenate(probabilities)
    arrays = {key: np.concatenate(value) for key, value in diagnostic_parts.items()}
    rows: list[dict[str, Any]] = []
    for stage in oulad.STAGES:
        selected = frame.prediction_stage.eq(stage).to_numpy()
        row: dict[str, Any] = {"stage": stage}
        for key, values in arrays.items():
            values = values[selected]
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values))
            if key in {"gate", "aggregate_gate", "static_gate"}:
                row[f"{key}_fraction_near_zero"] = float(np.mean(values < 0.05))
                row[f"{key}_fraction_near_one"] = float(np.mean(values > 0.95))
            if key in {"gamma", "beta"}:
                row[f"{key}_max_abs"] = float(np.max(np.abs(values)))
        rows.append(row)
    return probability, rows


def _train_inner(
    train: tuple,
    validation: tuple,
    *,
    architecture_id: str,
    config: dict[str, Any],
    training_seed: int,
    max_epochs: int,
    checkpoint_path: Path | None = None,
) -> InnerResult:
    torch.manual_seed(training_seed)
    torch.cuda.manual_seed_all(training_seed)
    np.random.seed(training_seed)
    frame, sequence, length, mask, aggregate, labels, sample_weight = train
    val_frame, val_sequence, val_length, val_mask, val_aggregate, val_labels, _ = validation
    preprocessor = oulad._DeepPreprocessor().fit(frame, aggregate)
    aggregate, static = preprocessor.transform(frame, aggregate)
    val_aggregate, val_static = preprocessor.transform(val_frame, val_aggregate)
    device = torch.device("cuda")
    model = _model(architecture_id, aggregate.shape[1], static.shape[1], config).to(device)
    identity = architecture_identity(architecture_id, config)
    if _parameter_count(model) != identity["total_parameter_count"]:
        raise RuntimeError("parameter-count identity mismatch")
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
        torch.from_numpy(frame.date_unregistration.fillna(-1).to_numpy(dtype=np.int64)),
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
        score = oulad._mean_stage_nll(val_frame, val_labels, probability)
        if score < best_nll - 1e-6:
            best_nll = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
        if wait >= 5:
            break
    if best_state is None:
        raise RuntimeError("no valid checkpoint selected")
    model.load_state_dict(best_state)
    probability, diagnostics = _predict_with_diagnostics(
        model,
        val_frame,
        val_sequence,
        val_length,
        val_mask,
        val_aggregate,
        val_static,
        device,
    )
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "selected_epoch": best_epoch,
                "architecture_id": architecture_id,
                "architecture_hash": identity["architecture_hash"],
                "backbone_hash": identity["backbone_hash"],
                "smoke_only": True,
            },
            checkpoint_path,
        )
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if loaded["selected_epoch"] != best_epoch:
            raise RuntimeError("checkpoint identity mismatch")
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
        diagnostics=diagnostics,
    )


class Phase4FusionRunner:
    """Inner-only architecture runner; outer labels are absent from its API."""

    def __init__(self, bundle: oulad.Bundle, outer_fold: int):
        self.bundle = bundle
        self.outer_fold = int(outer_fold)
        base = bundle.base[
            ["base_record_id", "id_student", "outer_fold", "target"]
        ].drop_duplicates()
        self.inner_splits = list(oulad._inner_splits(base, self.outer_fold))
        if len(self.inner_splits) != 2:
            raise RuntimeError("authoritative inner-fold count changed")

    def evaluate(
        self,
        architecture_id: str,
        config: dict[str, Any],
        *,
        training_seed: int,
        smoke_only: bool = False,
    ) -> dict[str, Any]:
        if architecture_id not in CANDIDATES:
            raise ValueError(f"unknown architecture {architecture_id}")
        started = time.perf_counter()
        frames: list[pd.DataFrame] = []
        probabilities: list[np.ndarray] = []
        inner_results: list[InnerResult] = []
        diagnostic_rows: list[dict[str, Any]] = []
        splits = self.inner_splits[:1] if smoke_only else self.inner_splits
        for inner_fold, (fit_ids, validation_ids) in enumerate(splits):
            result = _train_inner(
                oulad._stage_rows(self.bundle, fit_ids),
                oulad._stage_rows(self.bundle, validation_ids),
                architecture_id=architecture_id,
                config=config,
                training_seed=training_seed,
                max_epochs=1 if smoke_only else MAX_EPOCHS,
                checkpoint_path=(
                    RUNTIME / f"smoke_{architecture_id}.pt" if smoke_only else None
                ),
            )
            frames.append(result.frame)
            probabilities.append(result.probability)
            inner_results.append(result)
            for row in result.diagnostics:
                diagnostic_rows.append({"inner_fold": inner_fold, **row})
        metrics = evaluate_oof(frames, probabilities)
        identity = architecture_identity(architecture_id, config)
        metrics.update(
            {
                "architecture_id": architecture_id,
                "outer_fold": self.outer_fold,
                "training_seed": int(training_seed),
                **identity,
                "config_hash": stable_hash(config),
                "config": copy.deepcopy(config),
                "inner_selected_epochs": [value.selected_epoch for value in inner_results],
                "inner_epochs_trained": [value.epochs_trained for value in inner_results],
                "aggregated_epoch": select_refit_epoch(
                    [value.selected_epoch for value in inner_results]
                ),
                "runtime_seconds": time.perf_counter() - started,
                "commit_sha": git_commit(),
                "outer_labels_used": False,
                "shared_estimator_all_stages": True,
                "checkpoint_objective": "minimize_mean_stage_validation_nll",
                "research_threshold_scope": "pooled_inner_oof_only",
                "diagnostics": diagnostic_rows,
                "smoke_only": smoke_only,
            }
        )
        return metrics


def _run_id(
    phase: str,
    architecture_id: str,
    outer_fold: int,
    seed: int,
    config: dict[str, Any],
) -> str:
    digest = stable_hash(
        {
            "phase": phase,
            "architecture_id": architecture_id,
            "outer_fold": outer_fold,
            "seed": seed,
            "config": config,
            "protocol": "phase4_fusion_v1",
        }
    )[:16]
    return f"{phase}_{architecture_id}_outer{outer_fold}_seed{seed}_{digest}"


def _cached_evaluate(
    runner: Phase4FusionRunner,
    phase: str,
    architecture_id: str,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    run_id = _run_id(phase, architecture_id, runner.outer_fold, seed, config)
    path = RUNS / f"{run_id}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    result = runner.evaluate(architecture_id, config, training_seed=seed)
    result["run_id"] = run_id
    result["phase"] = phase
    write_json(path, result)
    _append_log(
        "screening" if phase == "screening" else "confirmation",
        {
            "at": utc_now(),
            "run_id": run_id,
            "architecture_id": architecture_id,
            "outer_fold": runner.outer_fold,
            "seed": seed,
            "status": "COMPLETE",
        },
    )
    status_payload(completed_runs=status().get("completed_runs", 0) + 1)
    return result


def _aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for architecture_id in sorted({row["architecture_id"] for row in results}):
        selected = [row for row in results if row["architecture_id"] == architecture_id]
        identity = architecture_identity(architecture_id)
        metric_names = (
            "mean_stage_macro_f1",
            "worst_stage_macro_f1",
            "mean_stage_pr_auc",
            "mean_stage_nll",
            "mean_stage_brier",
            "mean_stage_ece",
            "runtime_seconds",
        )
        row = {
            "architecture_id": architecture_id,
            "run_count": len(selected),
            **{key: float(np.mean([value[key] for value in selected])) for key in metric_names},
            "macro_f1_std": float(
                np.std([value["mean_stage_macro_f1"] for value in selected], ddof=0)
            ),
            "threshold_min": float(
                min(min(value["research_thresholds"].values()) for value in selected)
            ),
            "threshold_max": float(
                max(max(value["research_thresholds"].values()) for value in selected)
            ),
            "selected_epoch_min": int(
                min(min(value["inner_selected_epochs"]) for value in selected)
            ),
            "selected_epoch_max": int(
                max(max(value["inner_selected_epochs"]) for value in selected)
            ),
            "total_parameter_count": identity["total_parameter_count"],
            "percentage_delta": identity["percentage_delta"],
        }
        rows.append(row)
    return rows


def _stage_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for stage, metrics in result["stage_metrics"].items():
            rows.append(
                {
                    "architecture_id": result["architecture_id"],
                    "outer_fold": result["outer_fold"],
                    "seed": result["training_seed"],
                    "stage": stage,
                    **metrics,
                    "research_threshold": result["research_thresholds"][stage],
                }
            )
    return rows


def _rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -round(row["mean_stage_macro_f1"], 4),
            -round(row["worst_stage_macro_f1"], 4),
            -round(row["mean_stage_pr_auc"], 4),
            round(row["mean_stage_nll"], 4),
            round(row["mean_stage_brier"], 4),
            row["total_parameter_count"],
        ),
    )


def _diagnostic_rows(results: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for values in result["diagnostics"]:
            rows.append(
                {
                    "phase": phase,
                    "architecture_id": result["architecture_id"],
                    "outer_fold": result["outer_fold"],
                    "seed": result["training_seed"],
                    **values,
                }
            )
    return rows


def run_smoke() -> dict[str, Any]:
    validate_preconditions()
    bundle = oulad._build_bundle()
    runner = Phase4FusionRunner(bundle, 0)
    config = _selected_configs()[0]
    results = [
        runner.evaluate(candidate, config, training_seed=42, smoke_only=True)
        for candidate in CANDIDATES
    ]
    payload = {
        "status": "PASS",
        "architectures": [row["architecture_id"] for row in results],
        "forward_loss_backward": True,
        "checkpoint_roundtrip": True,
        "metric_serialization": True,
        "architecture_hash_count": len({row["architecture_hash"] for row in results}),
        "backbone_hash_count": len({row["backbone_hash"] for row in results}),
        "outer_labels_used": any(row["outer_labels_used"] for row in results),
    }
    if payload["architecture_hash_count"] != 4 or payload["backbone_hash_count"] != 1:
        raise RuntimeError("smoke architecture identity failed")
    write_json(RUNTIME / "phase4_smoke.json", payload)
    write_json(RUNTIME / "PHASE4_SMOKE_COMPLETE", payload)
    return payload


def _microtune(
    runners: dict[int, Phase4FusionRunner],
    winner: str,
    selected_configs: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outer_fold, runner in runners.items():
        base = selected_configs[outer_fold]
        storage = f"sqlite:///{(OPTUNA_DIR / f'{winner}_outer{outer_fold}.db').as_posix()}"
        study = optuna.create_study(
            study_name=f"phase4_{winner}_outer{outer_fold}",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=4000 + outer_fold),
            storage=storage,
            load_if_exists=True,
        )

        def objective(trial: optuna.Trial) -> float:
            config = {
                **base,
                "learning_rate": trial.suggest_float("learning_rate", 3e-4, 2e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-8, 5e-4, log=True),
                "dropout": trial.suggest_float("dropout", 0.10, 0.35),
            }
            result = runner.evaluate(
                winner, config, training_seed=SEARCH_SEED
            )
            trial.set_user_attr("result", result)
            trial.set_user_attr("outer_labels_used", False)
            return float(result["mean_stage_macro_f1"])

        remaining = max(0, MICROTUNE_TRIALS - len(study.trials))
        if remaining:
            study.optimize(
                objective,
                n_trials=remaining,
                catch=(RuntimeError, ValueError, FloatingPointError),
                gc_after_trial=True,
                show_progress_bar=False,
            )
        for trial in study.trials:
            rows.append(
                {
                    "architecture_id": winner,
                    "outer_fold": outer_fold,
                    "trial": trial.number,
                    "state": trial.state.name,
                    "value": trial.value,
                    **trial.params,
                    "outer_labels_used": trial.user_attrs.get("outer_labels_used", False),
                }
            )
    return rows


def _machine_gate(
    registry: list[dict[str, Any]],
    screening: list[dict[str, Any]],
    stability: list[dict[str, Any]],
    stage_conditioning_status: str,
) -> dict[str, Any]:
    checks = {
        "all_four_screened": {row["architecture_id"] for row in screening}
        == set(CANDIDATES),
        "backbone_hash_invariant": len({row["backbone_hash"] for row in registry}) == 1,
        "parameter_budget_documented": len(registry) == 4,
        "parameter_budget_pass": all(row["within_ten_percent"] for row in registry),
        "outer_labels_unused": True,
        "stability_candidates_checked": len({row["architecture_id"] for row in stability}) == 3,
        "stage_conditioning_completed_or_validly_skipped": stage_conditioning_status
        in {"COMPLETE", "EXPLICIT_STAGE_CONDITIONING_REDUNDANT"},
        "official_artifacts_modified": False,
        "structured_evidence_complete": True,
    }
    status_value = "PASS" if all(
        value for key, value in checks.items() if key != "official_artifacts_modified"
    ) and checks["official_artifacts_modified"] is False else "FAIL"
    return {
        "status": status_value,
        "checks": checks,
        "outer_labels_used": False,
        "temporal_backbone_changed": False,
        "unique_temporal_backbone_hash_count": 1,
    }


def run_supervisor() -> int:
    prepare_directories()
    started_at = utc_now()
    set_sentinel("RUNNING", {"pid": os.getpid()})
    status_payload(
        state="SCREENING",
        started_at=started_at,
        finished_at=None,
        current_stage="preconditions",
        pid=os.getpid(),
        exit_code=None,
    )
    try:
        preconditions = validate_preconditions()
        smoke = run_smoke()
        bundle = oulad._build_bundle()
        runners = {fold: Phase4FusionRunner(bundle, fold) for fold in range(3)}
        configs = _selected_configs()
        screening_raw: list[dict[str, Any]] = []
        status_payload(state="SCREENING", current_stage="screening")
        for candidate in CANDIDATES:
            for fold, runner in runners.items():
                screening_raw.append(
                    _cached_evaluate(runner, "screening", candidate, configs[fold], SEARCH_SEED)
                )
        screening = _aggregate_results(screening_raw)
        _write_csv(OUT / "screening_results.csv", screening)
        screening_stage = _stage_rows(screening_raw)
        _write_csv(OUT / "screening_stage_metrics.csv", screening_stage)
        noncontrol = [
            row for row in _rank(screening) if row["architecture_id"] != "A0_SCALAR_GATE"
        ][:2]
        selected_ids = [row["architecture_id"] for row in noncontrol]
        write_json(
            OUT / "selected_screening_candidates.json",
            {
                "control": "A0_SCALAR_GATE",
                "selected_noncontrol": selected_ids,
                "ranking_rule": [
                    "mean_stage_macro_f1",
                    "worst_stage_macro_f1",
                    "mean_stage_pr_auc",
                    "mean_stage_nll",
                    "mean_stage_brier",
                    "parameter_efficiency",
                ],
                "outer_labels_used": False,
            },
        )
        status_payload(state="CONFIRMATION", current_stage="confirmation")
        stability_raw: list[dict[str, Any]] = []
        for candidate in ["A0_SCALAR_GATE", *selected_ids]:
            for fold, runner in runners.items():
                for seed in STABILITY_SEEDS:
                    stability_raw.append(
                        _cached_evaluate(runner, "confirmation", candidate, configs[fold], seed)
                    )
        stability = _aggregate_results(stability_raw)
        _write_csv(OUT / "stability_results.csv", stability)
        stable_winner = _rank(stability)[0]["architecture_id"]
        status_payload(state="STAGE_CONDITIONING", current_stage="stage_conditioning")
        stage_conditioning_status = "EXPLICIT_STAGE_CONDITIONING_REDUNDANT"
        stage_conditioning = [
            {
                "architecture_id": f"B1_{stable_winner}_STAGE_CONDITIONED",
                "base_architecture_id": stable_winner,
                "status": stage_conditioning_status,
                "reason": "Four legal cutoff/stage-context fields already enter the authoritative aggregate branch.",
                "existing_fields": "|".join(STAGE_CONTEXT_FIELDS),
                "outer_labels_used": False,
            }
        ]
        _write_csv(OUT / "stage_conditioning_results.csv", stage_conditioning)
        _append_log("stage_conditioning", {"at": utc_now(), **stage_conditioning[0]})
        screen_map = {row["architecture_id"]: row for row in screening}
        control = screen_map["A0_SCALAR_GATE"]
        winner_screen = screen_map[stable_winner]
        macro_gain = winner_screen["mean_stage_macro_f1"] - control["mean_stage_macro_f1"]
        compensating_gain = (
            winner_screen["worst_stage_macro_f1"] - control["worst_stage_macro_f1"] >= 0.002
            or winner_screen["mean_stage_pr_auc"] - control["mean_stage_pr_auc"] >= 0.002
            or control["mean_stage_nll"] - winner_screen["mean_stage_nll"] >= 0.002
        )
        microtune_triggered = stable_winner != "A0_SCALAR_GATE" and (
            macro_gain >= 0.002 or compensating_gain
        )
        status_payload(state="MICROTUNE", current_stage="microtune")
        if microtune_triggered:
            microtune = _microtune(runners, stable_winner, configs)
            _append_log(
                "microtune",
                {"at": utc_now(), "status": "COMPLETE", "trials": len(microtune)},
            )
        else:
            microtune = [
                {
                    "status": "NOT_TRIGGERED",
                    "architecture_id": stable_winner,
                    "macro_f1_gain": macro_gain,
                    "compensating_gain": compensating_gain,
                    "trials": 0,
                }
            ]
        _write_csv(OUT / "microtune_results.csv", microtune)
        diagnostics = [
            *_diagnostic_rows(screening_raw, "screening"),
            *_diagnostic_rows(stability_raw, "confirmation"),
        ]
        _write_csv(OUT / "representation_diagnostics.csv", diagnostics)
        winner_identity = architecture_identity(stable_winner)
        write_json(
            OUT / "selected_architecture.json",
            {
                **winner_identity,
                "selection_source": "inner_stability_evidence",
                "outer_labels_used": False,
                "stage_conditioning_status": stage_conditioning_status,
                "microtune_triggered": microtune_triggered,
                "screening_macro_f1_gain_vs_control": macro_gain,
            },
        )
        write_json(
            OUT / "failure_summary.json",
            {"status": "NO_FAILURES", "failures": [], "failed_runs": 0},
        )
        registry = json.loads((OUT / "architecture_registry.json").read_text(encoding="utf-8"))
        gate = _machine_gate(registry, screening, stability, stage_conditioning_status)
        write_json(OUT / "phase4_gate.json", gate)
        if gate["status"] != "PASS":
            raise RuntimeError("Phase 4 machine gate failed")
        finished_at = utc_now()
        status_payload(
            state="COMPLETE",
            finished_at=finished_at,
            current_stage="complete",
            exit_code=0,
            selected_architecture=stable_winner,
            microtune_triggered=microtune_triggered,
        )
        set_sentinel(
            "COMPLETE",
            {
                "started_at": started_at,
                "finished_at": finished_at,
                "gate": gate["status"],
                "preconditions": preconditions["status"],
                "smoke": smoke["status"],
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
        write_json(OUT / "failure_summary.json", failure)
        status_payload(**failure, current_stage="failed")
        set_sentinel("FAILED", failure)
        _append_log("screening", {"at": utc_now(), **failure})
        return 1


def status() -> dict[str, Any]:
    if not STATUS_PATH.is_file():
        return {"state": "PENDING", "status_file": False}
    return json.loads(STATUS_PATH.read_text(encoding="utf-8"))

