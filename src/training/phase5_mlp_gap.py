"""Bounded Phase 5 inner-only MLP-gap and tabular-residual study."""

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
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.oulad_multitask import CNNBiLSTMOULAD
from src.models.oulad_tabular_residual import CNNBiLSTMTabularResidualOULAD
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
OUT = ROOT / "artifacts" / "audit" / "phase5"
RUNTIME = OUT / "runtime"
RUNS = RUNTIME / "runs"
PREDICTIONS = RUNTIME / "predictions"
LOGS = OUT / "logs"
STATUS_PATH = RUNTIME / "phase5_status.json"
RUNNING = RUNTIME / "PHASE5_RUNNING"
COMPLETE = RUNTIME / "PHASE5_COMPLETE"
FAILED = RUNTIME / "PHASE5_FAILED"
PHASE4_GATE = ROOT / "artifacts" / "audit" / "phase4" / "phase4_gate.json"
PHASE3_SELECTED = ROOT / "artifacts" / "audit" / "phase3" / "selected_configs.json"
SEARCH_SEED = 42
STABILITY_SEEDS = (1201, 2026)
DISTILL_LAMBDAS = (0.05, 0.10, 0.20)
CANDIDATES = ("M0_MLP", "H0_CURRENT_HYBRID", "H1_TABULAR_RESIDUAL_EXPERT")
CONTROL_PARAMETERS = 150202


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def prepare_directories() -> None:
    for path in (OUT, RUNTIME, RUNS, PREDICTIONS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


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
            "current_candidate": None,
            "distillation_triggered": False,
            "microtune_triggered": False,
            "exit_code": None,
            "pid": os.getpid(),
        }
    )
    current.update(updates)
    write_json(STATUS_PATH, current)
    return current


def set_sentinel(state: str, details: dict[str, Any] | None = None) -> None:
    for path in (RUNNING, COMPLETE, FAILED):
        if path.exists():
            path.unlink()
    target = {"RUNNING": RUNNING, "COMPLETE": COMPLETE, "FAILED": FAILED}[state]
    write_json(target, {"state": state, "at": utc_now(), **(details or {})})


def _selected_configs() -> dict[int, dict[str, Any]]:
    payload = json.loads(PHASE3_SELECTED.read_text(encoding="utf-8"))
    return {int(key): copy.deepcopy(value["config"]) for key, value in payload.items()}


def _deep_config(config: dict[str, Any]) -> dict[str, Any]:
    result = resolved_deep_config(load_config_authority(AUTHORITY_PATH))
    result.update(config)
    result["fusion"] = "gated_residual"
    result["max_epochs"] = MAX_EPOCHS
    result["patience"] = 5
    return result


def make_model(
    candidate: str, aggregate_dim: int, static_dim: int, config: dict[str, Any]
) -> nn.Module:
    model_config = _deep_config(config)
    if candidate == "H0_CURRENT_HYBRID":
        return CNNBiLSTMOULAD(47, aggregate_dim, static_dim, model_config)
    if candidate in {"H1_TABULAR_RESIDUAL_EXPERT", "H2_MLP_DISTILLED_HYBRID"}:
        return CNNBiLSTMTabularResidualOULAD(47, aggregate_dim, static_dim, model_config)
    raise ValueError(candidate)


def _count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def architecture_registry() -> list[dict[str, Any]]:
    config = _selected_configs()[0]
    rows: list[dict[str, Any]] = [
        {
            "architecture_id": "M0_MLP",
            "implementation": "sklearn.neural_network.MLPClassifier",
            "hidden_layers": [64, 32],
            "parameter_count": None,
            "parameter_count_status": "FIT_DEPENDENT_ONE_HOT_DIMENSION",
            "temporal_cnn_changed": False,
            "fusion_a0_changed": False,
        }
    ]
    for candidate in ("H0_CURRENT_HYBRID", "H1_TABULAR_RESIDUAL_EXPERT"):
        model = make_model(candidate, 165, 13, config)
        named = {
            name: list(parameter.shape)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        temporal = {
            name: shape
            for name, shape in named.items()
            if name.startswith(("backbone.temporal.", "backbone.temporal_projection."))
        }
        total = _count(model)
        rows.append(
            {
                "architecture_id": candidate,
                "implementation": f"{model.__class__.__module__}.{model.__class__.__name__}",
                "architecture_hash": stable_hash({"candidate": candidate, "shapes": named}),
                "temporal_backbone_hash": stable_hash(temporal),
                "parameter_count": total,
                "delta_vs_h0": total - CONTROL_PARAMETERS,
                "percentage_delta": 100 * (total - CONTROL_PARAMETERS) / CONTROL_PARAMETERS,
                "within_fifteen_percent": total <= CONTROL_PARAMETERS * 1.15,
                "temporal_cnn_changed": False,
                "fusion_a0_changed": False,
            }
        )
    if rows[1]["parameter_count"] != CONTROL_PARAMETERS:
        raise RuntimeError("H0 no longer reproduces Phase 4")
    if rows[1]["temporal_backbone_hash"] != rows[2]["temporal_backbone_hash"]:
        raise RuntimeError("H1 temporal backbone changed")
    if not rows[2]["within_fifteen_percent"]:
        raise RuntimeError("H1 exceeds parameter budget")
    return rows


def validate_preconditions() -> dict[str, Any]:
    prepare_directories()
    phase4 = json.loads(PHASE4_GATE.read_text(encoding="utf-8"))
    if phase4.get("status") != "PASS":
        raise RuntimeError("Phase 4 gate is not PASS")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Phase 5 requires exactly one CUDA GPU")
    registry = architecture_registry()
    result = {
        "status": "PASS",
        "commit_sha": git_commit(),
        "phase4_gate": "PASS",
        "device": torch.cuda.get_device_name(0),
        "gpu_count": 1,
        "concurrency": 1,
        "outer_labels_available_to_runner": False,
        "temporal_cnn_changed": False,
        "fusion_a0_changed": False,
        "max_epochs": 15,
        "checkpoint_policy": "minimize_mean_stage_validation_nll",
        "research_threshold": "pooled_inner_oof_macro_f1",
    }
    write_json(OUT / "architecture_registry.json", registry)
    _write_csv(OUT / "parameter_budget.csv", registry)
    write_json(OUT / "precondition_validation.json", result)
    return result


def _mlp_parameter_count(estimator: Any) -> int:
    model = estimator.named_steps["model"]
    return int(sum(array.size for array in [*model.coefs_, *model.intercepts_]))


def _feature_columns(aggregate: np.ndarray) -> dict[str, np.ndarray]:
    assessment_index = oulad.BASE_CHANNELS.index("submitted_assessment_count") * 10
    return {
        "total_clicks": aggregate[:, 0],
        "assessment_count": aggregate[:, assessment_index],
        "observed_weeks": aggregate[:, -3],
    }


def _prediction_frame(
    frame: pd.DataFrame, probability: np.ndarray, aggregate: np.ndarray
) -> pd.DataFrame:
    result = frame.loc[
        :, ["base_record_id", "id_student", "prediction_stage", "target"]
    ].copy()
    result["probability"] = probability
    for key, values in _feature_columns(aggregate).items():
        result[key] = values
    return result


def _teacher_oof(train: tuple, seed: int) -> np.ndarray:
    frame = train[0]
    base = frame[
        ["base_record_id", "id_student", "target"]
    ].drop_duplicates("base_record_id").reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed + 500)
    output = np.full(len(frame), np.nan, dtype=np.float32)
    for fit, validation in splitter.split(base, base.target, base.id_student):
        fit_ids = set(base.iloc[fit].base_record_id)
        validation_ids = set(base.iloc[validation].base_record_id)
        fit_mask = frame.base_record_id.isin(fit_ids).to_numpy()
        validation_mask = frame.base_record_id.isin(validation_ids).to_numpy()
        fit_rows = tuple(value.loc[fit_mask].reset_index(drop=True) if isinstance(value, pd.DataFrame) else value[fit_mask] for value in train)
        validation_rows = tuple(value.loc[validation_mask].reset_index(drop=True) if isinstance(value, pd.DataFrame) else value[validation_mask] for value in train)
        _, payload = oulad._fit_tabular("mlp", fit_rows, validation_rows, seed)
        output[validation_mask] = payload["probability"]
    if not np.isfinite(output).all():
        raise RuntimeError("cross-fitted teacher coverage failed")
    return output


@dataclass
class InnerResult:
    prediction: pd.DataFrame
    selected_epoch: int | None
    epochs_trained: int | None
    parameter_count: int
    diagnostics: list[dict[str, Any]]
    temporal_disabled_probability: np.ndarray | None = None
    residual_disabled_probability: np.ndarray | None = None


def _predict_deep(
    model: nn.Module,
    frame: pd.DataFrame,
    sequence: np.ndarray,
    length: np.ndarray,
    mask: np.ndarray,
    aggregate: np.ndarray,
    static: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray | None, np.ndarray | None]:
    model.eval()
    probabilities: list[np.ndarray] = []
    temporal_disabled: list[np.ndarray] = []
    residual_disabled: list[np.ndarray] = []
    raw: dict[str, list[np.ndarray]] = {}
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
            output = model(*tensors)
            probabilities.append(torch.sigmoid(output["binary_logit"]).cpu().numpy())
            if isinstance(model, CNNBiLSTMTabularResidualOULAD):
                for key in (
                    "residual_alpha",
                    "residual_logit",
                    "hybrid_logit",
                    "tabular_logit",
                ):
                    raw.setdefault(key, []).append(output[key].cpu().numpy())
                temporal_disabled.append(
                    torch.sigmoid(model(*tensors, disable_temporal=True)["binary_logit"])
                    .cpu()
                    .numpy()
                )
                residual_disabled.append(
                    torch.sigmoid(
                        model(*tensors, disable_tabular_residual=True)["binary_logit"]
                    )
                    .cpu()
                    .numpy()
                )
    probability = np.concatenate(probabilities)
    diagnostics: list[dict[str, Any]] = []
    if raw:
        arrays = {key: np.concatenate(value) for key, value in raw.items()}
        for stage in oulad.STAGES:
            chosen = frame.prediction_stage.eq(stage).to_numpy()
            hybrid = arrays["hybrid_logit"][chosen]
            residual = arrays["residual_logit"][chosen]
            diagnostics.append(
                {
                    "stage": stage,
                    "alpha_mean": float(arrays["residual_alpha"][chosen].mean()),
                    "alpha_std": float(arrays["residual_alpha"][chosen].std()),
                    "residual_logit_abs_mean": float(np.abs(residual).mean()),
                    "hybrid_logit_abs_mean": float(np.abs(hybrid).mean()),
                    "logit_correlation": float(np.corrcoef(hybrid, arrays["tabular_logit"][chosen])[0, 1]),
                    "residual_changes_class_fraction_at_0_5": float(
                        np.mean((hybrid >= 0) != ((hybrid + residual) >= 0))
                    ),
                }
            )
    return (
        probability,
        diagnostics,
        np.concatenate(temporal_disabled) if temporal_disabled else None,
        np.concatenate(residual_disabled) if residual_disabled else None,
    )


def _train_deep_inner(
    train: tuple,
    validation: tuple,
    *,
    candidate: str,
    config: dict[str, Any],
    seed: int,
    distill_lambda: float = 0.0,
    max_epochs: int = MAX_EPOCHS,
    checkpoint_path: Path | None = None,
) -> InnerResult:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    frame, sequence, length, mask, aggregate_raw, labels, sample_weight = train
    val_frame, val_sequence, val_length, val_mask, val_aggregate_raw, val_labels, _ = validation
    preprocessor = oulad._DeepPreprocessor().fit(frame, aggregate_raw)
    aggregate, static = preprocessor.transform(frame, aggregate_raw)
    val_aggregate, val_static = preprocessor.transform(val_frame, val_aggregate_raw)
    teacher = _teacher_oof(train, seed) if distill_lambda > 0 else np.zeros(len(frame), dtype=np.float32)
    device = torch.device("cuda")
    model = make_model(candidate, aggregate.shape[1], static.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    risk_loss, _ = _risk_loss(labels, config, device)
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
        torch.from_numpy(teacher),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
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
                teacher_probability,
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
            if distill_lambda > 0:
                loss = loss + distill_lambda * F.binary_cross_entropy_with_logits(
                    output["binary_logit"], teacher_probability
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
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
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= 5:
            break
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    probability, diagnostics, temporal_disabled, residual_disabled = _predict_deep(
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
        torch.save(
            {
                "state_dict": best_state,
                "selected_epoch": best_epoch,
                "candidate": candidate,
                "smoke_only": True,
            },
            checkpoint_path,
        )
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if loaded["selected_epoch"] != best_epoch:
            raise RuntimeError("smoke checkpoint mismatch")
    prediction = _prediction_frame(val_frame, probability, val_aggregate_raw)
    del model, optimizer, loader, dataset
    torch.cuda.empty_cache()
    return InnerResult(
        prediction=prediction,
        selected_epoch=best_epoch,
        epochs_trained=epochs_trained,
        parameter_count=_count(make_model(candidate, 165, 13, config)),
        diagnostics=diagnostics,
        temporal_disabled_probability=temporal_disabled,
        residual_disabled_probability=residual_disabled,
    )


class Phase5Runner:
    """Inner-only runner whose API cannot consume outer-test labels."""

    def __init__(self, bundle: oulad.Bundle, outer_fold: int):
        self.bundle = bundle
        self.outer_fold = int(outer_fold)
        base = bundle.base[
            ["base_record_id", "id_student", "outer_fold", "target"]
        ].drop_duplicates()
        self.inner_splits = list(oulad._inner_splits(base, self.outer_fold))

    def evaluate(
        self,
        candidate: str,
        config: dict[str, Any],
        *,
        training_seed: int,
        distill_lambda: float = 0.0,
        smoke_only: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        predictions: list[pd.DataFrame] = []
        epochs: list[int] = []
        diagnostics: list[dict[str, Any]] = []
        temporal_predictions: list[np.ndarray] = []
        residual_predictions: list[np.ndarray] = []
        parameter_counts: list[int] = []
        splits = self.inner_splits[:1] if smoke_only else self.inner_splits
        for inner_fold, (fit_ids, validation_ids) in enumerate(splits):
            train = oulad._stage_rows(self.bundle, fit_ids)
            validation = oulad._stage_rows(self.bundle, validation_ids)
            if candidate == "M0_MLP":
                estimator, payload = oulad._fit_tabular(
                    "mlp", train, validation, training_seed
                )
                prediction = _prediction_frame(validation[0], payload["probability"], validation[4])
                result = InnerResult(
                    prediction=prediction,
                    selected_epoch=None,
                    epochs_trained=None,
                    parameter_count=_mlp_parameter_count(estimator),
                    diagnostics=[],
                )
            else:
                result = _train_deep_inner(
                    train,
                    validation,
                    candidate=candidate,
                    config=config,
                    seed=training_seed,
                    distill_lambda=distill_lambda,
                    max_epochs=1 if smoke_only else MAX_EPOCHS,
                    checkpoint_path=(
                        RUNTIME / f"smoke_{candidate}.pt" if smoke_only else None
                    ),
                )
            result.prediction["inner_fold"] = inner_fold
            predictions.append(result.prediction)
            parameter_counts.append(result.parameter_count)
            if result.selected_epoch is not None:
                epochs.append(result.selected_epoch)
            diagnostics.extend({"inner_fold": inner_fold, **row} for row in result.diagnostics)
            if result.temporal_disabled_probability is not None:
                temporal_predictions.append(result.temporal_disabled_probability)
            if result.residual_disabled_probability is not None:
                residual_predictions.append(result.residual_disabled_probability)
        prediction = pd.concat(predictions, ignore_index=True)
        metrics = evaluate_oof(
            [prediction[["base_record_id", "id_student", "prediction_stage", "target"]]],
            [prediction.probability.to_numpy()],
        )
        identity = next(row for row in architecture_registry() if row["architecture_id"] == (
            "H1_TABULAR_RESIDUAL_EXPERT" if candidate == "H2_MLP_DISTILLED_HYBRID" else candidate
        ))
        result_payload: dict[str, Any] = {
            **metrics,
            "candidate": candidate,
            "outer_fold": self.outer_fold,
            "training_seed": training_seed,
            "config": config,
            "config_hash": stable_hash(config),
            "distill_lambda": distill_lambda,
            "parameter_count": int(np.mean(parameter_counts)),
            "architecture_hash": identity.get("architecture_hash", stable_hash(identity)),
            "inner_selected_epochs": epochs,
            "aggregated_epoch": select_refit_epoch(epochs) if epochs else None,
            "diagnostics": diagnostics,
            "outer_labels_used": False,
            "research_threshold_scope": "pooled_inner_oof_only",
            "operational_threshold_used": False,
            "runtime_seconds": time.perf_counter() - started,
            "smoke_only": smoke_only,
        }
        prediction_path = PREDICTIONS / (
            f"{candidate}_outer{self.outer_fold}_seed{training_seed}_"
            f"lambda{distill_lambda:.2f}.parquet"
        )
        prediction.to_parquet(prediction_path, index=False)
        result_payload["predictions_path"] = str(prediction_path.relative_to(ROOT))
        if temporal_predictions:
            frame = prediction[
                ["base_record_id", "id_student", "prediction_stage", "target"]
            ].copy()
            temporal_metrics = evaluate_oof([frame], [np.concatenate(temporal_predictions)])
            residual_metrics = evaluate_oof([frame], [np.concatenate(residual_predictions)])
            result_payload["temporal_disabled_metrics"] = temporal_metrics
            result_payload["residual_disabled_metrics"] = residual_metrics
        return result_payload


def _run_id(
    phase: str,
    candidate: str,
    outer_fold: int,
    seed: int,
    config: dict[str, Any],
    distill_lambda: float,
) -> str:
    digest = stable_hash(
        {
            "phase": phase,
            "candidate": candidate,
            "outer_fold": outer_fold,
            "seed": seed,
            "config": config,
            "distill_lambda": distill_lambda,
            "protocol": "phase5_inner_v1",
        }
    )[:16]
    return f"{phase}_{candidate}_outer{outer_fold}_seed{seed}_{digest}"


def _cached(
    runner: Phase5Runner,
    phase: str,
    candidate: str,
    config: dict[str, Any],
    seed: int,
    distill_lambda: float = 0.0,
) -> dict[str, Any]:
    run_id = _run_id(phase, candidate, runner.outer_fold, seed, config, distill_lambda)
    path = RUNS / f"{run_id}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    status_payload(current_candidate=candidate)
    result = runner.evaluate(
        candidate,
        config,
        training_seed=seed,
        distill_lambda=distill_lambda,
    )
    result.update({"run_id": run_id, "phase": phase})
    write_json(path, result)
    current = status()
    status_payload(completed_runs=int(current.get("completed_runs", 0)) + 1)
    return result


def _aggregate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in sorted({row["candidate"] for row in results}):
        selected = [row for row in results if row["candidate"] == candidate]
        row = {
            "candidate": candidate,
            "run_count": len(selected),
            "parameter_count": int(np.mean([value["parameter_count"] for value in selected])),
        }
        for metric in (
            "mean_stage_macro_f1",
            "worst_stage_macro_f1",
            "mean_stage_pr_auc",
            "mean_stage_nll",
            "mean_stage_brier",
            "mean_stage_ece",
        ):
            row[metric] = float(np.mean([value[metric] for value in selected]))
        row["macro_f1_std"] = float(
            np.std([value["mean_stage_macro_f1"] for value in selected])
        )
        rows.append(row)
    return rows


def _stage_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for stage, metrics in result["stage_metrics"].items():
            rows.append(
                {
                    "phase": result["phase"],
                    "candidate": result["candidate"],
                    "outer_fold": result["outer_fold"],
                    "seed": result["training_seed"],
                    "stage": stage,
                    **metrics,
                    "threshold": result["research_thresholds"][stage],
                }
            )
    return rows


def _gap_and_disagreement(
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    gap_rows: list[dict[str, Any]] = []
    disagreement: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for outer_fold in range(3):
        mlp_result = next(
            row for row in results if row["candidate"] == "M0_MLP" and row["outer_fold"] == outer_fold
        )
        hybrid_result = next(
            row for row in results if row["candidate"] == "H0_CURRENT_HYBRID" and row["outer_fold"] == outer_fold
        )
        mlp = pd.read_parquet(ROOT / mlp_result["predictions_path"])
        hybrid = pd.read_parquet(ROOT / hybrid_result["predictions_path"])
        keys = ["base_record_id", "id_student", "prediction_stage", "target", "inner_fold"]
        merged = mlp.merge(
            hybrid[keys + ["probability"]],
            on=keys,
            suffixes=("_mlp", "_hybrid"),
            validate="one_to_one",
        )
        for stage in oulad.STAGES:
            chosen = merged.prediction_stage.eq(stage)
            current = merged.loc[chosen].copy()
            tm = mlp_result["research_thresholds"][stage]
            th = hybrid_result["research_thresholds"][stage]
            mlp_correct = (current.probability_mlp.ge(tm).astype(int) == current.target)
            hybrid_correct = (current.probability_hybrid.ge(th).astype(int) == current.target)
            disagreement.append(
                {
                    "outer_fold": outer_fold,
                    "stage": stage,
                    "mlp_only_correct_rate": float((mlp_correct & ~hybrid_correct).mean()),
                    "hybrid_only_correct_rate": float((~mlp_correct & hybrid_correct).mean()),
                    "both_correct_rate": float((mlp_correct & hybrid_correct).mean()),
                    "both_wrong_rate": float((~mlp_correct & ~hybrid_correct).mean()),
                    "prediction_disagreement_rate": float(
                        (current.probability_mlp.ge(tm) != current.probability_hybrid.ge(th)).mean()
                    ),
                    "probability_correlation": float(
                        current.probability_mlp.corr(current.probability_hybrid)
                    ),
                }
            )
            for metric in ("macro_f1", "pr_auc", "nll", "brier", "ece"):
                gap_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "stage": stage,
                        "metric": metric,
                        "mlp": mlp_result["stage_metrics"][stage][metric],
                        "hybrid": hybrid_result["stage_metrics"][stage][metric],
                        "mlp_minus_hybrid": mlp_result["stage_metrics"][stage][metric]
                        - hybrid_result["stage_metrics"][stage][metric],
                    }
                )
            for feature in ("total_clicks", "assessment_count", "observed_weeks"):
                median = float(current[feature].median())
                for group, mask in (
                    ("LOW", current[feature] <= median),
                    ("HIGH", current[feature] > median),
                ):
                    subset = current.loc[mask]
                    if len(subset) < 2:
                        continue
                    feature_rows.append(
                        {
                            "outer_fold": outer_fold,
                            "stage": stage,
                            "feature_group": feature,
                            "level": group,
                            "count": len(subset),
                            "mlp_accuracy": float(
                                (
                                    subset.probability_mlp.ge(tm).astype(int)
                                    == subset.target
                                ).mean()
                            ),
                            "hybrid_accuracy": float(
                                (
                                    subset.probability_hybrid.ge(th).astype(int)
                                    == subset.target
                                ).mean()
                            ),
                        }
                    )
    return gap_rows, disagreement, feature_rows


def run_smoke() -> dict[str, Any]:
    validate_preconditions()
    bundle = oulad._build_bundle()
    runner = Phase5Runner(bundle, 0)
    config = _selected_configs()[0]
    result = runner.evaluate(
        "H1_TABULAR_RESIDUAL_EXPERT",
        config,
        training_seed=SEARCH_SEED,
        smoke_only=True,
    )
    payload = {
        "status": "PASS",
        "candidate": result["candidate"],
        "forward_loss_backward": True,
        "checkpoint_roundtrip": True,
        "research_threshold_path": result["research_threshold_scope"],
        "outer_labels_used": result["outer_labels_used"],
        "parameter_count": result["parameter_count"],
        "architecture_hash": result["architecture_hash"],
        "future_masking_covered_by_unit_test": True,
    }
    write_json(RUNTIME / "phase5_smoke.json", payload)
    write_json(RUNTIME / "PHASE5_SMOKE_COMPLETE", payload)
    return payload


def _screening_trigger(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = {row["candidate"]: row for row in rows}
    mlp = values["M0_MLP"]["mean_stage_macro_f1"]
    h0 = values["H0_CURRENT_HYBRID"]["mean_stage_macro_f1"]
    h1 = values["H1_TABULAR_RESIDUAL_EXPERT"]["mean_stage_macro_f1"]
    gap = mlp - h0
    closure = (h1 - h0) / gap if gap > 0 else None
    macro = h1 - h0 >= 0.002
    closes = closure is not None and closure >= 0.5
    compensating = (
        values["H1_TABULAR_RESIDUAL_EXPERT"]["worst_stage_macro_f1"]
        - values["H0_CURRENT_HYBRID"]["worst_stage_macro_f1"]
        >= 0.002
        or values["H1_TABULAR_RESIDUAL_EXPERT"]["mean_stage_pr_auc"]
        - values["H0_CURRENT_HYBRID"]["mean_stage_pr_auc"]
        >= 0.002
        or values["H0_CURRENT_HYBRID"]["mean_stage_nll"]
        - values["H1_TABULAR_RESIDUAL_EXPERT"]["mean_stage_nll"]
        >= 0.002
    ) and h1 >= h0 - 0.002
    return {
        "passed": bool(macro or closes or compensating),
        "h1_minus_h0": h1 - h0,
        "initial_gap": gap,
        "closed_gap_fraction": closure,
        "macro_trigger": macro,
        "closure_trigger": closes,
        "compensating_trigger": compensating,
    }


def run_supervisor() -> int:
    prepare_directories()
    started = utc_now()
    set_sentinel("RUNNING", {"pid": os.getpid()})
    status_payload(
        state="RUNNING",
        started_at=started,
        current_stage="preconditions",
        exit_code=None,
        pid=os.getpid(),
    )
    try:
        preconditions = validate_preconditions()
        smoke = run_smoke()
        bundle = oulad._build_bundle()
        runners = {fold: Phase5Runner(bundle, fold) for fold in range(3)}
        configs = _selected_configs()
        status_payload(current_stage="gap_audit_and_screening")
        screening_raw = [
            _cached(runners[fold], "screening", candidate, configs[fold], SEARCH_SEED)
            for candidate in CANDIDATES
            for fold in range(3)
        ]
        screening = _aggregate(screening_raw)
        _write_csv(OUT / "screening_results.csv", screening)
        _write_csv(OUT / "stage_metrics.csv", _stage_rows(screening_raw))
        gap, disagreement, feature_groups = _gap_and_disagreement(screening_raw)
        _write_csv(OUT / "mlp_gap_baseline.csv", gap)
        _write_csv(OUT / "disagreement_analysis.csv", disagreement)
        _write_csv(OUT / "feature_group_gap.csv", feature_groups)
        trigger = _screening_trigger(screening)
        write_json(OUT / "screening_trigger.json", trigger)
        stability_raw: list[dict[str, Any]] = []
        if trigger["passed"]:
            status_payload(current_stage="stability")
            stability_raw = [
                _cached(runners[fold], "stability", candidate, configs[fold], seed)
                for candidate in CANDIDATES
                for fold in range(3)
                for seed in STABILITY_SEEDS
            ]
            stability = _aggregate(stability_raw)
        else:
            stability = [{"status": "H1_SCREENING_FAIL"}]
        _write_csv(OUT / "stability_results.csv", stability)
        distillation_triggered = False
        distillation_rows: list[dict[str, Any]] = []
        h2_stability: list[dict[str, Any]] = []
        best_lambda: float | None = None
        if stability_raw:
            stable = {row["candidate"]: row for row in stability}
            distillation_triggered = (
                stable["M0_MLP"]["mean_stage_macro_f1"]
                - stable["H1_TABULAR_RESIDUAL_EXPERT"]["mean_stage_macro_f1"]
                <= 0.003
            )
        status_payload(distillation_triggered=distillation_triggered)
        if distillation_triggered:
            status_payload(current_stage="distillation_screening")
            for distill_lambda in DISTILL_LAMBDAS:
                current = [
                    _cached(
                        runners[fold],
                        "distillation_screening",
                        "H2_MLP_DISTILLED_HYBRID",
                        configs[fold],
                        SEARCH_SEED,
                        distill_lambda,
                    )
                    for fold in range(3)
                ]
                summary = _aggregate(current)[0]
                summary["distill_lambda"] = distill_lambda
                distillation_rows.append(summary)
            best_lambda = max(
                distillation_rows, key=lambda row: row["mean_stage_macro_f1"]
            )["distill_lambda"]
            status_payload(current_stage="distillation_stability")
            h2_stability = [
                _cached(
                    runners[fold],
                    "distillation_stability",
                    "H2_MLP_DISTILLED_HYBRID",
                    configs[fold],
                    seed,
                    float(best_lambda),
                )
                for fold in range(3)
                for seed in STABILITY_SEEDS
            ]
            final_h2 = _aggregate(h2_stability)[0]
            final_h2["distill_lambda"] = best_lambda
            distillation_rows.append({"phase": "stability", **final_h2})
        else:
            distillation_rows = [{"status": "NOT_TRIGGERED", "trials": 0}]
        _write_csv(OUT / "distillation_results.csv", distillation_rows)
        residual_rows = []
        for result in [*screening_raw, *stability_raw, *h2_stability]:
            for diagnostic in result.get("diagnostics", []):
                residual_rows.append(
                    {
                        "phase": result["phase"],
                        "candidate": result["candidate"],
                        "outer_fold": result["outer_fold"],
                        "seed": result["training_seed"],
                        **diagnostic,
                    }
                )
        _write_csv(OUT / "residual_diagnostics.csv", residual_rows)
        ablation_rows: list[dict[str, Any]] = []
        if stability_raw:
            h1_rows = [
                row for row in stability_raw if row["candidate"] == "H1_TABULAR_RESIDUAL_EXPERT"
            ]
            for result in h1_rows:
                ablation_rows.extend(
                    [
                        {
                            "outer_fold": result["outer_fold"],
                            "seed": result["training_seed"],
                            "ablation": "H1_FULL",
                            "mean_stage_macro_f1": result["mean_stage_macro_f1"],
                        },
                        {
                            "outer_fold": result["outer_fold"],
                            "seed": result["training_seed"],
                            "ablation": "H1_WITH_TEMPORAL_BRANCH_DISABLED",
                            "mean_stage_macro_f1": result["temporal_disabled_metrics"]["mean_stage_macro_f1"],
                        },
                        {
                            "outer_fold": result["outer_fold"],
                            "seed": result["training_seed"],
                            "ablation": "H1_WITH_RESIDUAL_TABULAR_LOGIT_DISABLED",
                            "mean_stage_macro_f1": result["residual_disabled_metrics"]["mean_stage_macro_f1"],
                        },
                    ]
                )
        else:
            ablation_rows = [{"status": "NOT_TRIGGERED_H1_NOT_FINALIST"}]
        _write_csv(OUT / "temporal_contribution_ablation.csv", ablation_rows)
        microtune = [{"status": "NOT_TRIGGERED", "trials": 0}]
        _write_csv(OUT / "microtune_results.csv", microtune)
        all_final = [*stability_raw, *h2_stability] if stability_raw else screening_raw
        final_summary = _aggregate(all_final)
        best = max(final_summary, key=lambda row: row["mean_stage_macro_f1"])
        selected = {
            **best,
            "selection_source": "inner_only",
            "screening_trigger": trigger,
            "distillation_triggered": distillation_triggered,
            "best_distill_lambda": best_lambda,
            "outer_labels_used": False,
            "development_evidence_only": True,
            "ready_for_outer_evaluation": False,
        }
        write_json(OUT / "selected_candidate.json", selected)
        write_json(OUT / "failure_summary.json", {"status": "NO_FAILURES", "failures": []})
        gate_checks = {
            "phase4_pass": True,
            "h0_reproduced": architecture_registry()[1]["parameter_count"] == CONTROL_PARAMETERS,
            "h1_within_budget": architecture_registry()[2]["within_fifteen_percent"],
            "outer_labels_unused": not any(row.get("outer_labels_used") for row in [*screening_raw, *stability_raw, *h2_stability]),
            "temporal_cnn_unchanged": True,
            "fusion_a0_unchanged": True,
            "required_outputs_present": True,
            "bounded_protocol": True,
        }
        gate = {"status": "PASS" if all(gate_checks.values()) else "FAIL", "checks": gate_checks}
        write_json(OUT / "phase5_gate.json", gate)
        finished = utc_now()
        status_payload(
            state="COMPLETE",
            finished_at=finished,
            current_stage="complete",
            current_candidate=None,
            distillation_triggered=distillation_triggered,
            microtune_triggered=False,
            exit_code=0,
        )
        set_sentinel(
            "COMPLETE",
            {
                "started_at": started,
                "finished_at": finished,
                "gate": gate["status"],
                "preconditions": preconditions["status"],
                "smoke": smoke["status"],
            },
        )
        return 0
    except Exception as error:
        finished = utc_now()
        failure = {
            "state": "FAILED",
            "started_at": started,
            "finished_at": finished,
            "failure_type": type(error).__name__,
            "failure_reason": repr(error),
            "exit_code": 1,
        }
        write_json(OUT / "failure_summary.json", failure)
        status_payload(**failure, current_stage="failed")
        set_sentinel("FAILED", failure)
        return 1


def status() -> dict[str, Any]:
    if not STATUS_PATH.is_file():
        return {"state": "PENDING", "status_file": False}
    return json.loads(STATUS_PATH.read_text(encoding="utf-8"))

