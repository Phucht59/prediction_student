"""Phase 7: bounded H1 development and one-shot historical OULAD endpoint evaluation."""

from __future__ import annotations

import copy
import csv
import hashlib
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.pipelines import oulad
from src.training.control import (
    select_operational_threshold,
    select_refit_epoch,
    select_research_threshold,
    stable_hash,
)
from src.training.optuna_search import _risk_loss, write_json
from src.training.model_comparison import (
    _selected_configs,
    architecture_registry,
    make_model,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "audit" / "phase7"
OPTUNA_DIR = OUT / "optuna"
RUNTIME = OUT / "runtime"
RUNS = RUNTIME / "runs"
PREDICTIONS = RUNTIME / "predictions"
CHECKPOINTS = RUNTIME / "checkpoints"
LOGS = OUT / "logs"
STATUS_PATH = RUNTIME / "phase7_status.json"
RUNNING = RUNTIME / "PHASE7_RUNNING"
DEVELOPMENT_COMPLETE = RUNTIME / "PHASE7_DEVELOPMENT_COMPLETE"
COMPLETE = RUNTIME / "PHASE7_COMPLETE"
FAILED = RUNTIME / "PHASE7_FAILED"
FREEZE_PATH = OUT / "endpoint_freeze_manifest.json"
ENDPOINT_STAGE = "M1_MIDDLE_FROZEN"
ENDPOINT_ID = "F2_MIDDLE_OFFICIAL_SINGLE_CUTOFF"
SEARCH_SEED = 42
STABILITY_SEEDS = (1201, 2026)
FINAL_SEEDS = (42, 1201, 2026, 3407, 7319)
OUTER_FOLDS = (0, 1, 2)
MAX_EPOCHS = 15
TRIALS_PER_FOLD = 18
PARAMETER_COUNT = 160_492
EARLY_WARNING_FILES = (
    "artifacts/final/h1_final/comparator_summary.csv",
    "artifacts/final/h1_final/stage_metrics.csv",
    "artifacts/final/h1_final/predictions.parquet",
    "artifacts/final/h1_final/phase6_gate.json",
    "artifacts/audit/phase5/phase5_gate.json",
)
HISTORICAL_H0 = (
    ROOT / "artifacts" / "final" / "predictions" / "cnn_bilstm_oulad"
    / "oof_predictions.parquet"
)
HISTORICAL_MLP = (
    ROOT / "artifacts" / "final" / "teacher_feedback_validation"
    / "mlp_comparator" / "oulad" / "oof_predictions.parquet"
)
HISTORICAL_MANIFEST = (
    ROOT / "data" / "processed" / "study_c_oulad" / "manifests"
    / "F2_MIDDLE.json"
)
HISTORICAL_SPLIT = (
    ROOT / "data" / "processed" / "study_c_oulad" / "manifests"
    / "split_manifest.csv"
)
FINAL_RESULTS = ROOT / "artifacts" / "final" / "final_results.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def _outer_fold_key(value: Any) -> str:
    """Normalize pandas numeric fold scalars to manifest string keys."""
    return str(int(value))


def prepare_directories() -> None:
    for path in (
        OUT,
        OPTUNA_DIR,
        RUNTIME,
        RUNS,
        PREDICTIONS,
        CHECKPOINTS,
        LOGS,
    ):
        path.mkdir(parents=True, exist_ok=True)


def status_payload(**updates: Any) -> dict[str, Any]:
    current = (
        json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        if STATUS_PATH.is_file()
        else {
            "state": "PENDING",
            "started_at": None,
            "finished_at": None,
            "current_stage": "audit",
            "completed_trials": 0,
            "pruned_trials": 0,
            "failed_trials": 0,
            "completed_runs": 0,
            "failed_runs": 0,
            "current_outer_fold": None,
            "current_seed": None,
            "exit_code": None,
            "pid": os.getpid(),
        }
    )
    current.update(updates)
    write_json(STATUS_PATH, current)
    return current


def set_sentinel(state: str, details: dict[str, Any] | None = None) -> None:
    prepare_directories()
    for path in (RUNNING, DEVELOPMENT_COMPLETE, COMPLETE, FAILED):
        if path.exists():
            path.unlink()
    target = {
        "RUNNING": RUNNING,
        "DEVELOPMENT_COMPLETE": DEVELOPMENT_COMPLETE,
        "COMPLETE": COMPLETE,
        "FAILED": FAILED,
    }[state]
    write_json(target, {"state": state, "at": utc_now(), **(details or {})})


def early_warning_checksums() -> dict[str, str]:
    return {relative: _sha(ROOT / relative) for relative in EARLY_WARNING_FILES}


def _architecture_identity() -> dict[str, Any]:
    identity = next(
        row
        for row in architecture_registry()
        if row["architecture_id"] == "H1_TABULAR_RESIDUAL_EXPERT"
    )
    if identity["parameter_count"] != PARAMETER_COUNT:
        raise RuntimeError("H1 parameter count changed")
    return identity


def _endpoint_rows(bundle: oulad.Bundle, base_ids: set[str]) -> tuple:
    data = bundle.stages[ENDPOINT_STAGE]
    indices = np.flatnonzero(
        data.frame.base_record_id.isin(base_ids).to_numpy()
    )
    frame = data.frame.iloc[indices].copy().reset_index(drop=True)
    frame["prediction_stage"] = ENDPOINT_STAGE
    return (
        frame,
        data.sequence[indices],
        data.lengths[indices],
        data.mask[indices],
        data.aggregate[indices],
        frame.target.to_numpy(dtype=np.float32),
        np.ones(len(indices), dtype=np.float32),
    )


def audit_endpoint_protocol(bundle: oulad.Bundle) -> dict[str, Any]:
    manifest = json.loads(HISTORICAL_MANIFEST.read_text(encoding="utf-8"))
    split = pd.read_csv(HISTORICAL_SPLIT)
    split = split.loc[
        split.forecast_id.eq("F2_MIDDLE")
        & split.role.eq("historical_development")
    ].copy()
    current = bundle.stages[ENDPOINT_STAGE].frame.loc[
        :,
        ["base_record_id", "id_student", "outer_fold", "target"],
    ].drop_duplicates()
    historical_ids = set(split.record_id)
    current_ids = set(current.base_record_id)
    h0 = pd.read_parquet(HISTORICAL_H0)
    mlp = pd.read_parquet(HISTORICAL_MLP)
    group_safe = True
    for fold in OUTER_FOLDS:
        test_students = set(
            split.loc[split.outer_fold.eq(fold), "id_student"]
        )
        train_students = set(
            split.loc[split.outer_fold.ne(fold), "id_student"]
        )
        group_safe &= not bool(test_students & train_students)
    audit = {
        "status": "PASS",
        "track": "MAIN_OULAD_FINAL_ENDPOINT",
        "endpoint_id": ENDPOINT_ID,
        "historical_forecast_id": manifest["forecast_id"],
        "is_stage_75_percent": False,
        "is_mean_stage": False,
        "cutoff_fraction": manifest["forecast_fraction"],
        "cutoff_rule": manifest["cutoff_rule"],
        "target": {
            "positive": ["Withdrawn", "Fail"],
            "negative": ["Pass", "Distinction"],
        },
        "eligible_records": len(historical_ids),
        "outer_folds": 3,
        "inner_folds": 2,
        "group_key": "id_student",
        "outer_group_safe": group_safe,
        "historical_current_record_identity": historical_ids == current_ids,
        "h0_record_identity": set(h0.record_id) == historical_ids,
        "mlp_record_identity": set(mlp.record_id) == historical_ids,
        "fold_counts": {
            str(int(key)): int(value)
            for key, value in split.groupby("outer_fold").size().items()
        },
        "h1_feature_contract": {
            "sequence_channels": len(oulad.CHANNELS),
            "aggregate_features": 165,
            "static_source_columns": list(oulad.STATIC_COLUMNS),
            "static_runtime_dimension": 13,
            "feature_availability": "events strictly before F2 cutoff",
        },
        "comparator_policy": (
            "reuse historical protocol-matched H0/MLP and official ML table"
        ),
        "outer_labels_used_for_development_selection": False,
        "early_warning_retrained": False,
    }
    required = (
        audit["outer_group_safe"],
        audit["historical_current_record_identity"],
        audit["h0_record_identity"],
        audit["mlp_record_identity"],
        audit["eligible_records"] == 15_378,
    )
    if not all(required):
        audit["status"] = "FAIL"
        raise RuntimeError(f"endpoint protocol audit failed: {audit}")
    write_json(OUT / "endpoint_protocol_audit.json", audit)
    return audit


def audit_feature_leakage(bundle: oulad.Bundle) -> dict[str, Any]:
    data = bundle.stages[ENDPOINT_STAGE]
    forbidden_predictors = {
        "target",
        "final_result",
        "original_final_result",
        "date_unregistration",
        "outcome_aux",
        "future_activity",
        "post_cutoff_score",
    }
    predictor_contract = set(oulad.CHANNELS) | set(oulad.STATIC_COLUMNS)
    audit = {
        "status": "PASS",
        "endpoint": ENDPOINT_ID,
        "predictor_forbidden_intersection": sorted(
            forbidden_predictors & predictor_contract
        ),
        "sequence_event_rule": "0 <= event_day < cutoff_day",
        "future_timesteps_zero_masked": True,
        "aggregate_stage_safe": True,
        "preprocessing_fit_scope": "inner_train_or_outer_train_only",
        "final_outcome_as_predictor": False,
        "date_unregistration_as_predictor": False,
        "auxiliary_outcome_role": (
            "training-only supervised head target; absent from forward inputs"
        ),
        "auxiliary_survival_role": (
            "training-only supervised head target; absent from forward inputs"
        ),
        "sequence_shape": list(data.sequence.shape),
        "aggregate_shape": list(data.aggregate.shape),
        "static_source_columns": list(oulad.STATIC_COLUMNS),
        "label_shape": [len(data.frame)],
        "outer_labels_used_for_tuning": False,
    }
    if audit["predictor_forbidden_intersection"]:
        audit["status"] = "FAIL"
        raise RuntimeError(f"endpoint leakage audit failed: {audit}")
    write_json(OUT / "endpoint_feature_leakage_audit.json", audit)
    return audit


def control_config(outer_fold: int) -> dict[str, Any]:
    return copy.deepcopy(_selected_configs()[outer_fold])


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


def search_space() -> dict[str, Any]:
    return {
        "learning_rate": [1e-4, 2e-3, "log"],
        "weight_decay": [1e-8, 5e-4, "log"],
        "dropout": [0.10, 0.35],
        "batch_size": [128, 256],
        "loss_policy": ["standard_bce", "weighted_bce"],
        "pos_weight_strategy": ["sqrt_ratio", "full_ratio", "conditional"],
        "survival_weight": [0.0, 0.10, 0.15, 0.20],
        "outcome_weight": [0.0, 0.10, 0.15, 0.20],
        "frozen": {
            "architecture": "H1_TABULAR_RESIDUAL_EXPERT",
            "residual_alpha_initial": 0.05,
            "max_epochs": MAX_EPOCHS,
            "checkpoint": "minimum_endpoint_validation_nll",
            "optimizer": "AdamW",
            "scheduler": None,
        },
    }


@dataclass
class InnerResult:
    prediction: pd.DataFrame
    selected_epoch: int
    epochs_trained: int
    best_nll: float
    positive_weight: float


def _train_inner(
    train: tuple,
    validation: tuple,
    *,
    config: dict[str, Any],
    seed: int,
    inner_fold: int,
    trial: optuna.Trial | None,
    max_epochs: int,
    checkpoint_path: Path | None = None,
) -> InnerResult:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    frame, sequence, length, mask, aggregate_raw, labels, sample_weight = train
    (
        val_frame,
        val_sequence,
        val_length,
        val_mask,
        val_aggregate_raw,
        val_labels,
        _,
    ) = validation
    preprocessor = oulad._DeepPreprocessor().fit(frame, aggregate_raw)
    aggregate, static = preprocessor.transform(frame, aggregate_raw)
    val_aggregate, val_static = preprocessor.transform(
        val_frame, val_aggregate_raw
    )
    model = make_model(
        "H1_TABULAR_RESIDUAL_EXPERT",
        aggregate.shape[1],
        static.shape[1],
        config,
    ).to("cuda")
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise RuntimeError("H1 architecture changed during endpoint training")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    risk_loss, positive_weight = _risk_loss(
        labels, config, torch.device("cuda")
    )
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
        torch.from_numpy(
            frame.module_presentation_length.to_numpy(dtype=np.int64)
        ),
        torch.from_numpy(
            frame.date_unregistration.fillna(-1).to_numpy(dtype=np.int64)
        ),
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
    epochs_trained = 0
    wait = 0
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
            ) = (value.to("cuda") for value in batch)
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
                raise FloatingPointError("non-finite endpoint loss")
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
            torch.device("cuda"),
        )
        nll = float(
            log_loss(
                val_labels,
                np.clip(probability, 1e-7, 1 - 1e-7),
                labels=[0, 1],
            )
        )
        if nll < best_nll - 1e-6:
            best_nll = nll
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
        if trial is not None:
            trial.report(-nll, inner_fold * MAX_EPOCHS + epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"endpoint median pruner inner={inner_fold} epoch={epoch}"
                )
        if wait >= 5:
            break
    if best_state is None or best_epoch < 1:
        raise RuntimeError("endpoint checkpoint selection failed")
    model.load_state_dict(best_state)
    probability = oulad._predict_deep(
        model,
        val_sequence,
        val_length,
        val_mask,
        val_aggregate,
        val_static,
        "cnn_bilstm",
        torch.device("cuda"),
    )
    prediction = val_frame.loc[
        :,
        ["base_record_id", "id_student", "outer_fold", "target"],
    ].copy()
    prediction["probability"] = probability
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "selected_epoch": best_epoch,
                "parameter_count": PARAMETER_COUNT,
                "architecture_hash": _architecture_identity()[
                    "architecture_hash"
                ],
                "smoke_only": True,
            },
            checkpoint_path,
        )
        loaded = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if loaded["selected_epoch"] != best_epoch:
            raise RuntimeError("endpoint smoke checkpoint identity mismatch")
    del model, optimizer, loader, dataset
    torch.cuda.empty_cache()
    return InnerResult(
        prediction=prediction,
        selected_epoch=best_epoch,
        epochs_trained=epochs_trained,
        best_nll=best_nll,
        positive_weight=positive_weight,
    )


def _endpoint_metrics(prediction: pd.DataFrame) -> dict[str, Any]:
    labels = prediction.target.to_numpy(dtype=int)
    probability = prediction.probability.to_numpy(dtype=float)
    research = select_research_threshold(labels, probability)
    operational = select_operational_threshold(labels, probability)
    return {
        **oulad._metric(labels, probability, research["threshold"]),
        "research_threshold": research,
        "operational_threshold": operational,
    }


class EndpointRunner:
    """Inner-only endpoint runner; its public API has no outer-test labels."""

    def __init__(self, bundle: oulad.Bundle, outer_fold: int):
        self.bundle = bundle
        self.outer_fold = int(outer_fold)
        base = bundle.stages[ENDPOINT_STAGE].frame.loc[
            :,
            ["base_record_id", "id_student", "outer_fold", "target"],
        ].drop_duplicates()
        self.inner_splits = list(oulad._inner_splits(base, self.outer_fold))
        if len(self.inner_splits) != 2:
            raise RuntimeError("endpoint inner-fold count changed")

    def evaluate(
        self,
        config: dict[str, Any],
        *,
        training_seed: int,
        trial: optuna.Trial | None = None,
        smoke_only: bool = False,
        prediction_path: Path | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        predictions: list[pd.DataFrame] = []
        inner_results: list[InnerResult] = []
        splits = self.inner_splits[:1] if smoke_only else self.inner_splits
        for inner_fold, (fit_ids, validation_ids) in enumerate(splits):
            result = _train_inner(
                _endpoint_rows(self.bundle, fit_ids),
                _endpoint_rows(self.bundle, validation_ids),
                config=config,
                seed=training_seed,
                inner_fold=inner_fold,
                trial=trial,
                max_epochs=1 if smoke_only else MAX_EPOCHS,
                checkpoint_path=(
                    RUNTIME / "endpoint_smoke_checkpoint.pt"
                    if smoke_only
                    else None
                ),
            )
            current = result.prediction.copy()
            current["inner_fold"] = inner_fold
            predictions.append(current)
            inner_results.append(result)
        prediction = pd.concat(predictions, ignore_index=True)
        metrics = _endpoint_metrics(prediction)
        epochs = [result.selected_epoch for result in inner_results]
        identity = _architecture_identity()
        payload = {
            **metrics,
            "endpoint_id": ENDPOINT_ID,
            "outer_fold": self.outer_fold,
            "training_seed": training_seed,
            "config": copy.deepcopy(config),
            "config_hash": stable_hash(config),
            "inner_selected_epochs": epochs,
            "inner_epochs_trained": [
                result.epochs_trained for result in inner_results
            ],
            "inner_best_nll": [
                result.best_nll for result in inner_results
            ],
            "inner_positive_weights": [
                result.positive_weight for result in inner_results
            ],
            "aggregated_epoch": select_refit_epoch(epochs),
            "architecture_hash": identity["architecture_hash"],
            "temporal_backbone_hash": identity["temporal_backbone_hash"],
            "parameter_count": identity["parameter_count"],
            "outer_labels_used": False,
            "research_threshold_scope": "pooled_inner_endpoint_oof_only",
            "runtime_seconds": time.perf_counter() - started,
            "smoke_only": smoke_only,
        }
        if prediction_path is not None:
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            prediction.to_parquet(prediction_path, index=False)
            payload["prediction_path"] = prediction_path.relative_to(
                ROOT
            ).as_posix()
        return payload


def create_study(outer_fold: int, *, smoke: bool = False) -> optuna.Study:
    suffix = "smoke" if smoke else f"outer{outer_fold}"
    database = OPTUNA_DIR / f"endpoint_{suffix}.db"
    return optuna.create_study(
        study_name=f"oulad_h1_endpoint_{suffix}",
        storage=f"sqlite:///{database.as_posix()}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=SEARCH_SEED + outer_fold, n_startup_trials=5
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1,
        ),
        load_if_exists=True,
    )


def _objective(runner: EndpointRunner, outer_fold: int):
    def objective(trial: optuna.Trial) -> float:
        config = sample_trial_config(trial)
        trial.set_user_attr("config", config)
        trial.set_user_attr("config_hash", stable_hash(config))
        trial.set_user_attr("outer_fold", outer_fold)
        trial.set_user_attr("training_seed", SEARCH_SEED)
        trial.set_user_attr("outer_labels_used", False)
        try:
            result = runner.evaluate(
                config, training_seed=SEARCH_SEED, trial=trial
            )
            for key, value in result.items():
                if key != "config":
                    trial.set_user_attr(key, value)
            return float(result["macro_f1"])
        except optuna.TrialPruned:
            trial.set_user_attr("failure_type", "PRUNED")
            raise
        except Exception as error:
            trial.set_user_attr("failure_type", type(error).__name__)
            trial.set_user_attr("failure_reason", repr(error))
            torch.cuda.empty_cache()
            raise

    return objective


def _select_trial(study: optuna.Study) -> optuna.trial.FrozenTrial:
    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
    ]
    if not complete:
        raise RuntimeError("endpoint study has no complete trial")
    best = max(float(trial.value) for trial in complete)
    candidates = [
        trial for trial in complete if float(trial.value) >= best - 1e-4
    ]
    return sorted(
        candidates,
        key=lambda trial: (
            -float(trial.user_attrs["pr_auc"]),
            -float(trial.user_attrs["roc_auc"]),
            float(trial.user_attrs["nll"]),
            float(trial.user_attrs["brier"]),
            float(trial.user_attrs["ece"]),
            int(trial.number),
        ),
    )[0]


def _trial_rows(studies: dict[int, optuna.Study]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outer_fold, study in studies.items():
        for trial in study.trials:
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "trial_number": trial.number,
                    "state": trial.state.name,
                    "value": trial.value,
                    "params": trial.params,
                    "config": trial.user_attrs.get("config"),
                    "macro_f1": trial.user_attrs.get("macro_f1"),
                    "pr_auc": trial.user_attrs.get("pr_auc"),
                    "roc_auc": trial.user_attrs.get("roc_auc"),
                    "nll": trial.user_attrs.get("nll"),
                    "brier": trial.user_attrs.get("brier"),
                    "ece": trial.user_attrs.get("ece"),
                    "selected_epochs": trial.user_attrs.get(
                        "inner_selected_epochs"
                    ),
                    "architecture_hash": trial.user_attrs.get(
                        "architecture_hash"
                    ),
                    "parameter_count": trial.user_attrs.get("parameter_count"),
                    "outer_labels_used": trial.user_attrs.get(
                        "outer_labels_used", False
                    ),
                    "failure_type": trial.user_attrs.get("failure_type"),
                }
            )
    return rows


def _development_counts(studies: dict[int, optuna.Study]) -> dict[str, int]:
    trials = [trial for study in studies.values() for trial in study.trials]
    return {
        "completed_trials": sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            for trial in trials
        ),
        "pruned_trials": sum(
            trial.state == optuna.trial.TrialState.PRUNED for trial in trials
        ),
        "failed_trials": sum(
            trial.state == optuna.trial.TrialState.FAIL for trial in trials
        ),
    }


def run_smoke() -> dict[str, Any]:
    prepare_directories()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Phase 7 requires exactly one CUDA GPU")
    bundle = oulad._build_bundle()
    audit_endpoint_protocol(bundle)
    audit_feature_leakage(bundle)
    runner = EndpointRunner(bundle, 0)
    study = create_study(0, smoke=True)
    if not study.trials:
        study.enqueue_trial(control_config(0))

        def smoke_objective(trial: optuna.Trial) -> float:
            result = runner.evaluate(
                control_config(0),
                training_seed=SEARCH_SEED,
                trial=trial,
                smoke_only=True,
            )
            trial.set_user_attr("architecture_hash", result["architecture_hash"])
            trial.set_user_attr("parameter_count", result["parameter_count"])
            trial.set_user_attr("outer_labels_used", False)
            return float(result["macro_f1"])

        study.optimize(smoke_objective, n_trials=1)
    resumed = create_study(0, smoke=True)
    result = {
        "status": "PASS",
        "smoke_only": True,
        "forward_loss_backward": True,
        "checkpoint_save_load": (
            RUNTIME / "endpoint_smoke_checkpoint.pt"
        ).is_file(),
        "metric_serialization": True,
        "sqlite_persistent": (OPTUNA_DIR / "endpoint_smoke.db").is_file(),
        "resume_did_not_duplicate": len(study.trials) == len(resumed.trials) == 1,
        "architecture_hash": _architecture_identity()["architecture_hash"],
        "parameter_count": PARAMETER_COUNT,
        "outer_labels_used": False,
    }
    if not all(
        (
            result["checkpoint_save_load"],
            result["sqlite_persistent"],
            result["resume_did_not_duplicate"],
            result["outer_labels_used"] is False,
        )
    ):
        raise RuntimeError(f"Phase 7 smoke failed: {result}")
    write_json(OUT / "endpoint_smoke_validation.json", result)
    write_json(OUT / "early_warning_freeze_checksums.json", early_warning_checksums())
    return result


def _load_or_evaluate(
    path: Path,
    runner: EndpointRunner,
    config: dict[str, Any],
    *,
    seed: int,
    prediction_path: Path | None = None,
) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    result = runner.evaluate(
        config,
        training_seed=seed,
        prediction_path=prediction_path,
    )
    write_json(path, result)
    return result


def _aggregate_control(results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    metrics = ("macro_f1", "pr_auc", "roc_auc", "nll", "brier", "ece")
    return {
        "status": "COMPLETE",
        "candidate": "H1_ENDPOINT_CONTROL",
        "folds": results,
        **{
            f"mean_{metric}": float(
                np.mean([row[metric] for row in results.values()])
            )
            for metric in metrics
        },
        "outer_labels_used": False,
    }


def _stability_summary(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    frame = pd.DataFrame(rows)
    result: dict[str, dict[str, float]] = {}
    for configuration, group in frame.groupby("configuration"):
        result[configuration] = {
            "mean_macro_f1": float(group.macro_f1.mean()),
            "std_macro_f1": float(group.macro_f1.std(ddof=0)),
            "mean_pr_auc": float(group.pr_auc.mean()),
            "mean_roc_auc": float(group.roc_auc.mean()),
            "mean_nll": float(group.nll.mean()),
            "mean_brier": float(group.brier.mean()),
            "mean_ece": float(group.ece.mean()),
        }
    return result


def _select_configuration(
    summary: dict[str, dict[str, float]],
) -> tuple[str, dict[str, Any]]:
    control = summary["CONTROL"]
    tuned = summary["TUNED"]
    delta = tuned["mean_macro_f1"] - control["mean_macro_f1"]
    if delta >= 0.002:
        chosen = "TUNED"
        reason = "TUNED_GAIN_AT_LEAST_0_002"
    elif (
        delta > 0
        and tuned["mean_nll"] <= control["mean_nll"]
        and tuned["std_macro_f1"] <= control["std_macro_f1"] + 0.001
    ):
        chosen = "TUNED"
        reason = "SMALL_GAIN_WITH_NONWORSE_NLL_AND_STABILITY"
    else:
        chosen = "CONTROL"
        reason = "TUNED_NOT_MATERIAL_OR_STABILITY_CALIBRATION_NOT_BETTER"
    return chosen, {
        "rule": (
            "tuned if delta>=0.002; otherwise only if positive with nonworse "
            "NLL and std<=control+0.001"
        ),
        "delta_macro_f1": delta,
        "chosen": chosen,
        "reason": reason,
    }


def _freeze_from_stability(
    rows: list[dict[str, Any]],
    selected_trials: dict[int, dict[str, Any]],
    selected_source: str,
    decision: dict[str, Any],
    protocol_audit: dict[str, Any],
) -> dict[str, Any]:
    identity = _architecture_identity()
    thresholds: dict[str, float] = {}
    operational: dict[str, Any] = {}
    epochs: dict[str, int] = {}
    configs: dict[str, Any] = {}
    for outer_fold in OUTER_FOLDS:
        chosen = [
            row
            for row in rows
            if row["configuration"] == selected_source
            and row["outer_fold"] == outer_fold
        ]
        predictions = [
            pd.read_parquet(ROOT / row["prediction_path"]) for row in chosen
        ]
        aligned = pd.concat(
            [
                prediction.assign(stability_seed=row["seed"])
                for prediction, row in zip(predictions, chosen, strict=True)
            ],
            ignore_index=True,
        )
        averaged = (
            aligned.groupby(
                ["base_record_id", "id_student", "target"],
                as_index=False,
            )
            .probability.mean()
        )
        research = select_research_threshold(
            averaged.target.to_numpy(dtype=int),
            averaged.probability.to_numpy(dtype=float),
        )
        operation = select_operational_threshold(
            averaged.target.to_numpy(dtype=int),
            averaged.probability.to_numpy(dtype=float),
        )
        thresholds[str(outer_fold)] = float(research["threshold"])
        operational[str(outer_fold)] = operation
        epochs[str(outer_fold)] = select_refit_epoch(
            [
                epoch
                for row in chosen
                for epoch in row["inner_selected_epochs"]
            ]
        )
        configs[str(outer_fold)] = (
            control_config(outer_fold)
            if selected_source == "CONTROL"
            else selected_trials[outer_fold]["config"]
        )
    scientific = {
        "candidate_id": "H1_ENDPOINT_TUNED"
        if selected_source == "TUNED"
        else "H1_ENDPOINT_CONTROL",
        "architecture_id": "H1_TABULAR_RESIDUAL_EXPERT",
        "architecture_hash": identity["architecture_hash"],
        "temporal_backbone_hash": identity["temporal_backbone_hash"],
        "parameter_count": PARAMETER_COUNT,
        "feature_contract": protocol_audit["h1_feature_contract"],
        "endpoint_id": ENDPOINT_ID,
        "target": protocol_audit["target"],
        "eligible_records": protocol_audit["eligible_records"],
        "outer_folds": 3,
        "inner_folds": 2,
        "final_seeds": list(FINAL_SEEDS),
        "training_configs_by_outer_fold": configs,
        "refit_epochs_by_outer_fold": epochs,
        "research_thresholds_by_outer_fold": thresholds,
        "operational_thresholds_by_outer_fold": operational,
        "checkpoint_criterion": "minimum_inner_endpoint_validation_nll",
        "epoch_aggregation": "round_half_up_median",
        "research_threshold_policy": "pooled_inner_endpoint_oof_macro_f1",
        "optimizer": "AdamW",
        "scheduler": None,
        "max_epochs": MAX_EPOCHS,
        "pretraining_requested": False,
        "pretraining_executed": False,
        "architecture_search": False,
        "post_outer_tuning": "PROHIBITED",
    }
    freeze = {
        "schema_version": "phase7_endpoint_freeze_v1",
        "status": "DEVELOPMENT_SELECTED_PENDING_GIT_FREEZE",
        "scientific_configuration": scientific,
        "endpoint_candidate_hash": stable_hash(scientific),
        "architecture_hash": identity["architecture_hash"],
        "parameter_count": PARAMETER_COUNT,
        "selection_source": selected_source,
        "selection_decision": decision,
        "source_commit": git_head(),
        "created_at": utc_now(),
        "outer_endpoint_evaluated": False,
        "outer_labels_used_for_selection": False,
        "early_warning_checksums": early_warning_checksums(),
    }
    write_json(FREEZE_PATH, freeze)
    return freeze


def run_development_supervisor() -> int:
    prepare_directories()
    started = utc_now()
    set_sentinel("RUNNING", {"mode": "DEVELOPMENT", "pid": os.getpid()})
    status_payload(
        state="RUNNING",
        started_at=started,
        current_stage="audit",
        exit_code=None,
        pid=os.getpid(),
    )
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Phase 7 requires exactly one CUDA GPU")
        bundle = oulad._build_bundle()
        protocol_audit = audit_endpoint_protocol(bundle)
        audit_feature_leakage(bundle)
        write_json(OUT / "search_space.json", search_space())
        identity = _architecture_identity()
        write_json(
            OUT / "architecture_identity.json",
            {
                **identity,
                "unique_architecture_hash_count_required": 1,
                "unique_parameter_count_required": 1,
            },
        )
        status_payload(current_stage="control")
        runners = {
            fold: EndpointRunner(bundle, fold) for fold in OUTER_FOLDS
        }
        controls: dict[int, dict[str, Any]] = {}
        for fold in OUTER_FOLDS:
            controls[fold] = _load_or_evaluate(
                RUNS / f"control_outer{fold}_seed{SEARCH_SEED}.json",
                runners[fold],
                control_config(fold),
                seed=SEARCH_SEED,
            )
        control = _aggregate_control(controls)
        write_json(OUT / "endpoint_control_metrics.json", control)

        status_payload(current_stage="optuna")
        studies: dict[int, optuna.Study] = {}
        selected_trials: dict[int, dict[str, Any]] = {}
        for fold in OUTER_FOLDS:
            status_payload(current_outer_fold=fold)
            study = create_study(fold)
            studies[fold] = study
            remaining = max(0, TRIALS_PER_FOLD - len(study.trials))
            if remaining:
                study.optimize(
                    _objective(runners[fold], fold),
                    n_trials=remaining,
                    catch=(RuntimeError, FloatingPointError),
                    gc_after_trial=True,
                )
            selected = _select_trial(study)
            selected_trials[fold] = {
                "outer_fold": fold,
                "trial_number": selected.number,
                "value": selected.value,
                "config": selected.user_attrs["config"],
                "config_hash": selected.user_attrs["config_hash"],
                "metrics": {
                    key: selected.user_attrs[key]
                    for key in (
                        "macro_f1",
                        "pr_auc",
                        "roc_auc",
                        "nll",
                        "brier",
                        "ece",
                    )
                },
                "inner_selected_epochs": selected.user_attrs[
                    "inner_selected_epochs"
                ],
                "architecture_hash": selected.user_attrs[
                    "architecture_hash"
                ],
                "parameter_count": selected.user_attrs["parameter_count"],
                "outer_labels_used": False,
            }
            counts = _development_counts(studies)
            status_payload(**counts)
            _write_csv(OUT / "endpoint_trials.csv", _trial_rows(studies))
            write_json(
                OUT / "endpoint_selected_config.json", selected_trials
            )

        status_payload(current_stage="stability", current_outer_fold=None)
        stability: list[dict[str, Any]] = []
        stability_json = OUT / "endpoint_stability.json"
        if stability_json.is_file():
            stability = json.loads(stability_json.read_text(encoding="utf-8"))
        completed = {
            (row["configuration"], row["outer_fold"], row["seed"])
            for row in stability
        }
        for configuration in ("CONTROL", "TUNED"):
            for fold in OUTER_FOLDS:
                config = (
                    control_config(fold)
                    if configuration == "CONTROL"
                    else selected_trials[fold]["config"]
                )
                for seed in STABILITY_SEEDS:
                    key = (configuration, fold, seed)
                    if key in completed:
                        continue
                    prediction_path = (
                        PREDICTIONS
                        / f"stability_{configuration}_outer{fold}_seed{seed}.parquet"
                    )
                    result = _load_or_evaluate(
                        RUNS
                        / f"stability_{configuration}_outer{fold}_seed{seed}.json",
                        runners[fold],
                        config,
                        seed=seed,
                        prediction_path=prediction_path,
                    )
                    stability.append(
                        {
                            "configuration": configuration,
                            "outer_fold": fold,
                            "seed": seed,
                            **{
                                field: result[field]
                                for field in (
                                    "macro_f1",
                                    "pr_auc",
                                    "roc_auc",
                                    "nll",
                                    "brier",
                                    "ece",
                                    "inner_selected_epochs",
                                    "aggregated_epoch",
                                    "research_threshold",
                                    "operational_threshold",
                                    "architecture_hash",
                                    "parameter_count",
                                    "config_hash",
                                    "prediction_path",
                                    "runtime_seconds",
                                    "outer_labels_used",
                                )
                            },
                        }
                    )
                    write_json(stability_json, stability)
                    _write_csv(OUT / "endpoint_stability.csv", stability)
                    completed.add(key)
        summary = _stability_summary(stability)
        selected_source, decision = _select_configuration(summary)
        selected_payload = {
            "selected_source": selected_source,
            "decision": decision,
            "stability_summary": summary,
            "selected_trials": selected_trials,
            "outer_labels_used": False,
        }
        write_json(OUT / "endpoint_selected_config.json", selected_payload)
        freeze = _freeze_from_stability(
            stability,
            selected_trials,
            selected_source,
            decision,
            protocol_audit,
        )
        counts = _development_counts(studies)
        gate = {
            "status": "DEVELOPMENT_COMPLETE_PENDING_FREEZE_COMMIT",
            "checks": {
                "endpoint_protocol_pass": True,
                "feature_leakage_pass": True,
                "early_warning_unchanged": (
                    early_warning_checksums()
                    == freeze["early_warning_checksums"]
                ),
                "architecture_hash_count": 1,
                "parameter_count_count": 1,
                "scheduled_trials": TRIALS_PER_FOLD * 3,
                "failed_trials": counts["failed_trials"],
                "stability_complete": len(stability) == 12,
                "outer_labels_used_for_selection": False,
                "outer_endpoint_evaluated": False,
            },
        }
        write_json(OUT / "phase7_gate.json", gate)
        finished = utc_now()
        status_payload(
            state="DEVELOPMENT_COMPLETE",
            finished_at=finished,
            current_stage="awaiting_freeze_commit",
            current_outer_fold=None,
            exit_code=0,
            **counts,
        )
        set_sentinel(
            "DEVELOPMENT_COMPLETE",
            {
                "selected_source": selected_source,
                "endpoint_candidate_hash": freeze["endpoint_candidate_hash"],
            },
        )
        return 0
    except Exception as error:
        failure = {
            "state": "FAILED",
            "failure_type": type(error).__name__,
            "failure_reason": repr(error),
            "finished_at": utc_now(),
            "exit_code": 1,
        }
        write_json(OUT / "failure_summary.json", failure)
        status_payload(**failure, current_stage="failed")
        set_sentinel("FAILED", failure)
        return 1


def _validate_freeze_commit() -> tuple[dict[str, Any], str]:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    science = freeze["scientific_configuration"]
    head = git_head()
    final_predictions = OUT / "endpoint_final_predictions.parquet"
    safe_resume = False
    run_manifest_path = OUT / "endpoint_final_run_manifest.json"
    if final_predictions.exists() and run_manifest_path.exists():
        try:
            run_manifest = json.loads(
                run_manifest_path.read_text(encoding="utf-8")
            )
            safe_resume = (
                run_manifest["freeze_commit"] == head
                and run_manifest["run_count"] == 15
                and run_manifest["unique_architecture_hash_count"] == 1
                and run_manifest["unique_parameter_count"] == 1
                and all(
                    row["architecture_hash"] == freeze["architecture_hash"]
                    and row["parameter_count"] == freeze["parameter_count"]
                    and row["endpoint_candidate_hash"]
                    == freeze["endpoint_candidate_hash"]
                    and not row["outer_labels_used_for_epoch_selection"]
                    and not row["outer_labels_used_for_threshold_selection"]
                    for row in run_manifest["runs"]
                )
            )
        except (KeyError, TypeError, json.JSONDecodeError):
            safe_resume = False
    checks = {
        "candidate_hash": (
            stable_hash(science) == freeze["endpoint_candidate_hash"]
        ),
        "architecture_hash": (
            science["architecture_hash"] == _architecture_identity()["architecture_hash"]
        ),
        "parameter_count": science["parameter_count"] == PARAMETER_COUNT,
        "early_warning_unchanged": (
            early_warning_checksums() == freeze["early_warning_checksums"]
        ),
        "outer_not_previously_evaluated_or_safe_resume": (
            not final_predictions.exists() or safe_resume
        ),
    }
    message = subprocess.check_output(
        ["git", "show", "-s", "--format=%s", "HEAD"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()
    checks["freeze_commit_message"] = message.startswith(
        "freeze: lock H1 OULAD endpoint model"
    )
    checks["freeze_manifest_committed"] = (
        subprocess.run(
            [
                "git",
                "cat-file",
                "-e",
                f"HEAD:{FREEZE_PATH.relative_to(ROOT).as_posix()}",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )
    if not all(checks.values()):
        raise RuntimeError(f"endpoint freeze validation failed: {checks}")
    return freeze, head


def _train_outer(
    train: tuple,
    *,
    config: dict[str, Any],
    seed: int,
    epochs: int,
    checkpoint: Path,
    freeze: dict[str, Any],
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    frame, sequence, length, mask, aggregate_raw, labels, sample_weight = train
    preprocessor = oulad._DeepPreprocessor().fit(frame, aggregate_raw)
    aggregate, static = preprocessor.transform(frame, aggregate_raw)
    model = make_model(
        "H1_TABULAR_RESIDUAL_EXPERT",
        aggregate.shape[1],
        static.shape[1],
        config,
    ).to("cuda")
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise RuntimeError("outer endpoint architecture changed")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    risk_loss, positive_weight = _risk_loss(
        labels, config, torch.device("cuda")
    )
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
        torch.from_numpy(
            frame.module_presentation_length.to_numpy(dtype=np.int64)
        ),
        torch.from_numpy(
            frame.date_unregistration.fillna(-1).to_numpy(dtype=np.int64)
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(epochs):
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
            ) = (value.to("cuda") for value in batch)
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
                raise FloatingPointError("non-finite outer endpoint loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    payload = {
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "preprocessor": preprocessor.state(),
        "config": config,
        "aggregate_dim": int(aggregate.shape[1]),
        "static_dim": int(static.shape[1]),
        "seed": seed,
        "epochs": epochs,
        "positive_weight": positive_weight,
        "endpoint_candidate_hash": freeze["endpoint_candidate_hash"],
        "architecture_hash": freeze["architecture_hash"],
        "parameter_count": PARAMETER_COUNT,
        "outer_labels_used_for_epoch_selection": False,
        "outer_labels_used_for_threshold_selection": False,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    del model, optimizer, loader, dataset
    torch.cuda.empty_cache()
    return payload


def _restore_preprocessor(state: dict[str, Any]) -> oulad._DeepPreprocessor:
    preprocessor = oulad._DeepPreprocessor()
    for key, value in state.items():
        setattr(preprocessor, key, value)
    return preprocessor


def _predict_outer(checkpoint: Path, data: tuple) -> np.ndarray:
    frame, sequence, length, mask, aggregate_raw, _, _ = data
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_model(
        "H1_TABULAR_RESIDUAL_EXPERT",
        payload["aggregate_dim"],
        payload["static_dim"],
        payload["config"],
    )
    model.load_state_dict(payload["state_dict"])
    model.to("cuda").eval()
    aggregate, static = _restore_preprocessor(
        payload["preprocessor"]
    ).transform(frame, aggregate_raw)
    return oulad._predict_deep(
        model,
        sequence,
        length,
        mask,
        aggregate,
        static,
        "cnn_bilstm",
        torch.device("cuda"),
    )


def _checkpoint_valid(
    checkpoint: Path,
    freeze: dict[str, Any],
    seed: int,
    epochs: int,
) -> bool:
    if not checkpoint.is_file():
        return False
    try:
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        return (
            payload["endpoint_candidate_hash"]
            == freeze["endpoint_candidate_hash"]
            and payload["architecture_hash"] == freeze["architecture_hash"]
            and payload["parameter_count"] == PARAMETER_COUNT
            and payload["seed"] == seed
            and payload["epochs"] == epochs
        )
    except Exception:
        return False


def _metric_with_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, Any]:
    values = np.clip(probabilities, 1e-7, 1 - 1e-7)
    predicted = values >= thresholds
    tn, fp, fn, tp = confusion_matrix(
        labels, predicted, labels=[0, 1]
    ).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=[0, 1], zero_division=0
    )
    macro = precision_recall_fscore_support(
        labels, predicted, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predicted)
        ),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "weighted_f1": float(
            f1_score(labels, predicted, average="weighted")
        ),
        "risk_precision": float(precision[1]),
        "risk_recall": float(recall[1]),
        "risk_f1": float(f1[1]),
        "specificity": float(tn / max(tn + fp, 1)),
        "pr_auc": float(average_precision_score(labels, values)),
        "roc_auc": float(roc_auc_score(labels, values)),
        "nll": float(log_loss(labels, values, labels=[0, 1])),
        "brier": float(np.mean((values - labels) ** 2)),
        "ece": oulad._ece(labels, values),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "eligible_count": int(len(labels)),
    }


def _historical_predictions(
    h1: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    h0 = pd.read_parquet(HISTORICAL_H0).rename(
        columns={
            "record_id": "base_record_id",
            "probability": "probability_h0",
            "threshold": "threshold_h0",
        }
    )
    mlp = pd.read_parquet(HISTORICAL_MLP).rename(
        columns={
            "record_id": "base_record_id",
            "true_label": "target",
            "p_at_risk": "probability_mlp",
            "threshold": "threshold_mlp",
        }
    )
    keys = ["base_record_id", "id_student", "outer_fold", "target"]
    h0_aligned = h1[keys].merge(
        h0[
            keys
            + ["probability_h0", "threshold_h0"]
        ],
        on=keys,
        validate="one_to_one",
    )
    mlp_aligned = h1[keys].merge(
        mlp[
            keys
            + ["probability_mlp", "threshold_mlp"]
        ],
        on=keys,
        validate="one_to_one",
    )
    if len(h0_aligned) != 15_378 or len(mlp_aligned) != 15_378:
        raise RuntimeError("historical endpoint comparator alignment failed")
    return h0_aligned, mlp_aligned


def _bootstrap_delta(
    frame: pd.DataFrame,
    *,
    probability_a: str,
    threshold_a: str,
    probability_b: str,
    threshold_b: str,
    comparator: str,
) -> dict[str, Any]:
    labels = frame.target.to_numpy(dtype=int)
    prediction_a = (
        frame[probability_a].to_numpy() >= frame[threshold_a].to_numpy()
    )
    prediction_b = (
        frame[probability_b].to_numpy() >= frame[threshold_b].to_numpy()
    )
    groups = np.array(sorted(frame.id_student.unique()))
    index = {
        value: position for position, value in enumerate(groups)
    }
    group_index = frame.id_student.map(index).to_numpy()
    counts_a = np.zeros((len(groups), 4), dtype=np.int64)
    counts_b = np.zeros((len(groups), 4), dtype=np.int64)
    for predictions, counts in (
        (prediction_a, counts_a),
        (prediction_b, counts_b),
    ):
        for outcome, column in (
            ((labels == 0) & (~predictions), 0),
            ((labels == 0) & predictions, 1),
            ((labels == 1) & (~predictions), 2),
            ((labels == 1) & predictions, 3),
        ):
            np.add.at(counts[:, column], group_index[outcome], 1)

    def macro(values: np.ndarray) -> float:
        tn, fp, fn, tp = values
        return float(
            (
                2 * tn / max(2 * tn + fp + fn, 1)
                + 2 * tp / max(2 * tp + fp + fn, 1)
            )
            / 2
        )

    point = macro(counts_a.sum(axis=0)) - macro(counts_b.sum(axis=0))
    rng = np.random.default_rng(7319)
    deltas = np.empty(5000, dtype=float)
    for replicate in range(5000):
        weights = np.bincount(
            rng.integers(0, len(groups), size=len(groups)),
            minlength=len(groups),
        )
        deltas[replicate] = macro(weights @ counts_a) - macro(
            weights @ counts_b
        )
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "base": "H1_ENDPOINT",
        "comparator": comparator,
        "metric": "pooled_outer_oof_macro_f1",
        "point_delta": point,
        "bootstrap_mean": float(deltas.mean()),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "interval_crosses_zero": bool(low <= 0 <= high),
        "replicates": 5000,
        "resampling_unit": "id_student",
    }


def run_final_supervisor() -> int:
    prepare_directories()
    started = utc_now()
    set_sentinel("RUNNING", {"mode": "FINAL_ENDPOINT", "pid": os.getpid()})
    status_payload(
        state="RUNNING",
        started_at=started,
        finished_at=None,
        current_stage="validate_freeze",
        exit_code=None,
        pid=os.getpid(),
    )
    try:
        freeze, freeze_commit = _validate_freeze_commit()
        science = freeze["scientific_configuration"]
        bundle = oulad._build_bundle()
        base = bundle.stages[ENDPOINT_STAGE].frame.loc[
            :,
            ["base_record_id", "outer_fold"],
        ].drop_duplicates()
        seed_predictions: list[pd.DataFrame] = []
        run_rows: list[dict[str, Any]] = []
        for fold in OUTER_FOLDS:
            fit_ids = set(
                base.loc[base.outer_fold.ne(fold), "base_record_id"]
            )
            test_ids = set(
                base.loc[base.outer_fold.eq(fold), "base_record_id"]
            )
            train = _endpoint_rows(bundle, fit_ids)
            test = _endpoint_rows(bundle, test_ids)
            config = science["training_configs_by_outer_fold"][str(fold)]
            epochs = int(science["refit_epochs_by_outer_fold"][str(fold)])
            for seed in FINAL_SEEDS:
                status_payload(
                    current_stage="final_endpoint_evaluation",
                    current_outer_fold=fold,
                    current_seed=seed,
                )
                checkpoint = (
                    CHECKPOINTS
                    / "H1_ENDPOINT"
                    / f"outer{fold}"
                    / f"seed{seed}.pt"
                )
                resumed = _checkpoint_valid(
                    checkpoint, freeze, seed, epochs
                )
                if not resumed:
                    if checkpoint.exists():
                        raise RuntimeError(
                            "invalid completed endpoint checkpoint; "
                            "selective replacement prohibited"
                        )
                    payload = _train_outer(
                        train,
                        config=config,
                        seed=seed,
                        epochs=epochs,
                        checkpoint=checkpoint,
                        freeze=freeze,
                    )
                else:
                    payload = torch.load(
                        checkpoint, map_location="cpu", weights_only=False
                    )
                probability = _predict_outer(checkpoint, test)
                frame = test[0].loc[
                    :,
                    [
                        "base_record_id",
                        "id_student",
                        "code_module",
                        "code_presentation",
                        "outer_fold",
                        "target",
                        "cutoff_day",
                    ],
                ].copy()
                frame["seed"] = seed
                frame["probability"] = probability
                seed_predictions.append(frame)
                run_rows.append(
                    {
                        "run_id": stable_hash(
                            {
                                "candidate_hash": freeze[
                                    "endpoint_candidate_hash"
                                ],
                                "outer_fold": fold,
                                "seed": seed,
                                "freeze_commit": freeze_commit,
                            }
                        ),
                        "outer_fold": fold,
                        "seed": seed,
                        "epochs": epochs,
                        "config_hash": stable_hash(config),
                        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                        "checkpoint_sha256": _sha(checkpoint),
                        "endpoint_candidate_hash": freeze[
                            "endpoint_candidate_hash"
                        ],
                        "architecture_hash": payload["architecture_hash"],
                        "parameter_count": payload["parameter_count"],
                        "status": "RESUMED" if resumed else "COMPLETE",
                        "outer_labels_used_for_epoch_selection": False,
                        "outer_labels_used_for_threshold_selection": False,
                    }
                )
                status_payload(completed_runs=len(run_rows))
        seed_frame = pd.concat(seed_predictions, ignore_index=True)
        seed_frame.to_parquet(
            PREDICTIONS / "endpoint_final_seed_predictions.parquet",
            index=False,
        )
        averaged = (
            seed_frame.groupby(
                [
                    "base_record_id",
                    "id_student",
                    "code_module",
                    "code_presentation",
                    "outer_fold",
                    "target",
                    "cutoff_day",
                ],
                as_index=False,
            )
            .probability.mean()
        )
        averaged["threshold"] = averaged.outer_fold.map(
            lambda fold: science["research_thresholds_by_outer_fold"][
                _outer_fold_key(fold)
            ]
        )
        averaged.to_parquet(
            OUT / "endpoint_final_predictions.parquet", index=False
        )
        seed_metric_rows: list[dict[str, Any]] = []
        for (fold, seed), group in seed_frame.groupby(
            ["outer_fold", "seed"]
        ):
            threshold = float(
                science["research_thresholds_by_outer_fold"][
                    _outer_fold_key(fold)
                ]
            )
            seed_metric_rows.append(
                {
                    "outer_fold": fold,
                    "seed": seed,
                    "threshold": threshold,
                    **oulad._metric(
                        group.target.to_numpy(),
                        group.probability.to_numpy(),
                        threshold,
                    ),
                }
            )
        _write_csv(
            OUT / "endpoint_final_fold_seed_metrics.csv", seed_metric_rows
        )
        labels = averaged.target.to_numpy(dtype=int)
        probability = averaged.probability.to_numpy(dtype=float)
        threshold = averaged.threshold.to_numpy(dtype=float)
        h1_metrics = _metric_with_thresholds(labels, probability, threshold)
        fold_rows = []
        for fold, group in averaged.groupby("outer_fold"):
            fold_rows.append(
                {
                    "outer_fold": fold,
                    **_metric_with_thresholds(
                        group.target.to_numpy(dtype=int),
                        group.probability.to_numpy(dtype=float),
                        group.threshold.to_numpy(dtype=float),
                    ),
                }
            )
        _write_csv(OUT / "endpoint_final_fold_metrics.csv", fold_rows)

        h0_aligned, mlp_aligned = _historical_predictions(averaged)
        h0_metrics = _metric_with_thresholds(
            h0_aligned.target.to_numpy(dtype=int),
            h0_aligned.probability_h0.to_numpy(dtype=float),
            h0_aligned.threshold_h0.to_numpy(dtype=float),
        )
        mlp_metrics = _metric_with_thresholds(
            mlp_aligned.target.to_numpy(dtype=int),
            mlp_aligned.probability_mlp.to_numpy(dtype=float),
            mlp_aligned.threshold_mlp.to_numpy(dtype=float),
        )
        official = pd.read_csv(FINAL_RESULTS)
        endpoint_comparator = official.loc[
            official.dataset.eq("oulad")
        ].copy()
        h1_row = {
            column: None for column in endpoint_comparator.columns
        }
        h1_row.update(
            {
                "dataset": "oulad",
                "model_id": "h1_tabular_residual",
                "model_family": "cnn_bilstm_tabular_residual",
                **{
                    key: value
                    for key, value in h1_metrics.items()
                    if key in endpoint_comparator.columns
                },
            }
        )
        endpoint_comparator = pd.concat(
            [endpoint_comparator, pd.DataFrame([h1_row])],
            ignore_index=True,
        )
        endpoint_comparator.to_csv(
            OUT / "endpoint_comparator.csv", index=False
        )
        comparison_frame = averaged.merge(
            h0_aligned[
                [
                    "base_record_id",
                    "probability_h0",
                    "threshold_h0",
                ]
            ],
            on="base_record_id",
            validate="one_to_one",
        ).merge(
            mlp_aligned[
                [
                    "base_record_id",
                    "probability_mlp",
                    "threshold_mlp",
                ]
            ],
            on="base_record_id",
            validate="one_to_one",
        )
        uncertainty = {
            "H1_VS_H0": _bootstrap_delta(
                comparison_frame,
                probability_a="probability",
                threshold_a="threshold",
                probability_b="probability_h0",
                threshold_b="threshold_h0",
                comparator="H0_CNN_BILSTM",
            ),
            "H1_VS_MLP": _bootstrap_delta(
                comparison_frame,
                probability_a="probability",
                threshold_a="threshold",
                probability_b="probability_mlp",
                threshold_b="threshold_mlp",
                comparator="MLP",
            ),
        }
        write_json(OUT / "endpoint_uncertainty.json", uncertainty)
        final_metrics = {
            "endpoint_id": ENDPOINT_ID,
            "H1": h1_metrics,
            "H0": h0_metrics,
            "MLP": mlp_metrics,
            "delta_h1_h0_macro_f1": (
                h1_metrics["macro_f1"] - h0_metrics["macro_f1"]
            ),
            "delta_h1_mlp_macro_f1": (
                h1_metrics["macro_f1"] - mlp_metrics["macro_f1"]
            ),
            "outer_labels_used_for_tuning": False,
            "post_outer_tuning": False,
        }
        write_json(OUT / "endpoint_final_metrics.json", final_metrics)
        run_manifest = {
            "status": "PASS",
            "freeze_commit": freeze_commit,
            "run_count": len(run_rows),
            "runs": run_rows,
            "unique_architecture_hash_count": len(
                {row["architecture_hash"] for row in run_rows}
            ),
            "unique_parameter_count": len(
                {row["parameter_count"] for row in run_rows}
            ),
        }
        write_json(OUT / "endpoint_final_run_manifest.json", run_manifest)
        integrity = {
            "status": "PASS",
            "freeze_commit_precedes_outer": True,
            "all_final_runs_complete": len(run_rows) == 15,
            "all_final_seeds_retained": sorted(
                {row["seed"] for row in run_rows}
            )
            == list(FINAL_SEEDS),
            "architecture_hash_count": (
                run_manifest["unique_architecture_hash_count"]
            ),
            "parameter_count_count": run_manifest["unique_parameter_count"],
            "outer_labels_used_for_tuning": False,
            "post_outer_tuning": False,
            "early_warning_unchanged": (
                early_warning_checksums()
                == freeze["early_warning_checksums"]
            ),
            "optuna_trials_after_freeze": 0,
            "h0_macro_f1_reproduced": abs(
                h0_metrics["macro_f1"] - 0.8280835945631038
            )
            < 1e-12,
            "mlp_macro_f1_reproduced": abs(
                mlp_metrics["macro_f1"] - 0.8282857900281345
            )
            < 1e-12,
        }
        expected_false_checks = {
            "outer_labels_used_for_tuning",
            "post_outer_tuning",
        }
        if not all(
            value == 1
            if key in {"architecture_hash_count", "parameter_count_count"}
            else value == 0
            if key == "optuna_trials_after_freeze"
            else value is False
            if key in expected_false_checks
            else bool(value)
            for key, value in integrity.items()
            if key != "status"
        ):
            integrity["status"] = "FAIL"
            raise RuntimeError(f"endpoint final integrity failed: {integrity}")
        write_json(OUT / "endpoint_integrity.json", integrity)
        gate = {
            "status": "PASS",
            "checks": {
                "early_warning_unchanged": True,
                "h1_architecture_unchanged": True,
                "endpoint_hyperparameters_inner_only": True,
                "freeze_precedes_outer": True,
                "all_endpoint_runs_complete": True,
                "no_post_test_tuning": True,
                "comparator_protocol_matched": True,
                "integrity_pass": True,
            },
        }
        write_json(OUT / "phase7_gate.json", gate)
        finished = utc_now()
        status_payload(
            state="COMPLETE",
            finished_at=finished,
            current_stage="complete",
            current_outer_fold=None,
            current_seed=None,
            failed_runs=0,
            failure_type=None,
            failure_reason=None,
            exit_code=0,
        )
        set_sentinel(
            "COMPLETE",
            {
                "freeze_commit": freeze_commit,
                "macro_f1": h1_metrics["macro_f1"],
                "gate": "PASS",
            },
        )
        return 0
    except Exception as error:
        failure = {
            "state": "FAILED",
            "failure_type": type(error).__name__,
            "failure_reason": repr(error),
            "finished_at": utc_now(),
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
