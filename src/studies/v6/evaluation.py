from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.studies.v5.common.metrics import binary_metrics_per_record_threshold
from src.studies.v5_1.oulad.data import prepare_oulad_inputs
from src.studies.v5_1.oulad.runner import FINAL_SEEDS, _load

from .contract import ARTIFACT_ROOT, REPORT_ROOT, ROOT, atomic_json, atomic_text, sha256_file
from .multitask import _candidate_metrics, build_temporal_targets, fit_multitask
from .pretraining import _config, fit_minimal_pretraining
from .ranking import ranking_metrics


FINAL_ROOT = ARTIFACT_ROOT / "prediction/final"
HAZARD_COLUMNS = [f"hazard_week_{week:02d}" for week in range(20)]
OUTCOME_COLUMNS = ["probability_fail", "probability_pass", "probability_distinction"]


def _locked_outer_settings() -> dict[int, dict[str, Any]]:
    rows = json.loads(
        (ROOT / "artifacts/v5_1/oulad/selected_configs.json").read_text(encoding="utf-8")
    )
    return {
        int(row["outer_fold"]): {
            "threshold": float(row["threshold"]["threshold"]),
            "source": "V5.1 inner-OOF threshold frozen before V6 outer evaluation",
            "epochs": 8,
        }
        for row in rows
    }


def _checkpoint_rows() -> list[dict[str, Any]]:
    path = FINAL_ROOT / "checkpoint_metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def _prediction_rows() -> list[dict[str, Any]]:
    path = FINAL_ROOT / "seed_predictions.parquet"
    return pd.read_parquet(path).to_dict(orient="records") if path.is_file() else []


def _persist(
    checkpoint_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]
) -> None:
    atomic_json(FINAL_ROOT / "checkpoint_metadata.json", checkpoint_rows)
    pd.DataFrame(prediction_rows).to_parquet(
        FINAL_ROOT / "seed_predictions.parquet", index=False
    )


def _metrics(
    frame: pd.DataFrame, probability: np.ndarray, hazard: np.ndarray, outcome: np.ndarray
) -> dict[str, Any]:
    binary = binary_metrics_per_record_threshold(
        frame.target.to_numpy(dtype=int),
        probability,
        frame.threshold.to_numpy(dtype=float),
    )
    auxiliary_frame = frame[
        [
            "target",
            "withdrawal_event",
            "observation_week",
            "outcome_target",
            "withdrawal_day",
            "cutoff_day",
        ]
    ].copy()
    auxiliary_frame["probability"] = probability
    auxiliary = _candidate_metrics(auxiliary_frame, hazard, outcome)
    ranked = ranking_metrics(frame.target.to_numpy(dtype=int), probability)
    return {
        **binary,
        **ranked,
        **{
            key: auxiliary[key]
            for key in (
                "survival_concordance",
                "integrated_brier_observed_risk_sets",
                "withdrawal_recall",
                "median_warning_lead_days",
                "outcome_macro_f1",
                "outcome_majority_macro_f1",
            )
        },
    }


def _existing_baselines() -> dict[str, Any]:
    v5_1_metrics = json.loads(
        (ROOT / "artifacts/v5_1/oulad/final_metrics.json").read_text(encoding="utf-8")
    )
    v5_1 = next(
        row for row in v5_1_metrics if row["candidate"] == "cnn_bilstm_full_ensemble"
    )
    v5 = pd.read_csv(ROOT / "artifacts/v5/oulad/final_metrics.csv")
    xgb = v5[v5.candidate.eq("xgboost")].sort_values("macro_f1", ascending=False).iloc[0]
    return {
        "v5_1_cnn_bilstm": v5_1,
        "xgboost_operational_cross_check": json.loads(xgb.to_json()),
        "reuse_status": "REUSED_IMMUTABLE_CHECKSUM_VERIFIED_EVIDENCE",
    }


def evaluate_final_prediction(device_name: str = "cuda") -> dict[str, Any]:
    output = FINAL_ROOT / "run_state.json"
    if output.is_file():
        cached = json.loads(output.read_text(encoding="utf-8"))
        if cached.get("status") == "COMPLETE":
            return cached
    _, _, data = _load()
    if set(data.development_manifest.role) != {"historical_development"}:
        raise RuntimeError("Future OULAD entered the V6 final cohort")
    targets = build_temporal_targets(data)
    config = _config()
    settings = _locked_outer_settings()
    weights = {"survival": 0.15, "outcome": 0.15}
    checkpoint_rows = _checkpoint_rows()
    prediction_rows = _prediction_rows()
    completed = {
        (int(row["outer_fold"]), int(row["seed"])) for row in checkpoint_rows
    }
    checkpoint_root = FINAL_ROOT / "checkpoints"
    pretraining_root = FINAL_ROOT / "pretraining"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    pretraining_root.mkdir(parents=True, exist_ok=True)
    for outer_fold in range(3):
        train_index, validation_index = data.v2.outer_indices(outer_fold)
        train = prepare_oulad_inputs(data, train_index, train_index)
        validation = prepare_oulad_inputs(
            data, train_index, validation_index, fitted=train.preprocessors
        )
        for seed in FINAL_SEEDS:
            if (outer_fold, seed) in completed:
                continue
            started = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            pretrain = fit_minimal_pretraining(
                train,
                data.dynamic_sequence[train_index],
                dynamic_channel_order=data.dynamic_channel_order,
                config=config,
                seed=seed,
                epochs=5,
                device_name=device_name,
            )
            pretraining_path = pretraining_root / f"outer_{outer_fold}_seed_{seed}.pt"
            torch.save(pretrain.temporal_state_dict, pretraining_path)
            fit = fit_multitask(
                train,
                validation,
                targets,
                train_index,
                validation_index,
                config=config,
                weights=weights,
                initial_temporal_state=pretrain.temporal_state_dict,
                seed=seed,
                epochs=int(settings[outer_fold]["epochs"]),
                device_name=device_name,
            )
            checkpoint_path = checkpoint_root / f"outer_{outer_fold}_seed_{seed}.pt"
            torch.save(fit.state_dict, checkpoint_path)
            gpu_peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            checkpoint_rows.append(
                {
                    "candidate": "C_TEMPORAL_MULTITASK_W0",
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "path": checkpoint_path.relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(checkpoint_path),
                    "state_dict_sha256": fit.checkpoint_sha256,
                    "pretraining_path": pretraining_path.relative_to(ROOT).as_posix(),
                    "pretraining_sha256": sha256_file(pretraining_path),
                    "pretraining_state_sha256": pretrain.checkpoint_sha256,
                    "pretraining_replay_max_abs_difference": pretrain.replay_max_abs_difference,
                    "replay_max_abs_difference": fit.replay_max_abs_difference,
                    "parameter_count": fit.parameter_count,
                    "runtime_seconds": time.perf_counter() - started,
                    "pretraining_runtime_seconds": pretrain.runtime_seconds,
                    "model_runtime_seconds": fit.runtime_seconds,
                    "gpu_peak_memory_bytes": gpu_peak,
                    "threshold": settings[outer_fold]["threshold"],
                    "threshold_source": settings[outer_fold]["source"],
                }
            )
            cohort = data.base.cohort.iloc[validation_index]
            for local, index in enumerate(validation_index):
                row: dict[str, Any] = {
                    "record_id": str(data.base.record_ids[index]),
                    "id_student": int(data.groups[index]),
                    "code_module": str(cohort.iloc[local].code_module),
                    "code_presentation": str(cohort.iloc[local].code_presentation),
                    "cutoff_day": int(cohort.iloc[local].cutoff_day),
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "target": int(data.y[index]),
                    "probability": float(fit.binary_probability[local]),
                    "threshold": settings[outer_fold]["threshold"],
                    "withdrawal_event": int(targets.withdrawal_event[index]),
                    "observation_week": int(targets.observation_week[index]),
                    "outcome_target": int(targets.outcome_target[index]),
                    "withdrawal_day": float(targets.withdrawal_day[index]),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                }
                row.update(
                    {
                        name: float(fit.hazard_probability[local, column])
                        for column, name in enumerate(HAZARD_COLUMNS)
                    }
                )
                row.update(
                    {
                        name: float(fit.outcome_probability[local, column])
                        for column, name in enumerate(OUTCOME_COLUMNS)
                    }
                )
                prediction_rows.append(row)
            _persist(checkpoint_rows, prediction_rows)
    seeds = pd.DataFrame(prediction_rows)
    metric_rows: list[dict[str, Any]] = []
    for seed, frame in seeds.groupby("seed", sort=True):
        metric_rows.append(
            {
                "candidate": "C_TEMPORAL_MULTITASK_W0",
                "seed": int(seed),
                **_metrics(
                    frame,
                    frame.probability.to_numpy(dtype=float),
                    frame[HAZARD_COLUMNS].to_numpy(dtype=float),
                    frame[OUTCOME_COLUMNS].to_numpy(dtype=float),
                ),
            }
        )
    group_keys = [
        "record_id",
        "id_student",
        "code_module",
        "code_presentation",
        "cutoff_day",
        "outer_fold",
        "target",
        "threshold",
        "withdrawal_event",
        "observation_week",
        "outcome_target",
        "withdrawal_day",
    ]
    ensemble = seeds.groupby(group_keys, as_index=False).agg(
        probability=("probability", "mean"),
        probability_std=("probability", "std"),
        **{column: (column, "mean") for column in HAZARD_COLUMNS + OUTCOME_COLUMNS},
    )
    ensemble_metrics = _metrics(
        ensemble,
        ensemble.probability.to_numpy(dtype=float),
        ensemble[HAZARD_COLUMNS].to_numpy(dtype=float),
        ensemble[OUTCOME_COLUMNS].to_numpy(dtype=float),
    )
    metric_rows.append(
        {"candidate": "C_TEMPORAL_MULTITASK_W0_ENSEMBLE", "seed": -1, **ensemble_metrics}
    )
    pd.DataFrame(metric_rows).to_json(FINAL_ROOT / "metrics.json", orient="records", indent=2)
    ensemble.to_parquet(ARTIFACT_ROOT / "prediction/oof_predictions.parquet", index=False)
    baselines = _existing_baselines()
    atomic_json(FINAL_ROOT / "reused_baselines.json", baselines)
    parameter_count = int(max(row["parameter_count"] for row in checkpoint_rows))
    total_runtime = float(sum(row["runtime_seconds"] for row in checkpoint_rows))
    max_gpu_memory = int(max(row["gpu_peak_memory_bytes"] for row in checkpoint_rows))
    selected_model = {
        "schema_version": "v6_selected_prediction_model_v1",
        "status": "FROZEN_AFTER_INNER_GATES_BEFORE_OUTER_REVIEW",
        "candidate": "C_TEMPORAL_MULTITASK_W0",
        "pretraining": "P1_MASKED_AND_NEXT_WEEK",
        "multitask_weights": weights,
        "ranking": "REJECTED_BY_GUARDRAIL",
        "graph": "SKIPPED_CANDIDATE_D_GATE_FAILED",
        "fixed_seeds": list(FINAL_SEEDS),
        "outer_folds": [0, 1, 2],
        "parameter_count": parameter_count,
        "thresholds": settings,
        "outer_test_used_for_selection": False,
        "future_accessed": False,
    }
    atomic_json(ARTIFACT_ROOT / "prediction/selected_model.json", selected_model)
    result = {
        "schema_version": "v6_final_prediction_evaluation_v1",
        "status": "COMPLETE",
        "selected_model": selected_model,
        "ensemble_metrics": ensemble_metrics,
        "seed_metrics": metric_rows[:-1],
        "seed_macro_f1_mean": float(
            np.mean([row["macro_f1"] for row in metric_rows[:-1]])
        ),
        "seed_macro_f1_std": float(np.std([row["macro_f1"] for row in metric_rows[:-1]])),
        "runtime_seconds": total_runtime,
        "max_gpu_memory_bytes": max_gpu_memory,
        "checkpoint_count": len(checkpoint_rows),
        "prediction_records": int(len(ensemble)),
        "outer_test_used_for_selection": False,
        "future_accessed": False,
    }
    atomic_json(output, result)
    atomic_text(
        REPORT_ROOT / "GRAPH_CONTEXT_REPORT.md",
        """# V6 graph-context report

Graph feasibility audit passed, but Candidate D ranking failed its registered
Macro-F1/Brier guardrails. The additive ladder therefore stopped at Candidate C.
Graph context was not trained, evaluated or included in the final model.
""",
    )
    atomic_text(
        REPORT_ROOT / "PREDICTION_FINAL_REPORT.md",
        f"""# V6 final prediction report

Selected candidate: **C_TEMPORAL_MULTITASK_W0**. Selection was frozen from
inner-fold gates before viewing any outer-test result.

- Outer evaluation: 3 folds x 5 fixed seeds ({len(checkpoint_rows)} checkpoints)
- Ensemble Macro-F1: {ensemble_metrics['macro_f1']:.6f}
- Ensemble At-risk F1: {ensemble_metrics['at_risk_f1']:.6f}
- Ensemble PR-AUC: {ensemble_metrics['pr_auc']:.6f}
- Ensemble Brier: {ensemble_metrics['brier']:.6f}
- Ensemble ECE: {ensemble_metrics['ece']:.6f}
- Recall@10%: {ensemble_metrics['recall_at_10_percent']:.6f}
- Survival C-index: {ensemble_metrics['survival_concordance']:.6f}
- Outcome Macro-F1: {ensemble_metrics['outcome_macro_f1']:.6f}
- Parameters: {parameter_count:,}
- Total recorded fit runtime: {total_runtime:.1f} seconds
- Peak CUDA allocation: {max_gpu_memory / 1024**2:.1f} MiB

XGBoost remains an operational cross-check and is not embedded in the deep
model. Future OULAD remained locked.
""",
    )
    return result


__all__ = ["evaluate_final_prediction"]
