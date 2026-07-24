from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from src.studies.oulad_v4.data import (
    build_v4_inner_manifest,
    load_v4_data,
    manifest_indices,
)
from src.studies.v5.common.metrics import binary_metrics, binary_metrics_per_record_threshold
from src.studies.v5_1.oulad.data import prepare_oulad_inputs
from src.studies.v5_1.oulad.models import OULADHybridV51
from src.studies.v5_1.oulad.training import choose_threshold
from src.studies.v5_1.common.protocol import ROOT, sha256_file

from .oulad_architecture import (
    CandidateSpec,
    OULADArchitectureDiagnosisNet,
    candidate_specs,
    parameter_breakdown,
)
from .oulad_training import fit_diagnosis_model, transform_order


ARTIFACT_ROOT = ROOT / "artifacts/v6_1_oulad_architecture_diagnosis"
CONFIG_PATH = ROOT / "configs/v6_1/oulad_architecture_diagnosis.yaml"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _load() -> tuple[dict[str, Any], dict[str, Any], Any]:
    protocol = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    v4_protocol = yaml.safe_load(
        (ROOT / "configs/oulad_v4_protocol.yaml").read_text(encoding="utf-8")
    )
    data = load_v4_data(ROOT / "data/processed/study_c_oulad", v4_protocol)
    if data.dynamic_sequence.shape[1:] != (20, 47):
        raise RuntimeError(f"Frozen sequence contract changed: {data.dynamic_sequence.shape}")
    if set(data.development_manifest.role) != {"historical_development"}:
        raise RuntimeError("Future OULAD role entered V6.1 diagnosis")
    return protocol, v4_protocol, data


def _current_config() -> dict[str, Any]:
    evidence = json.loads(
        (ROOT / "artifacts/v5_1/oulad/architecture_screening.json").read_text(
            encoding="utf-8"
        )
    )
    return dict(evidence["architecture"]["selected"]["config"])


def _training_config(protocol: dict[str, Any]) -> dict[str, Any]:
    current = _current_config()
    training = protocol["training"]
    return {
        **current,
        "max_epochs": int(training["max_epochs"]),
        "patience": int(training["patience"]),
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "dropout": float(training["dropout"]),
        "gradient_clip": float(training["gradient_clip"]),
    }


def _inner_splits(data: Any, outer_fold: int, v4_protocol: dict[str, Any]):
    manifest = build_v4_inner_manifest(data, outer_fold, v4_protocol)
    splits = [
        manifest_indices(data.v2, manifest, int(inner_fold))
        for inner_fold in sorted(manifest.inner_fold.unique())
    ]
    for train, valid in splits:
        train_groups = set(data.groups[train].tolist())
        valid_groups = set(data.groups[valid].tolist())
        if train_groups & valid_groups:
            raise RuntimeError("Student group leaked across an inner split")
    return splits


def _parameter_count(module: torch.nn.Module | None) -> int:
    return (
        0
        if module is None
        else int(sum(parameter.numel() for parameter in module.parameters()))
    )


def architecture_audit() -> dict[str, Any]:
    protocol, v4_protocol, data = _load()
    splits = _inner_splits(data, 0, v4_protocol)
    train_index, validation_index = splits[0]
    train = prepare_oulad_inputs(data, train_index, train_index[:16])
    config = _training_config(protocol)
    current = OULADHybridV51(
        train.sequence.shape[2],
        train.aggregate.shape[1],
        train.static.shape[1],
        config,
        "cnn_bilstm",
    )
    temporal = current.temporal
    current_groups = {
        "input_projection": _parameter_count(temporal.input_projection)
        + _parameter_count(temporal.input_norm),
        "cnn": _parameter_count(temporal.convolutions)
        + _parameter_count(temporal.residual)
        + _parameter_count(temporal.conv_norm),
        "bilstm": _parameter_count(temporal.recurrent),
        "temporal_pool_projection": _parameter_count(temporal.projection)
        + _parameter_count(current.temporal_projection),
        "aggregate_branch": _parameter_count(current.aggregate),
        "static_branch": _parameter_count(current.static),
        "fusion_head": _parameter_count(current.gates)
        + _parameter_count(current.head),
    }
    total = int(sum(parameter.numel() for parameter in current.parameters()))
    current_groups["total"] = total
    current_groups["ratios"] = {
        name: value / total for name, value in current_groups.items() if name != "total"
    }
    registry: list[dict[str, Any]] = []
    for spec in candidate_specs().values():
        model = OULADArchitectureDiagnosisNet(
            train.sequence.shape[2],
            train.aggregate.shape[1],
            train.static.shape[1],
            config,
            spec,
        )
        registry.append(
            {
                "candidate_id": spec.candidate_id,
                "phase": spec.phase,
                "temporal": spec.temporal,
                "branches": spec.branches,
                "kernels": list(spec.kernels),
                "dilations": list(spec.resolved_dilations()),
                "conv_channels": spec.conv_channels,
                "parameter_counts": parameter_breakdown(model),
            }
        )
    by_id = {row["candidate_id"]: row for row in registry}
    cnn_encoder = int(
        by_id["B2_cnn_matched_temporal"]["parameter_counts"]["input_projection"]
        + by_id["B2_cnn_matched_temporal"]["parameter_counts"]["cnn"]
    )
    bilstm_encoder = int(
        by_id["A2_bilstm_current_temporal"]["parameter_counts"]["input_projection"]
        + by_id["A2_bilstm_current_temporal"]["parameter_counts"]["bilstm"]
    )
    deviation = abs(cnn_encoder - bilstm_encoder) / max(cnn_encoder, bilstm_encoder)
    audit = {
        "schema_version": "v6_1_architecture_audit_v1",
        "status": "COMPLETE",
        "source_files": {
            "model": "src/studies/v5_1/oulad/models.py",
            "training": "src/studies/v5_1/oulad/training.py",
            "data": "src/studies/v5_1/oulad/data.py",
            "selected_config": "artifacts/v5_1/oulad/architecture_screening.json",
        },
        "input": {
            "records": int(len(data.y)),
            "temporal_dimensions": list(data.dynamic_sequence.shape),
            "sequence_channels": int(train.sequence.shape[2]),
            "aggregate_dimensions": int(train.aggregate.shape[1]),
            "static_dimensions": int(train.static.shape[1]),
            "compact_aggregate_columns": list(train.aggregate_columns),
        },
        "current_implementation": {
            "input_projection": 48,
            "kernels": [2, 3],
            "conv_channels_per_kernel": 24,
            "dilation": 2,
            "cnn_residual": "linear projection added before layer norm and GELU",
            "bilstm_hidden": 64,
            "bilstm_layers": 1,
            "bidirectional": True,
            "pooling": "padding-masked mean concatenated with padding-masked max",
            "pooling_projection": 48,
            "aggregate_branch": "49 -> 64 -> 64",
            "static_branch": "13 -> 32 -> 64",
            "fusion": "temporal + sigmoid_gate_0*aggregate + sigmoid_gate_1*static",
            "padding": "packed BiLSTM; zeroed projected/convolved/unpacked padding",
            "threshold": "pooled inner-OOF only",
            "outer_folds": 3,
            "inner_folds": 3,
            "seeds": protocol["data"]["final_seeds"],
        },
        "current_parameter_counts": current_groups,
        "capacity_hypothesis": {
            "current_cnn_params": current_groups["cnn"],
            "current_bilstm_params": current_groups["bilstm"],
            "bilstm_to_cnn_ratio": current_groups["bilstm"]
            / max(1, current_groups["cnn"]),
            "capacity_imbalance_present": current_groups["bilstm"]
            > 2 * current_groups["cnn"],
        },
        "parameter_match": {
            "cnn_encoder_params": cnn_encoder,
            "bilstm_encoder_params": bilstm_encoder,
            "deviation_fraction": deviation,
            "within_preregistered_10_percent": deviation <= 0.10,
        },
        "outer_test_accessed": False,
        "future_accessed": False,
    }
    if deviation > 0.10:
        raise RuntimeError(f"Parameter match guard failed: {deviation:.3%}")
    _atomic_json(ARTIFACT_ROOT / "architecture_audit.json", audit)
    _atomic_json(ARTIFACT_ROOT / "candidate_registry.json", registry)
    _atomic_json(
        ARTIFACT_ROOT / "parameter_counts.json",
        {
            "current": current_groups,
            "candidates": {
                row["candidate_id"]: row["parameter_counts"] for row in registry
            },
            "parameter_match": audit["parameter_match"],
        },
    )
    return audit


def _run_paths(scope: str, candidate: str, outer_fold: int, inner_fold: int, seed: int):
    stem = f"{candidate}_outer_{outer_fold}_inner_{inner_fold}_seed_{seed}"
    root = ARTIFACT_ROOT / "runs" / scope
    return (
        root / f"{stem}.json",
        root / f"{stem}.npz",
        ARTIFACT_ROOT / "checkpoints" / scope / f"{stem}.pt",
        ARTIFACT_ROOT / "logs" / "histories" / f"{scope}_{stem}.csv",
    )


def _run_fit(
    *,
    data: Any,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    config: dict[str, Any],
    spec: CandidateSpec,
    outer_fold: int,
    inner_fold: int,
    seed: int,
    scope: str,
    device: str,
    order: str = "original",
    fixed_epochs: int | None = None,
) -> dict[str, Any]:
    metadata_path, prediction_path, checkpoint_path, history_path = _run_paths(
        scope, spec.candidate_id, outer_fold, inner_fold, seed
    )
    if metadata_path.is_file() and prediction_path.is_file() and checkpoint_path.is_file():
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            cached.get("status") == "COMPLETE"
            and cached.get("order") == order
            and cached.get("config_sha256") == _sha256_json(config)
        ):
            return cached
    train = prepare_oulad_inputs(data, train_index, train_index)
    evaluation = prepare_oulad_inputs(
        data, train_index, validation_index, fitted=train.preprocessors
    )
    train = transform_order(train, order, seed)
    evaluation = transform_order(evaluation, order, seed)
    fit = fit_diagnosis_model(
        train,
        evaluation,
        config=config,
        spec=spec,
        seed=seed,
        device_name=device,
        fixed_epochs=fixed_epochs,
    )
    _atomic_torch(checkpoint_path, fit.state_dict)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = prediction_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            index=validation_index.astype(np.int64),
            target=evaluation.target.astype(np.int8),
            probability=fit.probability.astype(np.float32),
        )
    os.replace(temporary, prediction_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fit.history).to_csv(history_path, index=False)
    metadata = {
        "status": "COMPLETE",
        "candidate": spec.candidate_id,
        "scope": scope,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "seed": seed,
        "order": order,
        "records": int(len(validation_index)),
        "selected_epoch": fit.selected_epoch,
        "parameter_count": fit.parameter_count,
        "runtime_seconds": fit.runtime_seconds,
        "gpu_peak_memory_bytes": fit.gpu_peak_memory_bytes,
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "state_dict_sha256": fit.checkpoint_sha256,
        "prediction": prediction_path.relative_to(ROOT).as_posix(),
        "replay_max_abs_difference": fit.replay_max_abs_difference,
        "diagnostics": fit.diagnostics,
        "config_sha256": _sha256_json(config),
        "outer_test_accessed": False if scope != "final" else True,
        "future_accessed": False,
    }
    _atomic_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "completed": metadata["candidate"],
                "scope": scope,
                "outer": outer_fold,
                "inner": inner_fold,
                "seed": seed,
                "runtime_seconds": round(fit.runtime_seconds, 2),
            }
        ),
        flush=True,
    )
    return metadata


def _load_prediction(metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    loaded = np.load(ROOT / metadata["prediction"])
    return {name: loaded[name] for name in loaded.files}


def _recall_at_fraction(target: np.ndarray, probability: np.ndarray, fraction: float) -> float:
    count = max(1, int(np.ceil(len(target) * fraction)))
    selected = np.argsort(-probability)[:count]
    positives = max(1, int(np.asarray(target).sum()))
    return float(np.asarray(target)[selected].sum() / positives)


def _extended_metrics(
    target: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, Any]:
    result = binary_metrics(target, probability, threshold)
    result["roc_auc"] = float(roc_auc_score(target, probability))
    result["recall_at_10_percent"] = _recall_at_fraction(target, probability, 0.10)
    return result


def _summarize_inner(
    metadata_rows: list[dict[str, Any]], aliases: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, Any]] = []
    for metadata in metadata_rows:
        values = _load_prediction(metadata)
        for index, target, probability in zip(
            values["index"], values["target"], values["probability"]
        ):
            prediction_rows.append(
                {
                    "candidate": metadata["candidate"],
                    "outer_fold": metadata["outer_fold"],
                    "inner_fold": metadata["inner_fold"],
                    "seed": metadata["seed"],
                    "index": int(index),
                    "target": int(target),
                    "probability": float(probability),
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    for alias, source in aliases.items():
        copied = predictions[predictions.candidate.eq(source)].copy()
        copied["candidate"] = alias
        predictions = pd.concat([predictions, copied], ignore_index=True)
    seed_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for (candidate, seed), frame in predictions.groupby(["candidate", "seed"]):
        threshold = choose_threshold(
            frame.target.to_numpy(), frame.probability.to_numpy()
        )
        metrics = _extended_metrics(
            frame.target.to_numpy(),
            frame.probability.to_numpy(),
            float(threshold["threshold"]),
        )
        seed_rows.append(
            {
                "candidate": candidate,
                "seed": int(seed),
                **metrics,
            }
        )
        for inner_fold, fold in frame.groupby("inner_fold"):
            fold_rows.append(
                {
                    "candidate": candidate,
                    "seed": int(seed),
                    "inner_fold": int(inner_fold),
                    **_extended_metrics(
                        fold.target.to_numpy(),
                        fold.probability.to_numpy(),
                        float(threshold["threshold"]),
                    ),
                }
            )
    seed_metrics = pd.DataFrame(seed_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    aggregate_rows: list[dict[str, Any]] = []
    metric_names = [
        "macro_f1",
        "balanced_accuracy",
        "at_risk_precision",
        "at_risk_recall",
        "at_risk_f1",
        "pr_auc",
        "roc_auc",
        "brier",
        "ece",
        "recall_at_10_percent",
    ]
    specs = candidate_specs()
    for candidate, frame in seed_metrics.groupby("candidate"):
        source = aliases.get(candidate, candidate)
        aggregate: dict[str, Any] = {
            "candidate": candidate,
            "phase": "C" if candidate == "C2_cnn_d2_temporal" else specs[source].phase,
            "seeds": int(len(frame)),
        }
        for metric in metric_names:
            aggregate[f"{metric}_mean"] = float(frame[metric].mean())
            aggregate[f"{metric}_sd"] = float(frame[metric].std(ddof=0))
        candidate_metadata = [
            row for row in metadata_rows if row["candidate"] == source
        ]
        aggregate["parameter_count"] = int(
            candidate_metadata[0]["parameter_count"]
        )
        aggregate["runtime_seconds_total"] = float(
            sum(row["runtime_seconds"] for row in candidate_metadata)
        )
        aggregate["selected_epoch_mean"] = float(
            np.mean([row["selected_epoch"] for row in candidate_metadata])
        )
        aggregate_rows.append(aggregate)
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        "macro_f1_mean", ascending=False
    )
    predictions.to_parquet(ARTIFACT_ROOT / "inner_predictions.parquet", index=False)
    seed_metrics.to_csv(ARTIFACT_ROOT / "inner_results.csv", index=False)
    fold_metrics.to_csv(ARTIFACT_ROOT / "inner_fold_results.csv", index=False)
    return aggregate, seed_metrics, fold_metrics


def _phase_tables(aggregate: pd.DataFrame) -> None:
    mappings = {
        "ablation_results.csv": [
            "A0_aggregate_static_only",
            "A1_cnn_small_temporal",
            "A2_bilstm_current_temporal",
            "A3_serial_current_temporal",
            "A4_serial_current_full",
        ],
        "capacity_match_results.csv": [
            "A1_cnn_small_temporal",
            "B2_cnn_matched_temporal",
            "A2_bilstm_current_temporal",
        ],
        "dilation_results.csv": [
            "C1_cnn_d1_temporal",
            "C2_cnn_d2_temporal",
            "C3_cnn_multidilation_temporal",
        ],
        "serial_vs_skip_results.csv": [
            "A4_serial_current_full",
            "D_serial_with_cnn_skip",
        ],
        "parallel_results.csv": [
            "A4_serial_current_full",
            "D_serial_with_cnn_skip",
            "E_parallel_concat",
        ],
    }
    for filename, candidates in mappings.items():
        aggregate[aggregate.candidate.isin(candidates)].to_csv(
            ARTIFACT_ROOT / filename, index=False
        )


def _select_candidate(
    aggregate: pd.DataFrame, fold_metrics: pd.DataFrame, protocol: dict[str, Any]
) -> dict[str, Any]:
    baseline = aggregate.set_index("candidate").loc["A4_serial_current_full"]
    gates = protocol["selection"]
    candidates: list[dict[str, Any]] = []
    for candidate in ["D_serial_with_cnn_skip", "E_parallel_concat"]:
        row = aggregate.set_index("candidate").loc[candidate]
        baseline_folds = (
            fold_metrics[fold_metrics.candidate.eq("A4_serial_current_full")]
            .groupby("inner_fold")
            .macro_f1.mean()
        )
        candidate_folds = (
            fold_metrics[fold_metrics.candidate.eq(candidate)]
            .groupby("inner_fold")
            .macro_f1.mean()
        )
        positive_folds = int(
            (
                candidate_folds
                - baseline_folds
                >= float(gates["practical_tie_delta"])
            ).sum()
        )
        verdict = {
            "candidate": candidate,
            "macro_f1_delta": float(
                row.macro_f1_mean - baseline.macro_f1_mean
            ),
            "at_risk_recall_delta": float(
                row.at_risk_recall_mean - baseline.at_risk_recall_mean
            ),
            "brier_delta": float(row.brier_mean - baseline.brier_mean),
            "positive_inner_folds": positive_folds,
        }
        verdict["passes"] = bool(
            verdict["macro_f1_delta"]
            >= float(gates["minimum_mean_macro_f1_gain_vs_A4"])
            and positive_folds >= int(gates["minimum_positive_inner_folds"])
            and verdict["at_risk_recall_delta"]
            >= -float(gates["maximum_at_risk_recall_drop"])
            and verdict["brier_delta"]
            <= float(gates["maximum_brier_increase"])
        )
        candidates.append(verdict)
    passing = [row for row in candidates if row["passes"]]
    selected = (
        max(passing, key=lambda row: row["macro_f1_delta"])["candidate"]
        if passing
        else None
    )
    selected_spec = (
        None if selected is None else candidate_specs()[selected].__dict__
    )
    result = {
        "schema_version": "v6_1_selected_architecture_v1",
        "status": "FROZEN_FOR_OUTER_EVALUATION"
        if selected is not None
        else "NO_CANDIDATE_PASSED_DEVELOPMENT_GATE",
        "selected_candidate": selected,
        "selected_spec": selected_spec,
        "config": _training_config(protocol) if selected is not None else None,
        "gate_results": candidates,
        "selection_scope": "outer_training_fold_0_inner_cv_only",
        "outer_test_used_for_selection": False,
        "future_accessed": False,
    }
    result["config_sha256"] = _sha256_json(
        {"candidate": selected, "spec": selected_spec, "config": result["config"]}
    )
    (ARTIFACT_ROOT / "selected_config.yaml").write_text(
        yaml.safe_dump(result, sort_keys=False), encoding="utf-8"
    )
    return result


def run_inner(device: str = "cuda") -> dict[str, Any]:
    protocol, v4_protocol, data = _load()
    architecture_audit()
    config = _training_config(protocol)
    splits = _inner_splits(data, 0, v4_protocol)
    metadata_rows: list[dict[str, Any]] = []
    for spec in candidate_specs().values():
        for inner_fold, (train_index, validation_index) in enumerate(splits):
            for seed in protocol["data"]["screening_seeds"]:
                metadata_rows.append(
                    _run_fit(
                        data=data,
                        train_index=train_index,
                        validation_index=validation_index,
                        config=config,
                        spec=spec,
                        outer_fold=0,
                        inner_fold=inner_fold,
                        seed=int(seed),
                        scope="inner",
                        device=device,
                    )
                )
    aggregate, seed_metrics, fold_metrics = _summarize_inner(
        metadata_rows, {"C2_cnn_d2_temporal": "A1_cnn_small_temporal"}
    )
    aggregate.to_csv(ARTIFACT_ROOT / "candidate_summary.csv", index=False)
    _phase_tables(aggregate)
    selected = _select_candidate(aggregate, fold_metrics, protocol)
    state = {
        "status": "COMPLETE",
        "runs": len(metadata_rows),
        "unique_candidates_trained": len(candidate_specs()),
        "candidate_rows_reported": int(len(aggregate)),
        "selected": selected,
        "outer_test_accessed": False,
        "future_accessed": False,
    }
    _atomic_json(ARTIFACT_ROOT / "inner_run_state.json", state)
    return state


def run_order_audit(device: str = "cuda") -> dict[str, Any]:
    protocol, v4_protocol, data = _load()
    selected = yaml.safe_load(
        (ARTIFACT_ROOT / "selected_config.yaml").read_text(encoding="utf-8")
    )
    candidate = selected["selected_candidate"]
    if candidate is None:
        summary = pd.read_csv(ARTIFACT_ROOT / "candidate_summary.csv").set_index(
            "candidate"
        )
        candidate = max(
            ["D_serial_with_cnn_skip", "E_parallel_concat"],
            key=lambda name: float(summary.loc[name, "macro_f1_mean"]),
        )
        role = "best_architectural_diagnostic_no_outer_candidate"
    else:
        role = "frozen_selected_candidate"
    spec = candidate_specs()[candidate]
    config = _training_config(protocol)
    splits = _inner_splits(data, 0, v4_protocol)
    original = pd.read_parquet(ARTIFACT_ROOT / "inner_predictions.parquet")
    original = original[
        original.candidate.eq(candidate) & original.seed.eq(int(protocol["order_audit"]["seed"]))
    ].copy()
    original_threshold = float(
        choose_threshold(
            original.target.to_numpy(), original.probability.to_numpy()
        )["threshold"]
    )
    rows = [
        {
            "order": "original",
            **_extended_metrics(
                original.target.to_numpy(),
                original.probability.to_numpy(),
                original_threshold,
            ),
        }
    ]
    for order in ["reversed", "shuffled", "bag_of_weeks"]:
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for inner_fold, (train_index, validation_index) in enumerate(splits):
            metadata = _run_fit(
                data=data,
                train_index=train_index,
                validation_index=validation_index,
                config=config,
                spec=spec,
                outer_fold=0,
                inner_fold=inner_fold,
                seed=int(protocol["order_audit"]["seed"]),
                scope=f"order_{order}",
                device=device,
                order=order,
            )
            values = _load_prediction(metadata)
            predictions.append(values["probability"])
            targets.append(values["target"])
        rows.append(
            {
                "order": order,
                **_extended_metrics(
                    np.concatenate(targets),
                    np.concatenate(predictions),
                    original_threshold,
                ),
            }
        )
    original_macro = float(rows[0]["macro_f1"])
    for row in rows:
        row["macro_f1_delta_vs_original"] = float(
            row["macro_f1"] - original_macro
        )
    result = {
        "schema_version": "v6_1_temporal_order_audit_v1",
        "status": "COMPLETE",
        "candidate": candidate,
        "candidate_role": role,
        "seed": int(protocol["order_audit"]["seed"]),
        "threshold_from_original_inner_oof": original_threshold,
        "results": rows,
        "outer_test_accessed": False,
        "future_accessed": False,
    }
    _atomic_json(ARTIFACT_ROOT / "order_audit.json", result)
    return result


def _final_threshold(
    data: Any,
    v4_protocol: dict[str, Any],
    protocol: dict[str, Any],
    spec: CandidateSpec,
    outer_fold: int,
    device: str,
) -> tuple[float, int]:
    config = _training_config(protocol)
    rows: list[dict[str, Any]] = []
    for inner_fold, (train_index, validation_index) in enumerate(
        _inner_splits(data, outer_fold, v4_protocol)
    ):
        rows.append(
            _run_fit(
                data=data,
                train_index=train_index,
                validation_index=validation_index,
                config=config,
                spec=spec,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                seed=3407,
                scope="final_threshold",
                device=device,
            )
        )
    targets = []
    probabilities = []
    for row in rows:
        value = _load_prediction(row)
        targets.append(value["target"])
        probabilities.append(value["probability"])
    threshold = choose_threshold(np.concatenate(targets), np.concatenate(probabilities))
    epochs = max(1, int(np.median([row["selected_epoch"] for row in rows])))
    return float(threshold["threshold"]), epochs


def _outer_indices(data: Any, outer_fold: int) -> tuple[np.ndarray, np.ndarray]:
    return data.v2.outer_indices(outer_fold)


def _run_final_models(
    selected_candidate: str,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol, v4_protocol, data = _load()
    specs = candidate_specs()
    candidates = [
        "A2_bilstm_current_temporal",
        "B2_cnn_matched_temporal",
        selected_candidate,
    ]
    config = _training_config(protocol)
    prediction_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        spec = specs[candidate]
        for outer_fold in protocol["data"]["final_outer_folds"]:
            threshold, epochs = _final_threshold(
                data, v4_protocol, protocol, spec, int(outer_fold), device
            )
            threshold_rows.append(
                {
                    "candidate": candidate,
                    "outer_fold": int(outer_fold),
                    "threshold": threshold,
                    "fixed_epochs": epochs,
                }
            )
            train_index, validation_index = _outer_indices(data, int(outer_fold))
            for seed in protocol["data"]["final_seeds"]:
                metadata = _run_fit(
                    data=data,
                    train_index=train_index,
                    validation_index=validation_index,
                    config=config,
                    spec=spec,
                    outer_fold=int(outer_fold),
                    inner_fold=-1,
                    seed=int(seed),
                    scope="final",
                    device=device,
                    fixed_epochs=epochs,
                )
                value = _load_prediction(metadata)
                for index, target, probability in zip(
                    value["index"], value["target"], value["probability"]
                ):
                    prediction_rows.append(
                        {
                            "candidate": candidate,
                            "outer_fold": int(outer_fold),
                            "seed": int(seed),
                            "index": int(index),
                            "record_id": str(data.base.record_ids[int(index)]),
                            "id_student": int(data.groups[int(index)]),
                            "target": int(target),
                            "probability": float(probability),
                            "threshold": threshold,
                        }
                    )
    predictions = pd.DataFrame(prediction_rows)
    thresholds = pd.DataFrame(threshold_rows)
    predictions.to_parquet(ARTIFACT_ROOT / "final_outer_predictions.parquet", index=False)
    thresholds.to_csv(ARTIFACT_ROOT / "final_thresholds.csv", index=False)
    return predictions, thresholds


def _frozen_references() -> list[dict[str, Any]]:
    v51 = pd.DataFrame(
        json.loads(
            (ROOT / "artifacts/v5_1/oulad/final_metrics.json").read_text(
                encoding="utf-8"
            )
        )
    )
    v51_row = v51[v51.candidate.eq("cnn_bilstm_full_ensemble")].iloc[0].to_dict()
    v5 = pd.read_csv(ROOT / "artifacts/v5/oulad/final_metrics.csv")
    xgb = v5[v5.candidate.eq("xgboost")]
    if xgb.empty:
        xgb = v5[v5.candidate.astype(str).str.contains("xgboost", case=False)]
    xgb_row = xgb.iloc[0].to_dict()
    return [
        {
            "candidate": "current_frozen_cnn_bilstm_reference",
            "evidence_status": "REUSED_IMMUTABLE",
            **v51_row,
        },
        {
            "candidate": "xgboost_operational_cross_check",
            "evidence_status": "REUSED_IMMUTABLE",
            **xgb_row,
        },
    ]


def _summarize_final(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, seed), frame in predictions.groupby(["candidate", "seed"]):
        metrics = binary_metrics_per_record_threshold(
            frame.target.to_numpy(),
            frame.probability.to_numpy(),
            frame.threshold.to_numpy(),
        )
        metrics["roc_auc"] = float(
            roc_auc_score(frame.target.to_numpy(), frame.probability.to_numpy())
        )
        metrics["recall_at_10_percent"] = _recall_at_fraction(
            frame.target.to_numpy(), frame.probability.to_numpy(), 0.10
        )
        rows.append(
            {
                "candidate": candidate,
                "seed": int(seed),
                "evidence_status": "NEW_OUTER_EVALUATION",
                **metrics,
            }
        )
    seed_metrics = pd.DataFrame(rows)
    aggregate: list[dict[str, Any]] = []
    metrics = [
        "macro_f1",
        "balanced_accuracy",
        "at_risk_precision",
        "at_risk_recall",
        "at_risk_f1",
        "pr_auc",
        "roc_auc",
        "brier",
        "ece",
        "recall_at_10_percent",
    ]
    for candidate, frame in seed_metrics.groupby("candidate"):
        row: dict[str, Any] = {
            "candidate": candidate,
            "evidence_status": "NEW_OUTER_EVALUATION",
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_sd"] = float(frame[metric].std(ddof=1))
        aggregate.append(row)
    result = pd.DataFrame(aggregate)
    references = pd.DataFrame(_frozen_references())
    for metric in metrics:
        if metric in references.columns:
            references[f"{metric}_mean"] = references[metric]
            references[f"{metric}_sd"] = np.nan
    keep = ["candidate", "evidence_status"] + [
        name for metric in metrics for name in (f"{metric}_mean", f"{metric}_sd")
    ]
    result = pd.concat(
        [result.reindex(columns=keep), references.reindex(columns=keep)],
        ignore_index=True,
    )
    result.to_csv(ARTIFACT_ROOT / "final_outer_results.csv", index=False)
    seed_metrics.to_csv(ARTIFACT_ROOT / "final_seed_results.csv", index=False)
    return result


def _bootstrap(
    predictions: pd.DataFrame, selected_candidate: str, replicates: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    probabilities = predictions.pivot_table(
        index=["record_id", "id_student", "target"],
        columns="candidate",
        values="probability",
        aggfunc="mean",
    ).reset_index()
    thresholds = predictions.pivot_table(
        index=["record_id", "id_student", "target"],
        columns="candidate",
        values="threshold",
        aggfunc="first",
    ).reset_index()
    groups = probabilities.id_student.unique()
    comparisons = [
        candidate
        for candidate in ["A2_bilstm_current_temporal", "B2_cnn_matched_temporal"]
        if candidate in probabilities.columns
    ]
    distributions = {candidate: [] for candidate in comparisons}
    for _ in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate(
            [
                np.flatnonzero(probabilities.id_student.to_numpy() == group)
                for group in sampled
            ]
        )
        sample = probabilities.iloc[indices]
        sample_thresholds = thresholds.iloc[indices]
        target = sample.target.to_numpy()
        selected = binary_metrics_per_record_threshold(
            target,
            sample[selected_candidate].to_numpy(),
            sample_thresholds[selected_candidate].to_numpy(),
        )["macro_f1"]
        for candidate in comparisons:
            value = binary_metrics_per_record_threshold(
                target,
                sample[candidate].to_numpy(),
                sample_thresholds[candidate].to_numpy(),
            )["macro_f1"]
            distributions[candidate].append(float(selected - value))
    result = {
        "schema_version": "v6_1_group_paired_bootstrap_v1",
        "status": "COMPLETE",
        "unit": "id_student",
        "replicates": replicates,
        "selected_candidate": selected_candidate,
        "comparisons": {},
    }
    for candidate, values in distributions.items():
        result["comparisons"][candidate] = {
            "mean_delta_macro_f1": float(np.mean(values)),
            "ci_95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            "probability_delta_above_zero": float(np.mean(np.asarray(values) > 0)),
        }
    return result


def run_final(device: str = "cuda") -> dict[str, Any]:
    selected = yaml.safe_load(
        (ARTIFACT_ROOT / "selected_config.yaml").read_text(encoding="utf-8")
    )
    selected_candidate = selected["selected_candidate"]
    if selected_candidate is None:
        pd.DataFrame(
            columns=[
                "candidate",
                "evidence_status",
                "macro_f1_mean",
                "macro_f1_sd",
            ]
        ).to_csv(ARTIFACT_ROOT / "final_outer_results.csv", index=False)
        result = {
            "status": "SKIPPED_DEVELOPMENT_GATE_NOT_PASSED",
            "selected_candidate": None,
            "outer_test_accessed": False,
            "future_accessed": False,
        }
        _atomic_json(ARTIFACT_ROOT / "bootstrap_results.json", result)
        _atomic_json(ARTIFACT_ROOT / "final_run_state.json", result)
        return result
    predictions, _ = _run_final_models(selected_candidate, device)
    results = _summarize_final(predictions)
    bootstrap = _bootstrap(
        predictions,
        selected_candidate,
        replicates=2000,
        seed=7319,
    )
    _atomic_json(ARTIFACT_ROOT / "bootstrap_results.json", bootstrap)
    result = {
        "status": "COMPLETE",
        "selected_candidate": selected_candidate,
        "result_rows": int(len(results)),
        "outer_test_accessed_only_after_freeze": True,
        "future_accessed": False,
    }
    _atomic_json(ARTIFACT_ROOT / "final_run_state.json", result)
    return result


def recommendation_audit() -> dict[str, Any]:
    ranking = json.loads(
        (ROOT / "artifacts/v6/prediction/ranking/gate.json").read_text(
            encoding="utf-8"
        )
    )
    withdrawal_values: list[float] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "withdrawal_recall" and isinstance(item, (int, float)):
                    withdrawal_values.append(float(item))
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(ranking)
    result = {
        "schema_version": "v6_1_recommendation_logic_audit_v1",
        "status": "FIXED",
        "circular_semantic_logic_found": True,
        "previous_logic": [
            "activity_level = 1 - withdrawal_risk_horizon",
            "inactivity_streak = round(withdrawal_risk_horizon * 10)",
            "assessment_progress = 1 - probability_fail",
            "grade_trend = probability_pass + probability_distinction - probability_fail",
        ],
        "fix": {
            "prediction_outputs_allowed": [
                "risk probability",
                "fail probability",
                "withdrawal/hazard exploratory risk",
                "uncertainty",
                "confidence",
                "disagreement",
                "priority/escalation support",
            ],
            "observed_state_source": "REAL_PRE_CUTOFF_F2_MIDDLE_SEQUENCE_V1",
            "observed_fields": {
                "activity_level": "log-scaled mean total clicks in the last two valid weeks",
                "inactivity_streak": "actual current pre-cutoff inactivity streak channel",
                "assessment_progress": "actual submitted / currently available assessment count",
                "grade_trend": "actual recent cumulative mean score change",
            },
            "missing_observed_state": "abstain instead of synthesizing behavior from probabilities",
        },
        "withdrawal": {
            "observed_recall_values": withdrawal_values,
            "maximum_observed_recall": max(withdrawal_values, default=None),
            "reliability_gate_passed": False,
            "status": "EXPLORATORY_DISABLED_FOR_RECOMMENDATION",
            "decision": "withdrawal horizon retained in profile but cannot assert engagement mechanism or select mechanism-specific action",
        },
        "frozen_v6_artifacts_modified": False,
    }
    _atomic_json(ARTIFACT_ROOT / "recommendation_logic_audit.json", result)
    return result


def _checksums() -> None:
    excluded = {"checksums.sha256"}
    rows = []
    for path in sorted(ARTIFACT_ROOT.rglob("*")):
        if path.is_file() and path.name not in excluded and "logs" not in path.parts:
            rows.append(
                f"{sha256_file(path)}  {path.relative_to(ARTIFACT_ROOT).as_posix()}"
            )
    (ARTIFACT_ROOT / "checksums.sha256").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def run_all(device: str = "cuda") -> dict[str, Any]:
    started = time.perf_counter()
    inner = run_inner(device)
    order = run_order_audit(device)
    final = run_final(device)
    recommendation = recommendation_audit()
    result = {
        "status": "COMPUTE_COMPLETE",
        "inner": inner,
        "order": order,
        "final": final,
        "recommendation": recommendation,
        "runtime_seconds": time.perf_counter() - started,
    }
    _atomic_json(ARTIFACT_ROOT / "compute_run_state.json", result)
    _checksums()
    return result


__all__ = [
    "architecture_audit",
    "recommendation_audit",
    "run_all",
    "run_final",
    "run_inner",
    "run_order_audit",
]
