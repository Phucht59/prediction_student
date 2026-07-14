"""Frozen contracts and validation helpers for Neural Sanity Ablation V2.2."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd
import math

from src.evaluation.protocol import canonical_json

EXPERIMENTS = {
    "S0": {"experiment_id": "S0", "label": "control_replay", "drop_last_train": True},
    "S1": {"experiment_id": "S1", "label": "drop_last_only", "drop_last_train": False},
    "S2": {"experiment_id": "S2", "label": "kernel_one_only", "drop_last_train": True, "cnn_kernel_size": 1},
    "S3": {"experiment_id": "S3", "label": "training_budget_only", "drop_last_train": True, "max_epochs": 40, "patience": 8, "scheduler_patience": 3},
    "S4": {"experiment_id": "S4", "label": "drop_last_kernel_one", "drop_last_train": False, "cnn_kernel_size": 1},
    "S5": {"experiment_id": "S5", "label": "full_sanity_configuration", "drop_last_train": False, "cnn_kernel_size": 1, "max_epochs": 40, "patience": 8, "scheduler_patience": 3},
}
SEEDS = (42, 52, 62, 72, 82)
JOB_COLUMNS = ("experiment_id", "scenario", "model_name", "outer_fold", "training_seed")


def checksum(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def variant_config(source_config: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    if experiment_id not in EXPERIMENTS:
        raise ValueError(f"Unknown frozen experiment: {experiment_id}")
    output = dict(source_config)
    output.update({key: value for key, value in EXPERIMENTS[experiment_id].items()
                   if key in {"cnn_kernel_size", "max_epochs", "patience", "scheduler_patience"}})
    output["drop_last_train"] = bool(EXPERIMENTS[experiment_id]["drop_last_train"])
    return output


def feature_contract(fold_checksum: str, dataset_version: int = 1) -> dict[str, Any]:
    contract = {"contract_version": "neural_sanity_v2_2", "scenario": "late_stage",
                "cutoff": "before_outcome_after_g2", "feature_set_id": "G1+G2",
                "ordered_features": ["G1", "G2"], "preprocessing_contract": "fold_train_fit_minmax_selector",
                "scaler_contract": "minmax_train_only", "target_excluded": True,
                "temporal_availability_status": "verified_by_late_stage_allowlist",
                "class_order": ["Low", "Medium", "High"], "dataset_version": dataset_version,
                "fold_manifest_checksum": fold_checksum}
    contract["semantic_checksum"] = checksum(contract)
    return contract


def build_expected_job_contract(run_id: str, source_configs: dict[int, dict[str, Any]], validation_counts: dict[int, int], fold_checksum: str) -> dict[str, Any]:
    feature = feature_contract(fold_checksum)
    jobs=[]
    for experiment_id in EXPERIMENTS:
        for fold, count in sorted(validation_counts.items()):
            config = variant_config(source_configs[int(fold)], experiment_id)
            for seed in SEEDS:
                jobs.append({"run_id": run_id, "experiment_id": experiment_id, "scenario": "late_stage",
                             "model_name": "cnn_bilstm", "outer_fold": int(fold), "training_seed": int(seed),
                             "expected_record_count": int(count), "config_checksum": checksum(config),
                             "feature_contract_checksum": feature["semantic_checksum"], "fold_manifest_checksum": fold_checksum})
    contract={"contract_version":"neural_sanity_v2_2","source":"frozen_matrix_plus_selected_v2_configs_plus_fold_manifest",
              "feature_contract":feature,"jobs":jobs}
    contract["semantic_checksum"] = checksum(contract)
    return contract


def duplicate_job_rows(frame: pd.DataFrame) -> int:
    return int(frame.duplicated(list(JOB_COLUMNS), keep=False).sum())


def expected_job_keys(contract: dict[str, Any]) -> set[tuple[Any, ...]]:
    return {tuple(row[column] for column in JOB_COLUMNS) for row in contract["jobs"]}


def loader_statistics(dataset_size: int, batch_size: int, drop_last_train: bool) -> dict[str, int | bool]:
    """Deterministic sampler-pass accounting used by V2.2 diagnostics."""
    if dataset_size < 1 or batch_size < 1:
        raise ValueError("dataset_size and batch_size must be positive.")
    remainder = dataset_size % batch_size
    consumed = dataset_size - (remainder if drop_last_train and remainder else 0)
    return {"dataset_size": dataset_size, "batch_size": batch_size,
            "n_batches": dataset_size // batch_size if drop_last_train else math.ceil(dataset_size / batch_size),
            "final_batch_size": remainder if remainder else min(batch_size, dataset_size),
            "samples_consumed_per_epoch": consumed, "samples_dropped_per_epoch": dataset_size-consumed,
            "drop_last_train": bool(drop_last_train)}
