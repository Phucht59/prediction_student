"""Pure protocol helpers for the approved Strategy B Phase A-B run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from src.data_pipeline import SOURCE_ROW_NUMBER_COLUMN
from src.evaluation.protocol import canonical_json, source_record_identity
from src.model_selection import split_model_train_and_early_stop


PHASE_AB_PROTOCOL_VERSION = "strategy_b_phase_ab_v1"
APPROVED_SEEDS = [42, 123, 155]


def approved_candidate_registry() -> dict[str, Any]:
    return {
        "registry_version": "strategy_b_candidate_registry_v2",
        "activation_gate": "Phase A-B strict_validation PASS and explicit Phase C approval",
        "official_candidates": [
            {"id": "R0", "name": "G2 deterministic rule", "family": "rule"},
            {"id": "M1", "name": "Random Forest", "family": "machine_learning"},
            {"id": "M2", "name": "SVM", "family": "machine_learning"},
            {"id": "N0", "name": "Corrected compact nominal CNN-BiLSTM", "family": "cnn_bilstm"},
            {"id": "N1", "name": "Corrected compact ordinal CNN-BiLSTM", "family": "cnn_bilstm"},
            {"id": "N2", "name": "Tiny nominal MLP", "family": "neural_mlp"},
            {"id": "N3", "name": "Tiny ordered MLP", "family": "neural_mlp"},
            {"id": "A1", "name": "Parameter-matched CNN-only", "family": "ablation"},
            {"id": "A2", "name": "Parameter-matched BiLSTM-only", "family": "ablation"},
        ],
        "conditional_candidates": [
            {
                "id": "C1",
                "name": "Ordinal CNN-BiLSTM + Huber auxiliary regression",
                "gate": "pre_registered_ordinal_signal_gate",
            },
            {
                "id": "C2",
                "name": "Gated residual ordinal CNN-BiLSTM",
                "gate": "pre_registered_ordinal_signal_gate",
            },
        ],
        "reporting_contract": {
            "best_overall_model": "best model across the complete approved ML and DL pool",
            "best_thesis_hybrid_model": "best model within the CNN-BiLSTM family",
            "must_be_same_model": False,
        },
        "phase_ab_training_restriction": "No ordinal, residual, multitask or Phase C candidate is trained in Phase A-B.",
    }


def evidence_quarantine_registry() -> dict[str, Any]:
    """Classify existing evidence without reading prediction payloads."""

    return {
        "registry_version": "strategy_b_evidence_quarantine_v1",
        "entries": [
            {
                "id": "legacy_79",
                "paths": ["artifacts/final/*/locked_test_predictions.csv"],
                "status": "legacy_heldout_observed",
                "allowed_use": "historical context and checksum-only inventory",
                "prohibited_use": [
                    "model_selection", "architecture_selection", "hyperparameter_selection",
                    "calibration", "threshold_tuning", "final_confirmation",
                ],
                "payload_accessed_in_phase_ab": False,
            },
            {
                "id": "fair_cnn_rows",
                "paths": ["artifacts/baseline_comparison/fair-model-comparison-full/*"],
                "affected_rows": ["cnn_lstm", "cnn_bilstm"],
                "status": "invalid_protocol_config_resolution",
                "reason": "fixed loss/class-weight/resampling constants were absent from outer resolved params",
            },
            {
                "id": "nested_full_20260710",
                "paths": ["artifacts/model_selection/nested-full-20260710/*"],
                "status": "historical_old_estimator",
                "reason": "run predates the corrected full-partition refit estimator and exact source provenance",
            },
            {
                "id": "small_diagnostics",
                "paths": ["reports/project_strategy_v1/*", "historical smoke and sanity outputs"],
                "status": "diagnostic_or_feasibility_evidence",
                "allowed_use": "bug confirmation and hypothesis prioritization only",
                "prohibited_use": ["official_ranking", "final_model_selection", "superiority_claim"],
            },
        ],
    }


def development_source_rows(manifest: dict[str, Any]) -> list[int]:
    rows = sorted(int(row["source_row_number"]) for row in manifest["development_records"])
    if len(rows) != len(set(rows)) or len(rows) != 316:
        raise ValueError("Protocol V2 must contain exactly 316 unique development source rows.")
    return rows


def source_rows_hash(rows: Iterable[int]) -> str:
    values = [int(value) for value in rows]
    return hashlib.sha256(canonical_json(values).encode("utf-8")).hexdigest()


def assert_development_only_frame(frame: pd.DataFrame, manifest: dict[str, Any]) -> None:
    expected = development_source_rows(manifest)
    observed = sorted(frame[SOURCE_ROW_NUMBER_COLUMN].astype(int).tolist())
    if observed != expected:
        raise ValueError("Loaded frame is not exactly the immutable 316-record development cohort.")


def materialize_inner_fold_ledger(
    development: pd.DataFrame,
    outer_folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    dataset_version_id: int,
    target_col: str,
    inner_folds: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outer_fold, (outer_train_idx, _) in enumerate(outer_folds):
        outer_train = development.iloc[outer_train_idx]
        labels = outer_train[target_col].astype(int).to_numpy()
        splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + outer_fold)
        for inner_fold, (inner_train_idx, inner_validation_idx) in enumerate(splitter.split(outer_train, labels)):
            role_by_position = {
                int(position): "inner_train" for position in inner_train_idx
            }
            role_by_position.update({int(position): "inner_validation" for position in inner_validation_idx})
            for position in range(len(outer_train)):
                source_row = int(outer_train.iloc[position][SOURCE_ROW_NUMBER_COLUMN])
                rows.append({
                    "dataset_version_id": int(dataset_version_id),
                    "source_record_identity": source_record_identity(dataset_version_id, source_row),
                    "source_row_number": source_row,
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "role": role_by_position[position],
                    "split_seed": int(seed + outer_fold),
                })
    ledger = pd.DataFrame(rows).sort_values(
        ["outer_fold", "inner_fold", "role", "source_row_number"]
    ).reset_index(drop=True)
    expected_rows = sum(len(train_idx) for train_idx, _ in outer_folds) * inner_folds
    if len(ledger) != expected_rows:
        raise RuntimeError("Inner-fold ledger does not cover every outer-training row per inner fold.")
    return ledger


def materialize_early_stop_ledger(
    development: pd.DataFrame,
    outer_folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    dataset_version_id: int,
    target_col: str,
    seeds: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outer_fold, (outer_train_idx, outer_validation_idx) in enumerate(outer_folds):
        outer_train = development.iloc[outer_train_idx].copy()
        outer_validation = development.iloc[outer_validation_idx].copy()
        for seed in seeds:
            model_train, early_stop = split_model_train_and_early_stop(
                outer_train,
                target_col,
                seed=int(seed),
            )
            roles = {
                int(source_row): "model_train"
                for source_row in model_train[SOURCE_ROW_NUMBER_COLUMN].astype(int)
            }
            roles.update({
                int(source_row): "early_stop"
                for source_row in early_stop[SOURCE_ROW_NUMBER_COLUMN].astype(int)
            })
            roles.update({
                int(source_row): "outer_validation"
                for source_row in outer_validation[SOURCE_ROW_NUMBER_COLUMN].astype(int)
            })
            if len(roles) != len(development):
                raise RuntimeError("Early-stop ledger does not cover the full development cohort.")
            for source_row in sorted(roles):
                rows.append({
                    "dataset_version_id": int(dataset_version_id),
                    "source_record_identity": source_record_identity(dataset_version_id, source_row),
                    "source_row_number": int(source_row),
                    "outer_fold": int(outer_fold),
                    "seed": int(seed),
                    "role": roles[source_row],
                })
    return pd.DataFrame(rows).sort_values(
        ["outer_fold", "seed", "role", "source_row_number"]
    ).reset_index(drop=True)


def recompute_metrics_from_oof(oof: pd.DataFrame) -> pd.DataFrame:
    required = {"policy_id", "seed", "outer_fold", "true_label", "predicted_label"}
    missing = sorted(required - set(oof.columns))
    if missing:
        raise ValueError(f"OOF frame is missing metric columns: {missing}")
    rows: list[dict[str, Any]] = []
    for (policy_id, seed, outer_fold), frame in oof.groupby(
        ["policy_id", "seed", "outer_fold"], sort=True
    ):
        rows.append({
            "policy_id": str(policy_id),
            "seed": int(seed),
            "outer_fold": int(outer_fold),
            "records": int(len(frame)),
            "macro_f1": float(f1_score(frame["true_label"], frame["predicted_label"], average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(frame["true_label"], frame["predicted_label"])),
        })
    return pd.DataFrame(rows).sort_values(["policy_id", "seed", "outer_fold"]).reset_index(drop=True)


def validate_oof_coverage(
    oof: pd.DataFrame,
    *,
    development_rows: list[int],
    policy_ids: list[str],
    seeds: list[int],
) -> None:
    expected = sorted(int(value) for value in development_rows)
    for policy_id in policy_ids:
        for seed in seeds:
            selected = oof[(oof["policy_id"] == policy_id) & (oof["seed"] == seed)]
            observed = sorted(selected["source_row_number"].astype(int).tolist())
            if observed != expected:
                raise ValueError(f"OOF coverage mismatch for policy={policy_id}, seed={seed}.")
            if selected["source_row_number"].duplicated().any():
                raise ValueError(f"Duplicate OOF source rows for policy={policy_id}, seed={seed}.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
