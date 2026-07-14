"""Frozen V3.1 model-selection, estimator, and validation contracts."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.evaluation.protocol import canonical_json

SEEDS = (42, 52, 62, 72, 82)
CLASS_ORDER = ("Low", "Medium", "High")
V3_1_PROTOCOL_VERSION = "model_v3_1"

# Candidate families only. Historical and strong baselines are deliberately excluded.
MODEL_REGISTRY = {
    "M0": {"name": "pytorch_nominal_mlp", "ordinal": False, "regression": False,
           "target_supervision": "classification_only", "tracks": ["late_stage", "early_warning"],
           "training_engine": "pytorch_tabular_v3_1"},
    "M1": {"name": "pytorch_ordinal_mlp", "ordinal": True, "regression": False,
           "target_supervision": "classification_only", "tracks": ["late_stage", "early_warning"],
           "training_engine": "pytorch_tabular_v3_1"},
    "M2": {"name": "pytorch_multitask_ordinal_mlp", "ordinal": True, "regression": True,
           "target_supervision": "continuous_g3_enriched", "tracks": ["late_stage", "early_warning"],
           "training_engine": "pytorch_tabular_v3_1"},
    "M3": {"name": "pytorch_multitask_nominal_mlp", "ordinal": False, "regression": True,
           "target_supervision": "continuous_g3_enriched", "tracks": ["late_stage", "early_warning"],
           "training_engine": "pytorch_tabular_v3_1"},
    "M4": {"name": "fixed_sequence_backbone_ordinal_head_diagnostic", "ordinal": True,
           "regression": False, "target_supervision": "classification_only", "tracks": ["late_stage"],
           "training_engine": "pytorch_sequence_s3_fixed"},
    "B0": {"name": "ridge_g3_regression_baseline", "ordinal": False, "regression": True,
           "target_supervision": "continuous_g3_enriched", "tracks": ["late_stage", "early_warning"],
           "training_engine": "sklearn_ridge_deterministic"},
}

FIXED_REFERENCE_REGISTRY = {
    "REF_G2_RULE": "fixed late-stage G2 threshold reference",
    "REF_LOGISTIC_G2": "fixed late-stage logistic G2 reference",
    "REF_HGB": "fixed same-information HGB reference",
    "REF_SK_MLP": "exact Benchmark V2 scikit-learn Small MLP historical/source continuity reference; not a matched control",
    "REF_BILSTM_ONLY": "Benchmark V2 BiLSTM-only reference",
    "REF_CNN_S3_NOMINAL": "Neural Sanity V2.2 S3 nominal sequence reference",
}


def checksum(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def deterministic_study_seed(protocol_version: str, model_family: str, track: str, outer_fold: int) -> int:
    """Stable sampler seed independent of the five refit seeds."""
    text = f"{protocol_version}|{model_family}|{track}|{outer_fold}".encode()
    return int(hashlib.sha256(text).hexdigest()[:8], 16) % (2**31 - 1)


def build_selection_study_contract(run_id: str, fold_checksum: str, source_commit: str,
                                   search_space_checksum: str, target_contract: dict[str, Any], *,
                                   smoke: bool = False) -> dict[str, Any]:
    """Create selection studies before any compute; M4 is fixed and B0 uses its own ridge grid."""
    studies: list[dict[str, Any]] = []
    tracks = ("late_stage",) if smoke else ("late_stage", "early_warning")
    folds = (0,) if smoke else tuple(range(5))
    trial_budget = 1 if smoke else 20
    for family in ("M0", "M1", "M2", "M3"):
        for track in tracks:
            for fold in folds:
                study_id = f"{run_id}:{family}:{track}:outer{fold}"
                studies.append({
                    "study_id": study_id, "model_family": family, "track": track, "outer_fold": fold,
                    "study_seed": deterministic_study_seed(V3_1_PROTOCOL_VERSION, family, track, fold),
                    "inner_fold_checksum": checksum({"fold_manifest_checksum": fold_checksum, "outer_fold": fold, "inner_folds": 3}),
                    "trial_budget": trial_budget, "expected_trials": trial_budget,
                    "expected_inner_evaluations": trial_budget * 3,
                    "search_space_checksum": search_space_checksum,
                    "target_supervision_checksum": target_contract["semantic_checksum"],
                    "source_commit": source_commit,
                })
    contract = {"contract_version": V3_1_PROTOCOL_VERSION, "run_id": run_id,
                "created_before_compute": True, "selection_seed_is_not_training_seed": True,
                "studies": studies}
    contract["semantic_checksum"] = checksum(contract)
    return contract


def validate_selection_results(contract: dict[str, Any], trials: pd.DataFrame,
                               selected: pd.DataFrame) -> dict[str, int]:
    """Strictly validate study/trial/inner-fold evidence independently of predictions."""
    expected = {x["study_id"]: x for x in contract["studies"]}
    actual_ids = set(trials["study_id"].unique()) if not trials.empty else set()
    missing_studies = len(set(expected) - actual_ids)
    unexpected_studies = len(actual_ids - set(expected))
    duplicate_studies = int(trials[["study_id", "trial_id", "inner_fold"]].duplicated(keep=False).sum()) if not trials.empty else 0
    missing_trials = missing_inner = unexpected_trials = 0
    for study_id, study in expected.items():
        subset = trials[trials["study_id"] == study_id]
        ids = set(subset["trial_id"].unique())
        desired = set(range(study["expected_trials"]))
        missing_trials += len(desired - ids)
        unexpected_trials += len(ids - desired)
        for trial_id in desired & ids:
            folds = set(subset.loc[subset["trial_id"] == trial_id, "inner_fold"].unique())
            if folds != {0, 1, 2}:
                missing_inner += 1
    selected_not_completed = 0
    for row in selected.itertuples(index=False):
        found = trials[(trials.study_id == row.study_id) & (trials.trial_id == row.selected_trial_id)]
        if found.empty or not found.config_checksum.eq(row.config_checksum).all():
            selected_not_completed += 1
    return {"missing_studies": missing_studies, "unexpected_studies": unexpected_studies,
            "duplicate_study_trial_rows": duplicate_studies, "missing_trials": missing_trials,
            "unexpected_trials": unexpected_trials, "trials_missing_complete_inner_folds": missing_inner,
            "selected_config_not_completed": selected_not_completed}


def build_expected_jobs(run_id: str, fold_counts: dict[int, int], fold_checksum: str, source_commit: str,
                        feature_contracts: dict[str, Any], target_contract: dict[str, Any], *,
                        smoke: bool = False, config_checksums: dict[str, str] | None = None,
                        selection_contract_checksum: str | None = None) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    active_folds = {0: fold_counts[0]} if smoke else fold_counts
    tracks = ("late_stage",) if smoke else ("late_stage", "early_warning")
    for family, model in MODEL_REGISTRY.items():
        for track in tracks:
            if track not in model["tracks"]:
                continue
            feature = feature_contracts[track]
            seeds = (0,) if family == "B0" else ((42,) if smoke else SEEDS)
            for fold, count in active_folds.items():
                for seed in seeds:
                    config = {"model_family": family, "track": track, "smoke": smoke}
                    jobs.append({
                        "run_id": run_id, "model_family": family, "track": track, "scenario": track,
                        "feature_set_id": feature["feature_set_id"], "target_supervision_type": model["target_supervision"],
                        "training_engine": model["training_engine"], "outer_fold": fold, "training_seed": seed,
                        "estimator_group": f"{family}:{track}:outer{fold}", "expected_record_count": count,
                        # Full-run configs do not exist until the inner study completes. A synthetic
                        # checksum here would make a future refit look provenance-valid when it is not.
                        "config_checksum": (config_checksums or {}).get(family),
                        "selected_config_required": family in {"M0", "M1", "M2", "M3", "B0"},
                        "fold_manifest_checksum": fold_checksum, "feature_contract_checksum": feature["semantic_checksum"],
                        "target_contract_checksum": target_contract["semantic_checksum"],
                        "selection_study_contract_checksum": selection_contract_checksum,
                        "source_commit": source_commit,
                    })
    contract = {"contract_version": V3_1_PROTOCOL_VERSION, "run_id": run_id,
                "created_before_compute": True, "jobs": jobs}
    contract["semantic_checksum"] = checksum(contract)
    return contract


def duplicate_jobs(frame: pd.DataFrame) -> int:
    cols = ["model_family", "track", "outer_fold", "training_seed"]
    return int(frame.duplicated(cols, keep=False).sum())


def legacy_intersection(development_ids: set[str], legacy_ids: set[str]) -> set[str]:
    return development_ids & legacy_ids


def validate_shape_rows(frame: pd.DataFrame) -> bool:
    for row in frame.itertuples():
        expected = 2 if int(row.cnn_kernel_size) == 1 else 3 if int(row.cnn_kernel_size) == 2 else None
        if expected is None or int(row.cnn_output_sequence_length) != expected or int(row.bilstm_input_sequence_length) != expected:
            return False
    return True


def validate_loader_rows(frame: pd.DataFrame) -> bool:
    for row in frame.itertuples():
        n, batch = int(row.dataset_size), int(row.batch_size)
        dropped = 0 if not bool(row.drop_last_train) else n % batch
        if int(row.samples_dropped_per_epoch) != dropped or int(row.samples_consumed_per_epoch) != n - dropped:
            return False
    return True


def map_g3_to_class(values: np.ndarray) -> np.ndarray:
    """Frozen raw-value mapping: <10 Low, [10,15) Medium, >=15 High; no rounding."""
    x = np.asarray(values, dtype=float)
    return np.where(x < 10.0, 0, np.where(x < 15.0, 1, 2)).astype(int)


def regression_metric_summary(y_true: np.ndarray, raw_prediction: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y_true, dtype=float), np.asarray(raw_prediction, dtype=float)
    return {"mae_raw": float(mean_absolute_error(y, p)), "rmse_raw": float(mean_squared_error(y, p) ** 0.5),
            "r2_raw": float(r2_score(y, p))}


def pooled_oof_regression_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Primary R²: one pooled 316-record OOF computation per model/track/seed."""
    rows = []
    for key, group in predictions.dropna(subset=["predicted_g3_raw"]).groupby(["model_family", "track", "training_seed"]):
        if group.record_id.nunique() != len(group):
            raise ValueError("Pooled OOF regression rows contain duplicate records.")
        rows.append({"model_family": key[0], "track": key[1], "training_seed": key[2],
                     "n_records": len(group), "aggregation": "pooled_oof_primary", **regression_metric_summary(group.raw_g3, group.predicted_g3_raw)})
    return pd.DataFrame(rows)


def validate_outer_refit_config_checksums(rows: pd.DataFrame) -> bool:
    """One selected configuration must drive all five seed refits of an outer fold."""
    grouped = rows.groupby(["model_family", "track", "outer_fold"], dropna=False)
    return all(group.config_checksum.nunique() == 1 for _, group in grouped)


def validate_full_preflight(expected_jobs: dict[str, Any], selection_contract: dict[str, Any],
                            source_commit: str) -> None:
    """Reject V3.0/placeholder/stale contracts before a future full benchmark can start."""
    if expected_jobs.get("contract_version") != V3_1_PROTOCOL_VERSION:
        raise ValueError("Full V3 requires a V3.1 expected-job contract.")
    run_id = str(expected_jobs.get("run_id", ""))
    if not run_id or "PLACEHOLDER" in run_id or "REQUIRES_REVIEW" in run_id:
        raise ValueError("Full V3 run ID is not allocated.")
    if any(job.get("source_commit") != source_commit for job in expected_jobs.get("jobs", [])):
        raise ValueError("Expected-job contract source commit does not match the executable source.")
    if selection_contract.get("contract_version") != V3_1_PROTOCOL_VERSION:
        raise ValueError("Full V3 requires a V3.1 selection-study contract.")
    if any(study.get("source_commit") != source_commit for study in selection_contract.get("studies", [])):
        raise ValueError("Selection-study contract source commit does not match the executable source.")
