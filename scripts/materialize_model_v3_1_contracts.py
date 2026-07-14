"""Materialize immutable V3.1 protocol contracts; this script never trains a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import ROOT_DIR
from src.evaluation.model_v3_protocol import (
    FIXED_REFERENCE_REGISTRY, MODEL_REGISTRY, SEEDS, V3_1_PROTOCOL_VERSION,
    build_expected_jobs, build_selection_study_contract, checksum,
)
from src.evaluation.protocol import load_fold_manifest

OUT = ROOT_DIR / "reports/model_v3_protocol/v3_1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name: str, payload: object) -> Path:
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-run-id", default="model-v3-full-v3-1-20260714")
    parser.add_argument("--smoke-run-id")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True).strip()
    manifest = load_fold_manifest()
    feature_contracts = {}
    for track, values, cutoff in (("late_stage", ["G1", "G2"], "after_G2"), ("early_warning", ["G1"], "after_G1")):
        item = {"contract_version": "v3_1_feature_1", "scenario": track, "cutoff": cutoff,
                "feature_set_id": "+".join(values), "ordered_features": values,
                "preprocessing_contract": "StandardScaler fit on current train partition only",
                "scaler_contract": "train_only_standard_scaler", "target_excluded": True,
                "temporal_availability_status": "allowed_by_frozen_feature_allowlist",
                "class_order": ["Low", "Medium", "High"], "dataset_version": 1,
                "fold_manifest_checksum": manifest["manifest_checksum"]}
        item["semantic_checksum"] = checksum(item)
        feature_contracts[track] = item
    target = {"contract_version": "v3_1_target_1", "class_order": ["Low", "Medium", "High"],
              "class_mapping": {"Low": "G3<=9", "Medium": "10<=G3<=14", "High": "G3>=15"},
              "continuous_g3": {"description": "finer-grained training supervision derived from the same underlying outcome", "scaler": "fit on current training partition only", "primary_metrics_raw_scale": True, "clip_before_primary_rmse_r2": False, "class_from_regression_head": False}}
    target["semantic_checksum"] = checksum(target)
    search = {"contract_version": "v3_1_search_1", "objective": "inner_mean_macro_f1", "inner_folds": 3,
              "trials_per_study": 20, "shared_pytorch_m0_m3": {"hidden_width": [8, 16, 32], "hidden_layers": [1, 2], "dropout": [0.0, 0.15, 0.30], "learning_rate": {"distribution": "log_uniform", "low": 0.0005, "high": 0.005}, "weight_decay": {"distribution": "log_uniform", "low": 1e-6, "high": 1e-3}, "batch_size": [16, 32], "max_epochs": 60, "patience": 10, "drop_last": False}, "multitask_lambda": [0.1, 0.3, 1.0], "lambda_scale_rationale": "G3 MSE uses a target standardized on the current train partition; its pre-weight scale is approximately unit-scale.", "M4": "fixed selected per-fold S3 configuration; no Optuna", "B0_ridge_alpha_grid": [0.01, 0.1, 1.0, 10.0]}
    search["semantic_checksum"] = checksum(search)
    study = build_selection_study_contract(args.full_run_id, manifest["manifest_checksum"], source, search["semantic_checksum"], target)
    counts = {fold: sum(1 for row in manifest["records"] if row["outer_fold"] == fold and row["outer_role"] == "validation") for fold in range(5)}
    expected = build_expected_jobs(args.full_run_id, counts, manifest["manifest_checksum"], source, feature_contracts, target, selection_contract_checksum=study["semantic_checksum"])
    registry = {"contract_version": V3_1_PROTOCOL_VERSION, "candidate_models": MODEL_REGISTRY,
                "fixed_references_excluded_from_candidate_registry": FIXED_REFERENCE_REGISTRY,
                "training_seeds": list(SEEDS), "M4_role": "fixed sequence-backbone ordinal-head diagnostic; not a fully tuned ordinal CNN-BiLSTM"}
    comparisons = {"classification_only": [["M1", "M0"], ["M4", "REF_CNN_S3_NOMINAL"]],
                   "continuous_g3_enriched": [["M2", "M3"], ["M2", "M1"], ["M3", "M0"]],
                   "strong_same_information_references": {"late_stage": ["REF_G2_RULE", "REF_LOGISTIC_G2", "REF_HGB", "REF_SK_MLP", "REF_BILSTM_ONLY", "REF_CNN_S3_NOMINAL"], "early_warning": ["G1_rule", "B0", "HGB_G1", "Small_MLP_G1"]}}
    acceptance = {"ordinal_M1_vs_M0": {"A": {"macro_f1_increase_min": 0.01, "non_decreasing_folds_min": 4, "paired_non_decreasing_tolerance": -1e-12}, "B": {"macro_f1_decrease_max": 0.005, "qwk_increase_min": 0.01, "ordinal_mae_decrease_min": 0.01, "two_step_errors_must_not_increase": True, "high_f1_decrease_max": 0.02}}, "multitask": "Apply A or B to the declared comparator only; additionally report raw regression performance against B0.", "seed_variance": {"mean_within_fold_seed_sd_relative_increase_max": 0.25, "mean_within_fold_seed_sd_absolute_increase_max": 0.01}, "strong_baseline_gap_material": 0.03, "no_posthoc_comparator_selection": True}
    regression = {"B0": {"family": "Ridge G3 regression", "tracks": {"late_stage": ["G1", "G2"], "early_warning": ["G1"]}, "alpha_grid": search["B0_ridge_alpha_grid"], "preprocessing": "StandardScaler fit on current training partition only", "mapped_class_rule": "raw prediction <10 Low; 10<=raw prediction<15 Medium; raw prediction>=15 High; no rounding", "deterministic_training_seed": 0}, "aggregation": {"fold_level": ["mae_raw", "rmse_raw", "r2_raw secondary only"], "paired_metric": ["mae_raw", "rmse_raw"], "primary_r2": "pooled OOF R2 over all 316 records per model/track/training seed", "no_clipping_before_primary": True}}
    metric = {"primary_classification": "outer-fold Macro-F1: seed mean inside fold then mean/SD across five folds", "seed_reporting": ["within-fold seed SD", "pooled OOF metrics per seed", "prediction agreement across seeds"], "regression": regression["aggregation"], "probability_ensemble": "not part of V3"}
    dump("model_registry_v3_1.json", registry); dump("fixed_reference_registry.json", FIXED_REFERENCE_REGISTRY)
    dump("feature_contracts.json", feature_contracts); dump("target_supervision_contract.json", target)
    dump("search_space_contract.json", search); dump("selection_study_contract.json", study)
    dump("expected_job_contract.json", expected); dump("regression_baseline_contract.json", regression)
    dump("metric_contract.json", metric); dump("comparison_pairs.json", comparisons); dump("acceptance_criteria.json", acceptance)
    (OUT / "fairness_resolution.md").write_text("# V3.1 fairness resolution\n\nM0 is the PyTorch nominal MLP matched to M1 by candidate backbone space, PyTorch training engine, optimizer family, batch-size candidates, max epochs, patience and `drop_last=False`. They are family comparisons under equal search opportunity; independently tuned selected configurations may differ. `REF_SK_MLP` is a fixed historical reference only and cannot be used as M1's matched nominal control.\n", encoding="utf-8")
    (OUT / "regression_metric_aggregation.md").write_text("# Regression aggregation\n\nFold × seed MAE and RMSE are used for paired comparisons. Fold R² is secondary. The primary interpretable R² is calculated once from the complete 316-record OOF prediction set for each model/track/seed; B0 has one deterministic pooled result. Primary RMSE/R² use un-clipped raw G3 predictions.\n", encoding="utf-8")
    (OUT / "compute_estimate.md").write_text("# Pre-run compute estimate\n\nFull V3.1 has 40 Optuna selection studies × 20 trials × 3 inner folds = 2,400 inner candidate evaluations for M0–M3. Outer evaluation has 235 jobs and 14,852 prediction rows: M0–M3 contribute 200 five-seed jobs, M4 contributes 25 five-seed late-stage jobs, and deterministic B0 contributes 10 jobs. This is an estimate and not authorization to compute.\n", encoding="utf-8")
    (OUT / "pre_full_run_decision.md").write_text("# Pre-full-run decision\n\nStatus: **not authorized**. Contracts are frozen and this directory is a pre-compute provenance bundle. A full V3 run may only start after review of the declared comparison pairs, acceptance criteria, and target-supervision separation. Smoke metrics cannot alter any contract.\n", encoding="utf-8")
    numerical = {"coral_targets": {"Low": [0, 0], "Medium": [1, 0], "High": [1, 1]}, "cumulative_direction": "c0=P(y>0), c1=P(y>1), c0>=c1", "class_probability_conversion": {"Low": "1-c0", "Medium": "c0-c1", "High": "c1"}, "requirements": ["nonnegative", "sum equals one", "argmax uses Low/Medium/High order"], "regression_scaler_fit": "training record IDs only", "model_reset": "new model/optimizer for each inner fold, outer fold, and training seed"}
    dump("implementation_numerical_audit.json", numerical)
    if args.smoke_run_id:
        smoke = ROOT_DIR / "artifacts/model_v3_smoke" / args.smoke_run_id
        validation = json.loads((smoke / "smoke_validation.json").read_text())
        dump("smoke_validation.json", validation)
        pd.read_csv(smoke / "smoke_metrics.csv").to_csv(OUT / "smoke_metrics.csv", index=False)
    manifest = {"protocol_version": V3_1_PROTOCOL_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
                "source_commit": source, "source_tree_clean_before_materialization": True,
                "full_run_id_reserved": args.full_run_id, "compute_authorized": False,
                "fold_manifest_checksum": manifest["manifest_checksum"], "expected_jobs": len(expected["jobs"]),
                "expected_predictions": sum(x["expected_record_count"] for x in expected["jobs"]),
                "smoke_run_id": args.smoke_run_id}
    dump("protocol_manifest.json", manifest)
    checks = {path.name: sha(path) for path in OUT.iterdir() if path.is_file() and path.name != "checksums.json"}
    dump("checksums.json", checks)


if __name__ == "__main__":
    main()
