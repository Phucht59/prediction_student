"""Execute only the approved Strategy B Phase A-B development protocol.

This runner never fetches the 79 legacy observed records.  It materializes
split ledgers, reproduces one historical-compatible fixed control, compares
the replayable B1/B2 training policies and emits immutable evidence bundles.
It does not run Optuna or instantiate any Phase C candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.config import DATASETS, DEFAULT_SEED, ROOT_DIR
from src.data_pipeline import SOURCE_ROW_NUMBER_COLUMN, process_target_and_stratify
from src.estimator_factory import (
    RESOLVED_CONFIG_SCHEMA_VERSION,
    resolve_student_config,
    resolved_config_hash,
    resolved_config_schema,
    validate_resolved_config,
    with_training_policy,
)
from src.evaluation.protocol import (
    DEFAULT_FOLD_MANIFEST_PATH,
    file_checksum,
    load_fold_manifest,
    outer_folds_from_manifest,
    semantic_checksum,
    source_record_identity,
)
from src.model_selection import fit_fold_predict_proba, predict_with_fitted_estimator
from src.postgres_data_source import load_development_subset_from_postgres
from src.strategy_b_phase_ab import (
    APPROVED_SEEDS,
    PHASE_AB_PROTOCOL_VERSION,
    approved_candidate_registry,
    assert_development_only_frame,
    development_source_rows,
    evidence_quarantine_registry,
    materialize_early_stop_ledger,
    materialize_inner_fold_ledger,
    recompute_metrics_from_oof,
    sha256_file,
    source_rows_hash,
    validate_oof_coverage,
    write_json,
)


ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "strategy_b_phase_ab"
REPORT_ROOT = ROOT_DIR / "reports" / "strategy_b_phase_ab"
REPRODUCTION_TOLERANCE = 1e-7
POLICY_IDS = ["A0_historical_compatible", "B1_drop_last_true", "B2_drop_last_false"]
MINIMUM_OUTPUTS = [
    "protocol.json",
    "resolved_config_schema.json",
    "evidence_quarantine_registry.json",
    "dataset_manifest.json",
    "outer_fold_manifest.json",
    "inner_fold_ledger.csv",
    "early_stop_ledger.csv",
    "control_fold_seed_metrics.csv",
    "training_policy_comparison.csv",
    "sample_utilization.csv",
    "training_diagnostics.csv",
    "checkpoint_checksums.json",
    "source_provenance.json",
    "test_report.json",
    "strict_validation.json",
    "phase_ab_conclusion.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="student-mat", choices=["student-mat"])
    parser.add_argument("--dataset-version-id", type=int, default=1)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST_PATH)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ROOT_DIR / "artifacts" / "model_selection" / "nested-full-20260710" / "selected_config.json",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-tests", action="store_true", help="Developer-only; official Phase A-B must run tests.")
    return parser.parse_args()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT_DIR,
        check=check,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )


def _git_text(*args: str) -> str:
    return _run(["git", *args]).stdout.strip()


def _source_provenance() -> dict[str, Any]:
    commit = _git_text("rev-parse", "HEAD")
    branch = _git_text("branch", "--show-current")
    diff = _run(["git", "diff", "--binary", "HEAD"]).stdout.encode("utf-8")
    status = _run(["git", "status", "--porcelain=v1", "--untracked-files=all"]).stdout
    tracked = _git_text("ls-files").splitlines()
    source_paths = [
        path for path in tracked
        if path.startswith(("src/", "scripts/", "tests/", "config/", "database/"))
        or path in {"requirements.txt", "requirements-lock.txt", "environment.yml", "SCIENTIFIC_PROTOCOL_V2.md"}
    ]
    source_hashes = {
        path: sha256_file(ROOT_DIR / path)
        for path in sorted(source_paths)
        if (ROOT_DIR / path).is_file()
    }
    source_tree_hash = _sha256_bytes(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    pip_freeze = _run([sys.executable, "-m", "pip", "freeze"]).stdout.splitlines()
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "pip_freeze": sorted(pip_freeze),
        "lock_files": {
            path: sha256_file(ROOT_DIR / path)
            for path in ("requirements.txt", "requirements-lock.txt", "environment.yml")
            if (ROOT_DIR / path).exists()
        },
    }
    environment_hash = _sha256_bytes(
        json.dumps(environment, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return {
        "git_commit": commit,
        "git_branch": branch,
        "dirty_diff_hash": _sha256_bytes(diff),
        "dirty_diff_bytes": len(diff),
        "git_status_porcelain": status.splitlines(),
        "git_status_hash": _sha256_bytes(status.encode("utf-8")),
        "source_file_hashes": source_hashes,
        "source_tree_hash": source_tree_hash,
        "environment": environment,
        "environment_hash": environment_hash,
    }


def _base_flat_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = dict(payload.get("best_params", payload))
    required_historical = {
        "learning_rate", "weight_decay", "batch_size", "cnn_channels",
        "cnn_kernel_size", "lstm_hidden_dim", "dropout", "sequence_dropout",
        "max_epochs", "patience",
    }
    missing = sorted(required_historical - set(values))
    if missing:
        raise ValueError(f"Frozen base config is missing fields: {missing}")
    values.setdefault("architecture_variant", "cnn_bilstm")
    values.setdefault("oversample_method", "none")
    values.setdefault("class_weight_mode", "none")
    values.setdefault("loss", "cross_entropy")
    values.setdefault("smote_ratio", 1.0)
    values.setdefault("resampling_k_neighbors", 5)
    return values


def _policy_configs(base_path: Path) -> dict[str, dict[str, Any]]:
    flat = _base_flat_config(base_path)
    control = resolve_student_config(
        flat,
        architecture_variant="cnn_bilstm",
        suggested_parameters={},
        scheduler_type="legacy_reduce_on_plateau",
        swa_enabled=True,
        drop_last_train=True,
        evidence_role="phase_a_historical_compatible_control",
    )
    b1 = with_training_policy(
        control,
        scheduler_type="fixed_lr",
        swa_enabled=False,
        drop_last_train=True,
        evidence_role="phase_b_replayable_fixed_lr_drop_last_true",
    )
    b2 = with_training_policy(
        control,
        scheduler_type="fixed_lr",
        swa_enabled=False,
        drop_last_train=False,
        evidence_role="phase_b_replayable_fixed_lr_drop_last_false",
    )
    configs = {
        "A0_historical_compatible": control,
        "B1_drop_last_true": b1,
        "B2_drop_last_false": b2,
    }
    for config in configs.values():
        validate_resolved_config(config)
    return configs


def _save_checkpoint_and_preprocessor(
    output: Path,
    *,
    policy_id: str,
    fold: int,
    seed: int,
    result: Any,
    validation_frame: pd.DataFrame,
    spec: Any,
) -> tuple[dict[str, Any], float]:
    stem = f"outer{fold}_seed{seed}"
    checkpoint_path = output / "checkpoints" / policy_id / f"{stem}.pt"
    preprocessor_path = output / "preprocessors" / policy_id / f"{stem}.pkl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.refit_state_dict,
            "policy_id": policy_id,
            "outer_fold": int(fold),
            "seed": int(seed),
            "resolved_config_hash": resolved_config_hash(result.resolved_config),
        },
        checkpoint_path,
    )
    with preprocessor_path.open("wb") as handle:
        pickle.dump(
            {
                "preprocessor": result.refit_preprocessor,
                "selector": result.refit_selector,
                "resolved_config_hash": resolved_config_hash(result.resolved_config),
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    with preprocessor_path.open("rb") as handle:
        preprocessing = pickle.load(handle)
    reproduced = predict_with_fitted_estimator(
        frame=validation_frame,
        spec=spec,
        resolved_config=result.resolved_config,
        state_dict=checkpoint["state_dict"],
        preprocessor=preprocessing["preprocessor"],
        selector=preprocessing["selector"],
    )
    max_difference = float(np.max(np.abs(reproduced - result.probabilities)))
    return {
        "policy_id": policy_id,
        "outer_fold": int(fold),
        "seed": int(seed),
        "checkpoint": checkpoint_path.relative_to(output).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "preprocessor": preprocessor_path.relative_to(output).as_posix(),
        "preprocessor_sha256": sha256_file(preprocessor_path),
        "resolved_config_hash": resolved_config_hash(result.resolved_config),
        "prediction_reproduction_max_abs_difference": max_difference,
        "prediction_reproduction_tolerance": REPRODUCTION_TOLERANCE,
        "prediction_reproduction_pass": max_difference <= REPRODUCTION_TOLERANCE,
    }, max_difference


def _run_tests(skip_tests: bool) -> dict[str, Any]:
    if skip_tests:
        return {"command": None, "return_code": None, "status": "SKIPPED", "official": False}
    command = [sys.executable, "-m", "pytest", "-q"]
    started = time.perf_counter()
    completed = _run(command, check=False)
    return {
        "command": command,
        "return_code": int(completed.returncode),
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "official": True,
        "duration_seconds": time.perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _comparison_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    b1 = metrics[metrics["policy_id"] == "B1_drop_last_true"].copy()
    b2 = metrics[metrics["policy_id"] == "B2_drop_last_false"].copy()
    keys = ["seed", "outer_fold"]
    merged = b1.merge(b2, on=keys, suffixes=("_b1", "_b2"), validate="one_to_one")
    return pd.DataFrame({
        "seed": merged["seed"].astype(int),
        "outer_fold": merged["outer_fold"].astype(int),
        "records": merged["records_b1"].astype(int),
        "b1_macro_f1": merged["macro_f1_b1"],
        "b2_macro_f1": merged["macro_f1_b2"],
        "b2_minus_b1_macro_f1": merged["macro_f1_b2"] - merged["macro_f1_b1"],
        "b1_accuracy": merged["accuracy_b1"],
        "b2_accuracy": merged["accuracy_b2"],
        "b2_minus_b1_accuracy": merged["accuracy_b2"] - merged["accuracy_b1"],
    }).sort_values(keys).reset_index(drop=True)


def _conclusion_markdown(
    *,
    run_id: str,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    strict: dict[str, Any],
    tests: dict[str, Any],
    checkpoint_summary: dict[str, Any],
) -> str:
    summaries = metrics.groupby("policy_id").agg(
        mean_macro_f1=("macro_f1", "mean"),
        std_macro_f1=("macro_f1", "std"),
        mean_accuracy=("accuracy", "mean"),
    )
    table_rows = []
    for policy_id in POLICY_IDS:
        row = summaries.loc[policy_id]
        table_rows.append(
            f"| {policy_id} | {row.mean_macro_f1:.6f} | {row.std_macro_f1:.6f} | {row.mean_accuracy:.6f} |"
        )
    delta = comparison["b2_minus_b1_macro_f1"]
    wins = int((delta > 1e-12).sum())
    ties = int((delta.abs() <= 1e-12).sum())
    losses = int((delta < -1e-12).sum())
    remaining = [
        "Phase C candidate comparison chưa chạy; không có best_overall_model hoặc best_thesis_hybrid_model.",
        "Ordinal, residual và multitask candidates chưa được triển khai.",
        "Recommendation Level 0.5 chưa được sửa trong Phase A-B.",
        "Không có unseen external confirmation dataset; 79 records vẫn bị quarantine.",
        "Phase C vẫn cần phê duyệt riêng dù technical gate có thể PASS.",
    ]
    return "\n".join([
        f"# Kết luận Strategy B Phase A-B — `{run_id}`",
        "",
        "## 1. Correctness đã sửa",
        "",
        "- Canonical resolved configuration giữ cả suggested parameters và fixed constants.",
        "- Missing required key fail-fast; không còn silent balanced-class-weight fallback.",
        "- Inner, outer và final refit dùng chung `StudentEstimatorFactory` và một training-partition factory.",
        "- B1/B2 dùng fixed learning rate, không SWA; epoch-selection và full refit replay cùng policy.",
        "- Final estimator path refit trên toàn development frame.",
        "- PostgreSQL loader của selection chỉ truy vấn 316 development source rows.",
        "- Checkpoints và preprocessors được hash và tái tạo exact OOF probabilities.",
        "",
        "## 2. Control reproduction và B1/B2",
        "",
        "| Policy | Mean fold-seed Macro-F1 | SD | Mean accuracy |",
        "|---|---:|---:|---:|",
        *table_rows,
        "",
        f"Paired B2−B1 Macro-F1 mean: **{delta.mean():.6f}**; wins/ties/losses: **{wins}/{ties}/{losses}** trên 15 fold-seed pairs.",
        "",
        "A0 là historical-compatible diagnostic control và cố ý giữ scheduler/SWA mismatch để đo chênh lệch; không đủ điều kiện làm estimator chính thức. B1/B2 là corrected estimators.",
        "",
        "## 3. Validation",
        "",
        f"- Full test suite: **{tests['status']}** (`return_code={tests['return_code']}`).",
        f"- Estimator parity B1/B2: **{'PASS' if strict['estimator_parity_pass'] else 'FAIL'}**.",
        f"- Exact checkpoint reproduction: **{'PASS' if checkpoint_summary['all_reproduced'] else 'FAIL'}**, max abs diff `{checkpoint_summary['max_abs_difference']:.3g}`.",
        f"- Strict validation: **{strict['status']}**.",
        "- 79 observed payload accessed: **NO**.",
        "",
        "## 4. Vấn đề còn lại",
        "",
        *[f"- {item}" for item in remaining],
        "",
        "## 5. Phase C gate",
        "",
        f"Technical prerequisites: **{'ĐẠT' if strict['phase_c_technical_gate_pass'] else 'CHƯA ĐẠT'}**.",
        "",
        "Phase C **không được tự động mở**. Cần phê duyệt rõ của người dùng sau khi xem bundle này.",
        "",
    ])


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    run_id = args.run_id or f"phase-ab-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    artifact_final = ARTIFACT_ROOT / run_id
    report_final = REPORT_ROOT / run_id
    if artifact_final.exists() or report_final.exists():
        raise FileExistsError(f"Immutable run already exists: {run_id}")
    artifact_tmp = ARTIFACT_ROOT / f".{run_id}.tmp"
    report_tmp = REPORT_ROOT / f".{run_id}.tmp"
    if artifact_tmp.exists() or report_tmp.exists():
        raise FileExistsError(f"Temporary run path already exists: {run_id}")
    artifact_tmp.mkdir(parents=True)
    report_tmp.mkdir(parents=True)
    try:
        created_at = datetime.now(timezone.utc).isoformat()
        spec = DATASETS[args.dataset]
        manifest = load_fold_manifest(args.fold_manifest)
        if int(manifest["dataset_version_id"]) != int(args.dataset_version_id):
            raise ValueError("Dataset version must match the immutable outer-fold manifest.")
        allowed_rows = development_source_rows(manifest)
        raw_development, dataset_metadata = load_development_subset_from_postgres(
            args.dataset,
            args.dataset_version_id,
            allowed_rows,
        )
        development = process_target_and_stratify(
            raw_development.copy(), spec.target_col, spec.kind, "3class"
        ).dropna(subset=["_strat_target"]).drop(columns=["_strat_target"])
        development = development.sort_values(SOURCE_ROW_NUMBER_COLUMN).reset_index(drop=True)
        assert_development_only_frame(development, manifest)
        if dataset_metadata["content_hash"] != manifest["dataset_checksum"]:
            raise ValueError("Dataset hash differs from the immutable outer-fold manifest.")
        outer_folds = outer_folds_from_manifest(
            development, manifest, source_column=SOURCE_ROW_NUMBER_COLUMN
        )
        if len(outer_folds) != 5:
            raise ValueError("Official Phase A-B requires exactly five immutable outer folds.")

        provenance = _source_provenance()
        policies = _policy_configs(args.base_config)
        for policy_id, config in policies.items():
            config_path = artifact_tmp / "resolved_configs" / f"{policy_id}.json"
            write_json(config_path, config)
        write_json(artifact_tmp / "resolved_configs.json", policies)
        write_json(artifact_tmp / "resolved_config_schema.json", resolved_config_schema())
        write_json(artifact_tmp / "candidate_registry.json", approved_candidate_registry())
        write_json(artifact_tmp / "evidence_quarantine_registry.json", evidence_quarantine_registry())
        write_json(artifact_tmp / "source_provenance.json", provenance)

        dataset_manifest = {
            "dataset_code": args.dataset,
            "dataset_version_id": int(args.dataset_version_id),
            "dataset_hash": dataset_metadata["content_hash"],
            "hash_algorithm": dataset_metadata["hash_algorithm"],
            "dataset_row_count": int(dataset_metadata["dataset_row_count"]),
            "development_row_count": int(len(development)),
            "development_source_rows_hash": source_rows_hash(allowed_rows),
            "development_source_rows": allowed_rows,
            "loaded_source_rows_hash": source_rows_hash(dataset_metadata["loaded_source_row_numbers"]),
            "observed_79_fetched": False,
            "target_contract_hash": dataset_metadata["target_contract_hash"],
            "ingestion_contract_hash": dataset_metadata["ingestion_contract_hash"],
            "transaction_read_only": bool(dataset_metadata["transaction_read_only"]),
        }
        write_json(artifact_tmp / "dataset_manifest.json", dataset_manifest)
        shutil.copy2(args.fold_manifest, artifact_tmp / "outer_fold_manifest.json")
        inner_ledger = materialize_inner_fold_ledger(
            development,
            outer_folds,
            dataset_version_id=args.dataset_version_id,
            target_col=spec.target_col,
            inner_folds=3,
            seed=DEFAULT_SEED,
        )
        early_ledger = materialize_early_stop_ledger(
            development,
            outer_folds,
            dataset_version_id=args.dataset_version_id,
            target_col=spec.target_col,
            seeds=APPROVED_SEEDS,
        )
        inner_ledger.to_csv(artifact_tmp / "inner_fold_ledger.csv", index=False)
        early_ledger.to_csv(artifact_tmp / "early_stop_ledger.csv", index=False)

        protocol = {
            "protocol_version": PHASE_AB_PROTOCOL_VERSION,
            "run_id": run_id,
            "created_at": created_at,
            "scope": ["Phase A", "Phase B"],
            "phase_c_or_later_executed": False,
            "optuna_executed": False,
            "ordinal_model_executed": False,
            "recommendation_modified": False,
            "legacy_observed_79_fetched": False,
            "legacy_observed_79_prohibited_uses": [
                "selection", "architecture", "hyperparameters", "calibration",
                "threshold_tuning", "final_confirmation",
            ],
            "dataset_hash": dataset_metadata["content_hash"],
            "target_contract_hash": dataset_metadata["target_contract_hash"],
            "outer_fold_manifest_file_hash": file_checksum(args.fold_manifest),
            "outer_fold_manifest_semantic_hash": semantic_checksum(manifest),
            "resolved_config_schema_version": RESOLVED_CONFIG_SCHEMA_VERSION,
            "git_commit": provenance["git_commit"],
            "dirty_diff_hash": provenance["dirty_diff_hash"],
            "environment_hash": provenance["environment_hash"],
            "source_tree_hash": provenance["source_tree_hash"],
            "features": ["G1", "G2"],
            "class_weight_policy": "none",
            "oversampling_policy": "none",
            "seeds": APPROVED_SEEDS,
            "outer_folds": 5,
            "policies": {
                policy_id: {
                    "resolved_config_hash": resolved_config_hash(config),
                    "scheduler": config["scheduler"],
                    "swa": config["swa"],
                    "drop_last_train": config["drop_last_train"],
                    "evidence_role": config["evidence_role"],
                }
                for policy_id, config in policies.items()
            },
            "budget": {
                "phase_a_model_fits": 30,
                "phase_b_model_fits": 60,
                "total_model_fits": 90,
                "derivation": "3 policies × 5 outer folds × 3 seeds × 2 stages",
            },
            "reporting_contract": approved_candidate_registry()["reporting_contract"],
        }
        write_json(artifact_tmp / "protocol.json", protocol)

        oof_rows: list[dict[str, Any]] = []
        diagnostic_rows: list[dict[str, Any]] = []
        utilization_rows: list[dict[str, Any]] = []
        checkpoint_rows: list[dict[str, Any]] = []
        for policy_id in POLICY_IDS:
            config = policies[policy_id]
            for outer_fold, (train_idx, validation_idx) in enumerate(outer_folds):
                train_fold = development.iloc[train_idx].copy()
                validation_fold = development.iloc[validation_idx].copy()
                for seed in APPROVED_SEEDS:
                    result = fit_fold_predict_proba(
                        train_fold=train_fold,
                        validation_fold=validation_fold,
                        spec=spec,
                        params=config,
                        seed=int(seed),
                        fold_index=int(outer_fold),
                    )
                    checkpoint_row, _ = _save_checkpoint_and_preprocessor(
                        artifact_tmp,
                        policy_id=policy_id,
                        fold=outer_fold,
                        seed=seed,
                        result=result,
                        validation_frame=validation_fold,
                        spec=spec,
                    )
                    checkpoint_rows.append(checkpoint_row)
                    source_values = validation_fold[SOURCE_ROW_NUMBER_COLUMN].astype(int).to_numpy()
                    for index, source_row in enumerate(source_values):
                        oof_rows.append({
                            "policy_id": policy_id,
                            "seed": int(seed),
                            "outer_fold": int(outer_fold),
                            "dataset_version_id": int(args.dataset_version_id),
                            "source_record_identity": source_record_identity(args.dataset_version_id, int(source_row)),
                            "source_row_number": int(source_row),
                            "true_label": int(result.true_labels[index]),
                            "predicted_label": int(result.predictions[index]),
                            "probability_low": float(result.probabilities[index, 0]),
                            "probability_medium": float(result.probabilities[index, 1]),
                            "probability_high": float(result.probabilities[index, 2]),
                        })
                    diagnostics = dict(result.training_diagnostics or {})
                    diagnostic_rows.append({
                        "policy_id": policy_id,
                        "outer_fold": int(outer_fold),
                        "seed": int(seed),
                        "epochs_ran": diagnostics["epochs_ran"],
                        "selected_epoch": diagnostics["selected_epoch"],
                        "refit_epochs": diagnostics["refit_epochs"],
                        "max_epochs": diagnostics["max_epochs"],
                        "hit_epoch_cap": diagnostics["hit_epoch_cap"],
                        "best_internal_validation_macro_f1": diagnostics["best_internal_validation_macro_f1"],
                        "estimator_parity": diagnostics["estimator_parity"],
                        "criterion_parity": diagnostics["criterion_parity"],
                        "resampling_parity": diagnostics["resampling_parity"],
                        "scheduler_state_selection": json.dumps(diagnostics["scheduler_state_selection"], sort_keys=True),
                        "scheduler_state_refit": json.dumps(diagnostics["scheduler_state_refit"], sort_keys=True),
                        "swa_state_selection": json.dumps(diagnostics["swa_state_selection"], sort_keys=True),
                        "swa_state_refit": json.dumps(diagnostics["swa_state_refit"], sort_keys=True),
                        "resolved_config_hash": resolved_config_hash(config),
                    })
                    for stage_key, stage_name in (
                        ("sample_utilization_selection", "epoch_selection"),
                        ("sample_utilization_refit", "full_partition_refit"),
                    ):
                        stats = diagnostics[stage_key]
                        utilization_rows.append({
                            "policy_id": policy_id,
                            "outer_fold": int(outer_fold),
                            "seed": int(seed),
                            "stage": stage_name,
                            **stats,
                            "utilization_rate": float(stats["samples_consumed_per_epoch"] / stats["dataset_size"]),
                        })

        oof = pd.DataFrame(oof_rows).sort_values(
            ["policy_id", "seed", "outer_fold", "source_row_number"]
        ).reset_index(drop=True)
        validate_oof_coverage(
            oof,
            development_rows=allowed_rows,
            policy_ids=POLICY_IDS,
            seeds=APPROVED_SEEDS,
        )
        oof.to_csv(artifact_tmp / "oof_predictions.csv", index=False)
        metrics = recompute_metrics_from_oof(oof)
        metrics.to_csv(artifact_tmp / "all_policy_fold_seed_metrics.csv", index=False)
        metrics[metrics["policy_id"] == "A0_historical_compatible"].to_csv(
            artifact_tmp / "control_fold_seed_metrics.csv", index=False
        )
        comparison = _comparison_frame(metrics)
        comparison.to_csv(artifact_tmp / "training_policy_comparison.csv", index=False)
        utilization = pd.DataFrame(utilization_rows)
        utilization.to_csv(artifact_tmp / "sample_utilization.csv", index=False)
        diagnostics_frame = pd.DataFrame(diagnostic_rows)
        diagnostics_frame.to_csv(artifact_tmp / "training_diagnostics.csv", index=False)
        checkpoint_summary = {
            "reproduction_tolerance": REPRODUCTION_TOLERANCE,
            "checkpoint_count": len(checkpoint_rows),
            "all_reproduced": all(row["prediction_reproduction_pass"] for row in checkpoint_rows),
            "max_abs_difference": max(row["prediction_reproduction_max_abs_difference"] for row in checkpoint_rows),
            "entries": checkpoint_rows,
        }
        write_json(artifact_tmp / "checkpoint_checksums.json", checkpoint_summary)

        tests = _run_tests(args.skip_tests)
        write_json(artifact_tmp / "test_report.json", tests)
        recomputed = recompute_metrics_from_oof(pd.read_csv(artifact_tmp / "oof_predictions.csv"))
        metric_columns = ["macro_f1", "accuracy"]
        metric_recomputation_difference = float(
            np.max(np.abs(recomputed[metric_columns].to_numpy() - metrics[metric_columns].to_numpy()))
        )
        b_diagnostics = diagnostics_frame[diagnostics_frame["policy_id"].isin(["B1_drop_last_true", "B2_drop_last_false"])]
        b2_refit = utilization[
            (utilization["policy_id"] == "B2_drop_last_false")
            & (utilization["stage"] == "full_partition_refit")
        ]
        checks = [
            {"id": "resolved_config_constants", "pass": all(bool(config["fixed_constants"]) for config in policies.values())},
            {"id": "development_only_db_access", "pass": dataset_manifest["observed_79_fetched"] is False and dataset_manifest["development_row_count"] == 316},
            {"id": "outer_fold_manifest", "pass": len(outer_folds) == 5 and manifest["manifest_checksum"] == semantic_checksum(manifest)},
            {"id": "inner_fold_ledger", "pass": not inner_ledger.empty and set(inner_ledger["role"]) == {"inner_train", "inner_validation"}},
            {"id": "early_stop_ledger", "pass": not early_ledger.empty and set(early_ledger["role"]) == {"model_train", "early_stop", "outer_validation"}},
            {"id": "no_target_model_input", "pass": all(config["feature_contract"]["sequence_columns"] == ["G1", "G2"] for config in policies.values())},
            {"id": "criterion_parity", "pass": bool(b_diagnostics["criterion_parity"].all())},
            {"id": "resampling_parity", "pass": bool(b_diagnostics["resampling_parity"].all())},
            {"id": "estimator_factory_and_training_parity", "pass": bool(b_diagnostics["estimator_parity"].all())},
            {"id": "scheduler_refit_replayable", "pass": all(config["scheduler"]["replayable"] for key, config in policies.items() if key.startswith("B"))},
            {"id": "swa_disabled_phase_b", "pass": all(not config["swa"]["enabled"] for key, config in policies.items() if key.startswith("B"))},
            {"id": "drop_last_false_no_records_dropped", "pass": bool((b2_refit["samples_dropped_per_epoch"] == 0).all())},
            {"id": "checkpoint_reproduction", "pass": bool(checkpoint_summary["all_reproduced"])},
            {"id": "exact_source_provenance", "pass": bool(provenance["git_commit"] and provenance["source_tree_hash"] and provenance["environment_hash"])},
            {"id": "metrics_recomputed_from_saved_oof", "pass": metric_recomputation_difference <= 1e-12},
            {"id": "full_test_suite", "pass": tests["status"] == "PASS"},
        ]
        status = "PASS" if all(check["pass"] for check in checks) else "FAIL"
        strict = {
            "run_id": run_id,
            "status": status,
            "checks": checks,
            "estimator_parity_pass": bool(b_diagnostics["estimator_parity"].all()),
            "checkpoint_reproduction_pass": bool(checkpoint_summary["all_reproduced"]),
            "metric_recomputation_max_abs_difference": metric_recomputation_difference,
            "phase_c_technical_gate_pass": status == "PASS",
            "phase_c_authorized": False,
            "authorization_note": "Explicit user approval is still required; this runner never launches Phase C.",
        }
        write_json(artifact_tmp / "strict_validation.json", strict)
        conclusion = _conclusion_markdown(
            run_id=run_id,
            metrics=metrics,
            comparison=comparison,
            strict=strict,
            tests=tests,
            checkpoint_summary=checkpoint_summary,
        )
        (artifact_tmp / "phase_ab_conclusion.md").write_text(conclusion, encoding="utf-8")

        missing_outputs = [name for name in MINIMUM_OUTPUTS if not (artifact_tmp / name).exists()]
        if missing_outputs:
            raise RuntimeError(f"Phase A-B bundle is missing required outputs: {missing_outputs}")
        if status != "PASS":
            raise RuntimeError("Strict validation failed; temporary evidence retained only until cleanup.")

        artifact_checksums = {
            path.relative_to(artifact_tmp).as_posix(): sha256_file(path)
            for path in sorted(artifact_tmp.rglob("*"))
            if path.is_file() and path.name != "artifact_checksums.json"
        }
        write_json(artifact_tmp / "artifact_checksums.json", artifact_checksums)
        report_files = [
            "protocol.json", "resolved_config_schema.json", "evidence_quarantine_registry.json",
            "dataset_manifest.json", "outer_fold_manifest.json", "inner_fold_ledger.csv",
            "early_stop_ledger.csv", "control_fold_seed_metrics.csv",
            "training_policy_comparison.csv", "sample_utilization.csv",
            "training_diagnostics.csv", "checkpoint_checksums.json", "source_provenance.json",
            "test_report.json", "strict_validation.json", "phase_ab_conclusion.md",
            "candidate_registry.json", "artifact_checksums.json",
        ]
        for name in report_files:
            shutil.copy2(artifact_tmp / name, report_tmp / name)
        write_json(report_tmp / "artifact_index.json", {
            "run_id": run_id,
            "artifact_path": str(artifact_final),
            "report_path": str(report_final),
            "minimum_outputs": MINIMUM_OUTPUTS,
            "checkpoint_directory": str(artifact_final / "checkpoints"),
            "preprocessor_directory": str(artifact_final / "preprocessors"),
        })
        artifact_tmp.replace(artifact_final)
        report_tmp.replace(report_final)
        return artifact_final, report_final
    except Exception:
        if artifact_tmp.exists():
            shutil.rmtree(artifact_tmp)
        if report_tmp.exists():
            shutil.rmtree(report_tmp)
        raise


if __name__ == "__main__":
    artifact_path, report_path = run(parse_args())
    print(json.dumps({"artifact_path": str(artifact_path), "report_path": str(report_path)}, ensure_ascii=False))
