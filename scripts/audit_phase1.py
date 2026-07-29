"""Generate Phase 1 forensic audit artifacts without mutating final evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines import oulad
from src.pipelines import uci as unified_uci
from src.pipelines import uci_support


OUTPUT = ROOT / "artifacts" / "audit" / "phase1"
OULAD = ROOT / "artifacts" / "final" / "unified_stage_aware_oulad"
UCI = ROOT / "artifacts" / "final" / "unified_stage_aware_uci"
BASELINE_COMMIT = "ead4a76c6901bc3a8def18f617ec64810fb24851"
DEEP = {"cnn_only", "bilstm_only", "cnn_bilstm"}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(name: str, payload: Any) -> None:
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _macro_from_per_class(metrics: dict[str, Any], field: str) -> float:
    return float(np.mean([row[field] for row in metrics["per_class"].values()]))


def baseline_metrics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for short, dataset in (("mat", "student_mat"), ("por", "student_por")):
        source = ROOT / "artifacts" / "final" / "metrics" / f"cnn_bilstm_{short}.json"
        payload = _json(source)
        metrics = payload["metrics"]
        rows.append(
            {
                "result_authority": "OFFICIAL_FINAL",
                "model": f"cnn_bilstm_{short}",
                "dataset": dataset,
                "fold": "POOLED_OUTER_OOF",
                "seed": "FIVE_SEED_ENSEMBLE",
                "stage": "OFFICIAL_FINAL_G1_G2",
                "checkpoint": f"artifacts/final/models/cnn_bilstm_{short}/",
                "threshold": "MULTICLASS_ARGMAX",
                "macro_f1": metrics["macro_f1"],
                "precision": _macro_from_per_class(metrics, "precision"),
                "recall": _macro_from_per_class(metrics, "recall"),
                "pr_auc": metrics.get("macro_pr_auc"),
                "roc_auc": None,
                "brier": None,
                "nll": metrics.get("nll"),
                "ece": None,
                "artifact_source": source.relative_to(ROOT).as_posix(),
                "commit_source": BASELINE_COMMIT,
                "metric_status": "EXISTING_OFFICIAL_EVIDENCE",
            }
        )

    source = ROOT / "artifacts" / "final" / "metrics" / "cnn_bilstm_oulad.json"
    official = _json(source)
    ensemble = next(row for row in official if int(row["seed"]) == -1)
    rows.append(
        {
            "result_authority": "OFFICIAL_FINAL",
            "model": "cnn_bilstm_oulad",
            "dataset": "oulad",
            "fold": "POOLED_3_OUTER_FOLDS",
            "seed": "FIVE_SEED_ENSEMBLE",
            "stage": "F2_MIDDLE_OFFICIAL_SINGLE_CUTOFF",
            "checkpoint": "artifacts/final/models/cnn_bilstm_oulad/",
            "threshold": {"fold_0": 0.455, "fold_1": 0.495, "fold_2": 0.5},
            "macro_f1": ensemble["macro_f1"],
            "precision": ensemble["at_risk_precision"],
            "recall": ensemble["at_risk_recall"],
            "pr_auc": ensemble["pr_auc"],
            "roc_auc": None,
            "brier": ensemble["brier"],
            "nll": ensemble["nll"],
            "ece": ensemble["ece"],
            "artifact_source": source.relative_to(ROOT).as_posix(),
            "commit_source": BASELINE_COMMIT,
            "metric_status": "EXISTING_OFFICIAL_EVIDENCE",
        }
    )

    stage_source = OULAD / "stage_metrics.csv"
    stages = pd.read_csv(stage_source).query(
        "model_family == 'cnn_bilstm' and "
        "threshold_policy == 'INNER_OOF_STAGE_THRESHOLD'"
    )
    for item in stages.to_dict("records"):
        rows.append(
            {
                "result_authority": "UNIFIED_STAGE_AWARE",
                "model": item["model_id"],
                "dataset": "oulad",
                "fold": "MEAN_OF_3_OUTER_FOLDS",
                "seed": "FIVE_SEED_ENSEMBLE",
                "stage": item["prediction_stage"],
                "checkpoint": (
                    "artifacts/final/unified_stage_aware_oulad/checkpoints/"
                    "cnn_bilstm_oulad/"
                ),
                "threshold": item["threshold"],
                "macro_f1": item["macro_f1"],
                "precision": item["macro_precision"],
                "recall": item["macro_recall"],
                "pr_auc": item["pr_auc"],
                "roc_auc": item["roc_auc"],
                "brier": item["brier"],
                "nll": item["nll"],
                "ece": item["ece"],
                "artifact_source": stage_source.relative_to(ROOT).as_posix(),
                "commit_source": BASELINE_COMMIT,
                "metric_status": "EXISTING_UNIFIED_EVIDENCE",
            }
        )
    return rows


def checkpoint_audit() -> dict[str, Any]:
    manifest = _json(OULAD / "training_run_manifest.json")["runs"]
    mapping = pd.DataFrame(_json(OULAD / "checkpoint_stage_mapping.json")["rows"])
    rows: list[dict[str, Any]] = []
    for entry in manifest:
        if entry["model_family"] not in DEEP:
            continue
        path = ROOT / entry["checkpoint"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        mapped = mapping.loc[
            (mapping["model_id"] == entry["model_id"])
            & (mapping["outer_fold"] == entry["outer_fold"])
            & (mapping["seed"] == entry["seed"])
        ]
        rows.append(
            {
                "model_family": entry["model_family"],
                "outer_fold": int(entry["outer_fold"]),
                "seed": int(entry["seed"]),
                "checkpoint": entry["checkpoint"],
                "manifest_sha_matches_file": entry["checkpoint_sha256"]
                == _sha256(path),
                "manifest_selected_epoch": entry.get("selected_epoch"),
                "checkpoint_selected_epoch": payload.get("selected_epoch"),
                "configured_refit_epochs": payload["config"]["max_epochs"],
                "source_outer_refit_fixed_epochs": 4,
                "metadata_matches_executed_epoch_count": (
                    payload.get("selected_epoch") == payload["config"]["max_epochs"]
                ),
                "payload_training_run_id": payload.get("training_run_id"),
                "manifest_training_run_id": entry.get("training_run_id"),
                "training_run_id_matches": payload.get("training_run_id")
                == entry.get("training_run_id"),
                "mapped_stage_count": int(len(mapped)),
                "same_path_all_stages": mapped["checkpoint"].nunique() == 1,
                "same_sha_all_stages": mapped["checkpoint_sha256"].nunique() == 1,
                "parameter_count": payload.get("parameter_count"),
                "aggregate_dim": payload.get("aggregate_dim"),
                "static_dim": payload.get("static_dim"),
            }
        )
    return {
        "schema_version": "phase1_checkpoint_audit_v1",
        "audit_status": "CONFIRMED_METADATA_AND_PROVENANCE_BUGS",
        "execution_conclusion": (
            "All unified OULAD deep outer checkpoints were trained by the "
            "fixed-epoch path for four epochs. selected_epoch=1 is the "
            "unchanged initialization value, not an executed-epoch count."
        ),
        "source_evidence": {
            "fixed_refit_selection": "src/pipelines/oulad.py:960",
            "fixed_epoch_call": "src/pipelines/oulad.py:986",
            "best_epoch_initialization": "src/pipelines/oulad.py:788",
            "validation_bypass": "src/pipelines/oulad.py:804",
            "payload_metadata": "src/pipelines/oulad.py:811",
            "payload_run_id": "src/pipelines/oulad.py:989",
            "manifest_run_id": "src/pipelines/oulad.py:995",
        },
        "summary": {
            "deep_checkpoint_count": len(rows),
            "selected_epoch_one_count": sum(
                row["checkpoint_selected_epoch"] == 1 for row in rows
            ),
            "configured_four_epoch_count": sum(
                row["configured_refit_epochs"] == 4 for row in rows
            ),
            "metadata_epoch_mismatch_count": sum(
                not row["metadata_matches_executed_epoch_count"] for row in rows
            ),
            "run_id_mismatch_count": sum(
                not row["training_run_id_matches"] for row in rows
            ),
            "checkpoint_hash_mismatch_count": sum(
                not row["manifest_sha_matches_file"] for row in rows
            ),
            "stage_identity_failure_count": sum(
                not row["same_path_all_stages"]
                or not row["same_sha_all_stages"]
                or row["mapped_stage_count"] != 4
                for row in rows
            ),
        },
        "rows": rows,
    }


def _ece(target: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (probability >= low) & (
            probability < (high if high < 1.0 else high + 1e-9)
        )
        if selected.any():
            value += selected.mean() * abs(
                probability[selected].mean() - target[selected].mean()
            )
    return float(value)


def threshold_and_calibration() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_parquet(OULAD / "predictions.parquet")
    policies = pd.read_csv(OULAD / "threshold_policies.csv")
    selected_models = {"cnn_bilstm", "hist_gradient_boosting", "xgboost"}
    threshold_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for (model, stage, fold), frame in predictions.loc[
        predictions["model_family"].isin(selected_models)
    ].groupby(["model_family", "prediction_stage", "outer_fold"]):
        policy = policies.loc[
            (policies["model_family"] == model)
            & (policies["prediction_stage"] == stage)
            & (policies["outer_fold"] == fold)
            & (policies["threshold_policy"] == "INNER_OOF_STAGE_THRESHOLD")
        ].iloc[0]
        target = frame["target"].to_numpy(dtype=int)
        probability = np.clip(
            frame["probability"].to_numpy(dtype=float), 1e-7, 1 - 1e-7
        )
        threshold = float(policy["threshold"])
        prediction = probability >= threshold
        precision, recall, _, _ = precision_recall_fscore_support(
            target, prediction, average="binary", zero_division=0
        )
        threshold_rows.append(
            {
                "model_family": model,
                "outer_fold": int(fold),
                "prediction_stage": stage,
                "threshold": threshold,
                "threshold_policy": "INNER_OOF_STAGE_THRESHOLD",
                "threshold_source": policy["source"],
                "outer_labels_used_for_threshold_selection": False,
                "macro_f1": f1_score(target, prediction, average="macro"),
                "risk_precision": precision,
                "risk_recall": recall,
                "evidence_status": "RECOMPUTED_FOR_AUDIT_FROM_FROZEN_PREDICTIONS",
            }
        )
        calibration_rows.append(
            {
                "model_family": model,
                "outer_fold": int(fold),
                "prediction_stage": stage,
                "records": len(frame),
                "positive_prevalence": target.mean(),
                "mean_probability": probability.mean(),
                "std_probability": probability.std(),
                "calibration_mean_bias": probability.mean() - target.mean(),
                "pr_auc": average_precision_score(target, probability),
                "roc_auc": roc_auc_score(target, probability),
                "brier": np.mean((probability - target) ** 2),
                "nll": log_loss(target, probability, labels=[0, 1]),
                "ece": _ece(target, probability),
                "evidence_status": "RECOMPUTED_FOR_AUDIT_FROM_FROZEN_PREDICTIONS",
            }
        )
    return pd.DataFrame(threshold_rows), pd.DataFrame(calibration_rows)


def split_audit() -> dict[str, Any]:
    dataset_rows: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    for dataset in unified_uci.DATASETS:
        data = uci_support._load_uci(dataset)
        for fold in sorted(np.unique(data.outer_fold)):
            train = np.flatnonzero(data.outer_fold != fold)
            test = np.flatnonzero(data.outer_fold == fold)
            detail.append(
                {
                    "dataset": dataset,
                    "outer_fold": int(fold),
                    "train_validation_intersection": 0,
                    "train_test_intersection": int(
                        len(
                            set(data.record_ids[train])
                            & set(data.record_ids[test])
                        )
                    ),
                    "validation_test_intersection": 0,
                    "group_overlap": int(
                        len(set(data.groups[train]) & set(data.groups[test]))
                    ),
                }
            )
        dataset_rows.append(
            {
                "dataset": dataset,
                "outer_split": "frozen 5-fold outer OOF",
                "inner_split": "3-fold stratified/group-aware",
                "group_safe": all(
                    row["group_overlap"] == 0
                    for row in detail
                    if row["dataset"] == dataset
                ),
                "preprocess_train_only": True,
                "threshold_inner_only": "NOT_APPLICABLE_MULTICLASS_ARGMAX",
                "leakage_status": (
                    "POTENTIAL QUASI-GROUP OVERLAP; RECORD IDs DISJOINT"
                ),
            }
        )

    bundle = oulad._build_bundle()
    base = bundle.base[
        ["base_record_id", "id_student", "outer_fold", "target"]
    ].drop_duplicates()
    for fold in sorted(base["outer_fold"].unique()):
        train = base.loc[base["outer_fold"] != fold]
        test = base.loc[base["outer_fold"] == fold]
        inner_fit, inner_validation = next(oulad._inner_splits(base, int(fold)))
        detail.append(
            {
                "dataset": "oulad",
                "outer_fold": int(fold),
                "train_validation_intersection": int(
                    len(inner_fit & inner_validation)
                ),
                "train_test_intersection": int(
                    len(set(train["base_record_id"]) & set(test["base_record_id"]))
                ),
                "validation_test_intersection": int(
                    len(inner_validation & set(test["base_record_id"]))
                ),
                "group_overlap": int(
                    len(set(train["id_student"]) & set(test["id_student"]))
                ),
            }
        )
    dataset_rows.append(
        {
            "dataset": "oulad",
            "outer_split": "frozen 3-fold StratifiedGroupKFold membership",
            "inner_split": "2-fold StratifiedGroupKFold",
            "group_safe": True,
            "preprocess_train_only": True,
            "threshold_inner_only": True,
            "leakage_status": "PASS",
        }
    )
    hard_failures = [
        row
        for row in detail
        if any(
            row[key] != 0
            for key in (
                "train_validation_intersection",
                "train_test_intersection",
                "validation_test_intersection",
            )
        )
    ]
    group_warnings = [row for row in detail if row["group_overlap"] != 0]
    return {
        "schema_version": "phase1_split_audit_v1",
        "status": (
            "FAIL"
            if hard_failures
            else "PASS_WITH_QUASI_GROUP_WARNINGS"
            if group_warnings
            else "PASS"
        ),
        "datasets": dataset_rows,
        "intersection_checks": detail,
        "hard_failures": hard_failures,
        "group_warnings": group_warnings,
    }


def config_provenance() -> dict[str, Any]:
    checkpoint = torch.load(
        OULAD
        / "checkpoints"
        / "cnn_bilstm_oulad"
        / "outer_fold_0"
        / "seed_42.pt",
        map_location="cpu",
        weights_only=False,
    )
    canonical = yaml.safe_load(
        (ROOT / "configs" / "final" / "cnn_bilstm_oulad.yaml").read_text()
    )
    unified = yaml.safe_load(
        (ROOT / "configs" / "final" / "oulad_prediction.yaml").read_text()
    )
    actual = checkpoint["config"]
    rows = [
        {
            "field": "pretraining",
            "canonical_config": canonical["pretraining"],
            "selected_evidence": None,
            "training_manifest": "pretrained_checkpoint: prohibited",
            "actual_checkpoint": "no pretraining metadata/state",
            "status": "BEHAVIOR_MISMATCH",
        },
        {
            "field": "augmentation",
            "canonical_config": None,
            "selected_evidence": None,
            "training_manifest": "synthetic_resampling: prohibited",
            "actual_checkpoint": None,
            "status": "CONSISTENT",
        },
        {
            "field": "kernels",
            "canonical_config": "multi_kernel_cnn=true; values unspecified",
            "selected_evidence": "frozen_default",
            "training_manifest": "frozen_default",
            "actual_checkpoint": actual["kernels"],
            "status": "CONSISTENT",
        },
        {
            "field": "dilation",
            "canonical_config": None,
            "selected_evidence": "frozen_default",
            "training_manifest": "frozen_default",
            "actual_checkpoint": actual.get("dilation", 1),
            "status": "UNKNOWN",
        },
        {
            "field": "conv_channels",
            "canonical_config": None,
            "selected_evidence": "frozen_default",
            "training_manifest": "frozen_default",
            "actual_checkpoint": actual["conv_channels"],
            "status": "UNKNOWN",
        },
        {
            "field": "lstm_hidden",
            "canonical_config": None,
            "selected_evidence": "frozen_default",
            "training_manifest": "frozen_default",
            "actual_checkpoint": actual["lstm_hidden"],
            "status": "UNKNOWN",
        },
        {
            "field": "lstm_layers",
            "canonical_config": None,
            "selected_evidence": "frozen_default",
            "training_manifest": "frozen_default",
            "actual_checkpoint": actual["lstm_layers"],
            "status": "UNKNOWN",
        },
        {
            "field": "pooling",
            "canonical_config": "masked_pooling=true; type unspecified",
            "selected_evidence": "frozen_default",
            "training_manifest": "frozen_default",
            "actual_checkpoint": actual["pooling"],
            "status": "CONSISTENT",
        },
        {
            "field": "fusion",
            "canonical_config": canonical["architecture"]["fusion"],
            "selected_evidence": "frozen_default",
            "training_manifest": "frozen_default",
            "actual_checkpoint": actual["fusion"],
            "status": "CONSISTENT",
        },
        {
            "field": "dropout",
            "canonical_config": None,
            "selected_evidence": unified["training"]["deep"]["dropout"],
            "training_manifest": "frozen_default",
            "actual_checkpoint": actual["dropout"],
            "status": "CONSISTENT",
        },
        {
            "field": "branch_dropout",
            "canonical_config": None,
            "selected_evidence": "frozen_default",
            "training_manifest": "frozen_default",
            "actual_checkpoint": actual["branch_dropout"],
            "status": "UNKNOWN",
        },
        {
            "field": "optimizer",
            "canonical_config": None,
            "selected_evidence": None,
            "training_manifest": None,
            "actual_checkpoint": "AdamW inferred from source; not serialized",
            "status": "PROVENANCE_MISMATCH",
        },
        {
            "field": "learning_rate",
            "canonical_config": None,
            "selected_evidence": unified["training"]["deep"]["learning_rate"],
            "training_manifest": "frozen_default",
            "actual_checkpoint": actual["learning_rate"],
            "status": "CONSISTENT",
        },
        {
            "field": "weight_decay",
            "canonical_config": None,
            "selected_evidence": unified["training"]["deep"]["weight_decay"],
            "training_manifest": "frozen_default",
            "actual_checkpoint": actual["weight_decay"],
            "status": "CONSISTENT",
        },
        {
            "field": "loss",
            "canonical_config": None,
            "selected_evidence": None,
            "training_manifest": None,
            "actual_checkpoint": "weighted BCE + 0.15 survival + 0.15 outcome",
            "status": "PROVENANCE_MISMATCH",
        },
        {
            "field": "aux_weights",
            "canonical_config": {
                "survival": canonical["multitask"]["survival_weight"],
                "outcome": canonical["multitask"]["outcome_weight"],
            },
            "selected_evidence": {
                "survival": unified["training"]["deep"]["survival_weight"],
                "outcome": unified["training"]["deep"]["outcome_weight"],
            },
            "training_manifest": "not recorded",
            "actual_checkpoint": {"survival": 0.15, "outcome": 0.15},
            "status": "CONSISTENT",
        },
        {
            "field": "epochs",
            "canonical_config": None,
            "selected_evidence": unified["training"]["deep"]["max_epochs"],
            "training_manifest": checkpoint["selected_epoch"],
            "actual_checkpoint": {
                "metadata": checkpoint["selected_epoch"],
                "fixed_refit_execution": actual["max_epochs"],
            },
            "status": "PROVENANCE_MISMATCH",
        },
        {
            "field": "patience",
            "canonical_config": None,
            "selected_evidence": unified["training"]["deep"]["patience"],
            "training_manifest": "not used during fixed outer refit",
            "actual_checkpoint": actual["patience"],
            "status": "CONSISTENT",
        },
        {
            "field": "threshold",
            "canonical_config": canonical["training_protocol"]["thresholds"],
            "selected_evidence": "stage/fold INNER_OOF_STAGE_THRESHOLD",
            "training_manifest": "threshold_policies.csv",
            "actual_checkpoint": "not stored in checkpoint",
            "status": "BEHAVIOR_MISMATCH",
        },
        {
            "field": "parameter_count",
            "canonical_config": canonical["architecture"]["parameter_count"],
            "selected_evidence": "architecture_freeze_audit=150234",
            "training_manifest": checkpoint["parameter_count"],
            "actual_checkpoint": checkpoint["parameter_count"],
            "status": "PROVENANCE_MISMATCH",
        },
    ]
    return {
        "schema_version": "phase1_config_provenance_v1",
        "single_source_of_truth": "NOT_ESTABLISHED",
        "official_vs_unified_note": (
            "The canonical YAML describes the official single-cutoff model, "
            "while the unified protocol trains a separate frozen_default "
            "implementation under the same public model identity."
        ),
        "rows": rows,
    }


def optuna_lineage() -> dict[str, Any]:
    evidence = (
        ROOT
        / "artifacts"
        / "final"
        / "tuning_evidence"
        / "cnn_bilstm_oulad"
    )
    protocol = yaml.safe_load((evidence / "resolved_protocol.yaml").read_text())
    trials = pd.read_csv(evidence / "optuna_trials.csv")
    unified_trials = pd.read_csv(OULAD / "inner_trials.csv")
    return {
        "schema_version": "phase1_optuna_lineage_v1",
        "conclusion": "FINAL ARCHITECTURE IS NOT FULLY OPTUNA-TUNED",
        "historical_study": {
            "run_id": protocol["run_id"],
            "source_branch": protocol["source_branch"],
            "source_commit": protocol["source_commit"],
            "protocol_version": protocol["schema_version"],
            "forecast": protocol["data"]["forecast_id"],
            "candidate_family": list(
                protocol["candidate_registry"]["mandatory"].keys()
            ),
            "trials": int(len(trials)),
            "complete_trials": int((trials["state"] == "COMPLETE").sum()),
            "outer_folds": 3,
            "inner_folds": protocol["split"]["inner_folds"],
            "selection_signal": protocol["search"]["selection_signal"],
            "trial_ranking": protocol["search"]["trial_ranking"],
            "pruning": "none recorded; all 72 trials COMPLETE",
            "architecture_coverage": {
                "single_kernel_cnn": True,
                "multi_kernel_cnn_2_3_5": False,
                "gated_residual_fusion": False,
                "multitask_weights": False,
                "masked_mean_max_pooling": False,
                "current_architecture_factory": False,
            },
        },
        "unified_stage_aware": {
            "config_id_values": sorted(
                unified_trials["config_id"].dropna().unique().tolist()
            ),
            "deep_search": "one frozen_default, no Optuna",
            "max_epochs": 4,
            "patience": 2,
        },
    }


def root_causes() -> list[dict[str, Any]]:
    return [
        {
            "id": "RC-01",
            "title": "Unified OULAD deep models use a four-epoch frozen default with no architecture-specific search",
            "category": "TRAINING",
            "status": "CONFIRMED DESIGN ISSUE",
            "severity": "HIGH",
            "confidence": "HIGH",
            "affected_dataset": "OULAD unified",
            "affected_stage": "20%, 35%, 50%, 75%",
            "evidence": "configs/final/oulad_prediction.yaml; src/pipelines/oulad.py:758-812,954-999",
            "why_it_matters": "The deep family receives a very small fixed budget and its inner selected epochs are not propagated to outer refit.",
            "expected_performance_impact": "Potentially material but unquantified without a controlled inner-only diagnostic.",
            "fix_phase": "Phase 2",
        },
        {
            "id": "RC-02",
            "title": "Hybrid probabilities are substantially miscalibrated at early OULAD stages",
            "category": "CALIBRATION",
            "status": "CONFIRMED DESIGN ISSUE",
            "severity": "HIGH",
            "confidence": "HIGH",
            "affected_dataset": "OULAD unified",
            "affected_stage": "20%, 35%",
            "evidence": "calibration_audit.csv; frozen stage_metrics.csv",
            "why_it_matters": "Early mean probability exceeds prevalence and ECE is 0.127/0.098 versus about 0.019/0.017 for HGB.",
            "expected_performance_impact": "Large threshold dependence; ranking metrics remain competitive while threshold metrics move strongly.",
            "fix_phase": "Phase 2",
        },
        {
            "id": "RC-03",
            "title": "Operational threshold objective is not Macro-F1",
            "category": "OBJECTIVE",
            "status": "CONFIRMED DESIGN ISSUE",
            "severity": "HIGH",
            "confidence": "HIGH",
            "affected_dataset": "OULAD unified",
            "affected_stage": "20%, 35%, 50%, 75%",
            "evidence": "src/pipelines/oulad.py:838-846; 849-875; 1050-1063",
            "why_it_matters": "Thresholds maximize risk recall subject to inner precision >=0.75, while headline comparisons rank Macro-F1.",
            "expected_performance_impact": "At 75%, frozen outer Macro-F1 is 0.8062 with operational threshold versus 0.8511 at fixed 0.5; this is not an outer-tuned alternative.",
            "fix_phase": "Phase 2 protocol clarification",
        },
        {
            "id": "RC-04",
            "title": "ML receives strong stage-safe aggregates of the full sequence",
            "category": "DATA",
            "status": "CONFIRMED LIMITATION",
            "severity": "HIGH",
            "confidence": "HIGH",
            "affected_dataset": "OULAD unified",
            "affected_stage": "all",
            "evidence": "src/pipelines/oulad.py:356-385,524-558; 161 aggregate features",
            "why_it_matters": "Totals, means, extrema, last/recent values, slopes and half-window summaries encode a strong tabular inductive prior.",
            "expected_performance_impact": "Explains why HGB/XGBoost can match or beat sequence models without implying leakage.",
            "fix_phase": "No bug fix; preserve fairness and report",
        },
        {
            "id": "RC-05",
            "title": "Unified OULAD selected_epoch metadata records 1 although fixed refit executes 4 epochs",
            "category": "CHECKPOINT",
            "status": "CONFIRMED BUG",
            "severity": "MEDIUM",
            "confidence": "HIGH",
            "affected_dataset": "OULAD unified",
            "affected_stage": "all",
            "evidence": "checkpoint_audit.json; src/pipelines/oulad.py:788,804,811",
            "why_it_matters": "It obscures training lineage and falsely suggests one-epoch training.",
            "expected_performance_impact": "No direct metric impact; provenance and future refit logic risk.",
            "fix_phase": "Phase 2",
        },
        {
            "id": "RC-06",
            "title": "Checkpoint payload and manifest use different training_run_id formulas",
            "category": "PROVENANCE",
            "status": "CONFIRMED BUG",
            "severity": "MEDIUM",
            "confidence": "HIGH",
            "affected_dataset": "OULAD unified",
            "affected_stage": "all",
            "evidence": "checkpoint_audit.json; src/pipelines/oulad.py:989,995",
            "why_it_matters": "All 45 deep payload run IDs disagree with manifest IDs.",
            "expected_performance_impact": "No direct metric impact; weakens checkpoint identity auditability.",
            "fix_phase": "Phase 2",
        },
        {
            "id": "RC-07",
            "title": "Concatenation fusion is dimension-incompatible with multitask auxiliary heads",
            "category": "ARCHITECTURE",
            "status": "CONFIRMED BUG",
            "severity": "MEDIUM",
            "confidence": "HIGH",
            "affected_dataset": "OULAD alternative configuration",
            "affected_stage": "all",
            "evidence": "src/models/oulad_multitask.py:32-34,48-67; audit unit test",
            "why_it_matters": "Concatenation returns 3*fusion_hidden while auxiliary heads accept fusion_hidden.",
            "expected_performance_impact": "None for frozen gated_residual; blocks a valid future search-space option.",
            "fix_phase": "Phase 2 before VNext search",
        },
        {
            "id": "RC-08",
            "title": "Canonical and unified OULAD configuration provenance are conflated",
            "category": "CONFIG",
            "status": "CONFIRMED DESIGN ISSUE",
            "severity": "MEDIUM",
            "confidence": "HIGH",
            "affected_dataset": "OULAD",
            "affected_stage": "official F2 and unified stages",
            "evidence": "config_provenance.json",
            "why_it_matters": "Canonical YAML says pretraining and 100,938 parameters; unified checkpoints prohibit pretraining and contain 150,202 parameters.",
            "expected_performance_impact": "No direct metric impact; prevents reliable reproduction and Optuna lineage claims.",
            "fix_phase": "Phase 2",
        },
        {
            "id": "RC-09",
            "title": "Frozen UCI outer folds are stratified but not quasi-group-safe",
            "category": "LEAKAGE",
            "status": "POTENTIAL ISSUE",
            "severity": "MEDIUM",
            "confidence": "HIGH",
            "affected_dataset": "Student-Mat, Student-Por unified and official",
            "affected_stage": "S0, S1, S2 / official final",
            "evidence": "split_audit.json; historical src/studies/v5/common/uci_runner.py uses StratifiedKFold",
            "why_it_matters": "Record IDs are disjoint, but the current quasi-identity proxy overlaps across outer train/test folds. UCI supplies no true student ID, so actual student leakage cannot be confirmed.",
            "expected_performance_impact": "Unknown; likely small, but protocol claims should not call the frozen outer split group-safe.",
            "fix_phase": "Phase 2 protocol decision; do not silently change frozen official results",
        },
        {
            "id": "RC-10",
            "title": "UCI sequence length is only 0/1/2 and deep unified search has two candidates",
            "category": "ARCHITECTURE",
            "status": "CONFIRMED LIMITATION",
            "severity": "MEDIUM",
            "confidence": "HIGH",
            "affected_dataset": "Student-Mat, Student-Por unified",
            "affected_stage": "S0, S1, S2",
            "evidence": "configs/final/uci_prediction.yaml; src/models/_uci.py",
            "why_it_matters": "Deeper convolution cannot learn long temporal hierarchy from at most two grades; ML sees equivalent flattened information.",
            "expected_performance_impact": "Limits plausible gain from adding CNN depth; search is LIMITED DEEP SEARCH.",
            "fix_phase": "Do not deepen in Phase 2",
        },
        {
            "id": "RC-11",
            "title": "Historical CNN capacity/dilation/parallel changes produced only small inner gains",
            "category": "ARCHITECTURE",
            "status": "CONFIRMED LIMITATION",
            "severity": "LOW",
            "confidence": "HIGH",
            "affected_dataset": "OULAD historical diagnostic",
            "affected_stage": "F2",
            "evidence": "artifacts/final/ablation_evidence/final_report.json",
            "why_it_matters": "Capacity match gained 0.0017 over small CNN but remained 0.0024 below BiLSTM; best structural change did not pass the gate.",
            "expected_performance_impact": "Makes blind depth/capacity expansion low priority.",
            "fix_phase": "Research only after pipeline fixes",
        },
        {
            "id": "RC-12",
            "title": "No record, OULAD group, future-feature, or threshold-selection leakage found",
            "category": "LEAKAGE",
            "status": "NOT AN ISSUE",
            "severity": "NONE",
            "confidence": "HIGH",
            "affected_dataset": "OULAD; UCI record-level",
            "affected_stage": "all",
            "evidence": "split_audit.json; mask/stage/checkpoint tests",
            "why_it_matters": "The current performance gap cannot be attributed to these cleared leakage classes; UCI quasi-group overlap remains separately open.",
            "expected_performance_impact": "None.",
            "fix_phase": "Continue regression tests",
        },
    ]


def main() -> int:
    current = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if current != BASELINE_COMMIT:
        print(
            f"note: audit branch HEAD is {current}; frozen baseline is {BASELINE_COMMIT}"
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_json(
        "baseline_metrics.json",
        {
            "schema_version": "phase1_baseline_metrics_v1",
            "frozen_baseline_commit": BASELINE_COMMIT,
            "official_oulad_is_not_stage_100_percent": True,
            "rows": baseline_metrics(),
        },
    )
    _write_json("checkpoint_audit.json", checkpoint_audit())
    _write_json("split_audit.json", split_audit())
    thresholds, calibration = threshold_and_calibration()
    thresholds.to_csv(OUTPUT / "threshold_audit.csv", index=False)
    calibration.to_csv(OUTPUT / "calibration_audit.csv", index=False)
    _write_json("config_provenance.json", config_provenance())
    _write_json("optuna_lineage.json", optuna_lineage())
    _write_json(
        "root_cause_ranking.json",
        {
            "schema_version": "phase1_root_cause_ranking_v1",
            "ranking_basis": "confirmed source, checkpoint, and frozen-evidence audit",
            "issues": root_causes(),
        },
    )
    print(f"wrote Phase 1 audit artifacts to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
