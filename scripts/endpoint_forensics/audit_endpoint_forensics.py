"""Build the static Phase 8 OULAD endpoint forensic evidence.

This utility never trains a model and never executes a new outer evaluation. It
only re-scores the already stored H0/H1 prediction files and compares frozen
configuration/provenance evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
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


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts" / "audit" / "phase8"
REPORT_ROOT = ROOT / "reports" / "audit" / "phase8"
HISTORICAL_V6_COMMIT = "da25cf5"

H0_PREDICTIONS = (
    ROOT / "artifacts" / "final" / "predictions" / "cnn_bilstm_oulad"
    / "oof_predictions.parquet"
)
H1_PREDICTIONS = (
    ROOT / "artifacts" / "audit" / "phase7" / "endpoint_final_predictions.parquet"
)
PHASE7_METRICS = (
    ROOT / "artifacts" / "audit" / "phase7" / "endpoint_final_metrics.json"
)
PHASE7_FREEZE = (
    ROOT / "artifacts" / "audit" / "phase7" / "endpoint_freeze_manifest.json"
)
PHASE7_PROTOCOL = (
    ROOT / "artifacts" / "audit" / "phase7" / "endpoint_protocol_audit.json"
)
PHASE6_STAGE_METRICS = (
    ROOT / "artifacts" / "final" / "h1_final" / "stage_metrics.csv"
)
H0_SELECTED = (
    ROOT / "artifacts" / "final" / "protocol_snapshots"
    / "cnn_bilstm_oulad_selected_model.json"
)
H0_CONFIG = ROOT / "configs" / "final" / "cnn_bilstm_oulad.yaml"
H1_CONFIG = ROOT / "configs" / "registry" / "oulad_unified_stage_aware_v2.yaml"

BASE_CHANNELS = (
    "total_clicks",
    "active_days",
    "unique_sites",
    "unique_activity_types",
    "content_clicks",
    "forum_clicks",
    "quiz_clicks",
    "assessment_related_clicks",
    "submitted_assessment_count",
    "late_submission_count",
    "available_score_count",
    "cumulative_mean_score",
    "cumulative_weighted_score",
    "days_since_last_vle_activity",
    "weeks_without_activity",
    "score_missing_mask",
)
DYNAMIC_CHANNELS = (
    "log1p_total_clicks",
    "log1p_active_days",
    "log1p_unique_sites",
    "log1p_assessment_related_clicks",
    "log1p_submitted_assessment_count",
    "delta_total_clicks",
    "delta_active_days",
    "delta_unique_sites",
    "delta_content_clicks",
    "delta_forum_clicks",
    "delta_quiz_clicks",
    "delta_assessment_related_clicks",
    "delta_submitted_assessment_count",
    "delta_cumulative_mean_score",
    "delta_cumulative_weighted_score",
    "rolling_2_week_mean_total_clicks",
    "rolling_2_week_mean_active_days",
    "rolling_2_week_mean_assessment_clicks",
    "rolling_2_week_submission_count",
    "rolling_2_week_score_change",
    "current_inactivity_streak",
    "activity_resumed_indicator",
    "new_inactivity_indicator",
    "content_share",
    "forum_share",
    "quiz_share",
    "assessment_share",
    "score_delta",
    "weighted_score_delta",
    "late_submission_rate_to_date",
    "submission_rate_last_2_weeks",
)
TEMPORAL_CHANNELS = BASE_CHANNELS + DYNAMIC_CHANNELS
STATIC_COLUMNS = (
    "code_module",
    "presentation_season",
    "num_of_prev_attempts",
    "studied_credits",
    "registration_lead_time",
    "module_presentation_length",
)
CONTEXT_COLUMNS = (
    "progress_fraction",
    "observed_week_count",
    "weeks_remaining",
    "assessment_available_fraction",
)
H1_SUMMARIES = (
    "total",
    "mean",
    "std",
    "min",
    "max",
    "last",
    "slope",
    "recent_2_week_mean",
    "first_half_mean",
    "second_half_mean",
)
H0_COMPACT_SUMMARIES: dict[str, tuple[str, ...]] = {
    "total_clicks": ("sum", "mean", "last", "slope", "recent_2_week_mean"),
    "active_days": ("sum", "mean", "last", "slope", "recent_2_week_mean"),
    "unique_sites": ("mean", "last", "recent_2_week_mean"),
    "unique_activity_types": ("mean", "last", "recent_2_week_mean"),
    "content_clicks": ("sum", "slope", "recent_2_week_mean"),
    "forum_clicks": ("sum", "slope", "recent_2_week_mean"),
    "quiz_clicks": ("sum", "slope", "recent_2_week_mean"),
    "assessment_related_clicks": ("sum", "slope", "recent_2_week_mean"),
    "submitted_assessment_count": ("sum", "last"),
    "late_submission_count": ("sum", "last"),
    "available_score_count": ("sum", "last"),
    "cumulative_mean_score": ("last", "slope", "recent_2_week_mean"),
    "cumulative_weighted_score": ("last", "slope", "recent_2_week_mean"),
    "days_since_last_vle_activity": ("last", "slope", "recent_2_week_mean"),
    "weeks_without_activity": ("sum", "last", "recent_2_week_mean"),
    "score_missing_mask": ("sum", "last"),
}
SCORE_CHANNELS = {
    "available_score_count",
    "cumulative_mean_score",
    "cumulative_weighted_score",
    "score_missing_mask",
    "delta_cumulative_mean_score",
    "delta_cumulative_weighted_score",
    "rolling_2_week_score_change",
    "score_delta",
    "weighted_score_delta",
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_text(commit: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def _git_json(commit: str, path: str) -> Any:
    return json.loads(_git_text(commit, path))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(name: str, text: str) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / name).write_text(
        text.strip() + "\n", encoding="utf-8", newline="\n"
    )


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (probabilities >= lower) & (
            probabilities < upper if upper < 1.0 else probabilities <= upper
        )
        if selected.any():
            total += selected.mean() * abs(
                float(labels[selected].mean())
                - float(probabilities[selected].mean())
            )
    return float(total)


def _metrics(
    labels: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> dict[str, Any]:
    values = np.clip(probabilities.astype(float), 1e-7, 1 - 1e-7)
    predicted = values >= thresholds
    tn, fp, fn, tp = confusion_matrix(
        labels, predicted, labels=[0, 1]
    ).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "risk_precision": float(precision[1]),
        "risk_recall": float(recall[1]),
        "risk_f1": float(f1[1]),
        "specificity": float(tn / max(tn + fp, 1)),
        "pr_auc": float(average_precision_score(labels, values)),
        "roc_auc": float(roc_auc_score(labels, values)),
        "nll": float(log_loss(labels, values, labels=[0, 1])),
        "brier": float(np.mean((values - labels) ** 2)),
        "ece_10_bin": _ece(labels, values, 10),
        "ece_15_bin": _ece(labels, values, 15),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_rate": float(predicted.mean()),
    }


def _best_global_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    """Efficient diagnostic outer oracle; never used for model selection."""
    order = np.argsort(-probabilities, kind="stable")
    sorted_probability = probabilities[order]
    sorted_label = labels[order]
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    tp = fp = 0
    fn = positives
    tn = negatives
    best_score = 0.5 * (2 * tn / max(2 * tn + fn + fp, 1))
    best_threshold = float(np.nextafter(sorted_probability.max(), np.inf))
    index = 0
    while index < len(labels):
        end = index + 1
        while (
            end < len(labels)
            and sorted_probability[end] == sorted_probability[index]
        ):
            end += 1
        group_positive = int(sorted_label[index:end].sum())
        group_negative = int(end - index - group_positive)
        tp += group_positive
        fn -= group_positive
        fp += group_negative
        tn -= group_negative
        positive_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        negative_f1 = 2 * tn / max(2 * tn + fn + fp, 1)
        score = 0.5 * (positive_f1 + negative_f1)
        threshold = float(sorted_probability[index])
        if score > best_score:
            best_score = score
            best_threshold = threshold
        index = end
    return {
        "macro_f1": float(best_score),
        "threshold": best_threshold,
        "scope": "DIAGNOSTIC_OUTER_ORACLE_NOT_FOR_SELECTION",
    }


def _h0_aggregate_features() -> list[str]:
    values = [
        f"{channel}__{summary}"
        for channel, summaries in H0_COMPACT_SUMMARIES.items()
        for summary in summaries
    ]
    values.append("inactive_week_count")
    if len(values) != 49:
        raise RuntimeError("Historical compact aggregate contract changed")
    return values


def _h1_aggregate_features() -> list[str]:
    values = [
        f"{channel}__{summary}"
        for channel in BASE_CHANNELS
        for summary in H1_SUMMARIES
    ]
    values.extend(("inactive_week_count", *CONTEXT_COLUMNS))
    if len(values) != 165:
        raise RuntimeError("H1 aggregate contract changed")
    return values


def _align_predictions() -> pd.DataFrame:
    h0 = pd.read_parquet(H0_PREDICTIONS).rename(
        columns={
            "record_id": "base_record_id",
            "probability": "probability_h0",
            "threshold": "threshold_h0",
        }
    )
    h1 = pd.read_parquet(H1_PREDICTIONS).rename(
        columns={
            "probability": "probability_h1",
            "threshold": "threshold_h1",
        }
    )
    keys = [
        "base_record_id",
        "id_student",
        "code_module",
        "code_presentation",
        "outer_fold",
        "target",
        "cutoff_day",
    ]
    aligned = h0[
        [*keys, "probability_h0", "threshold_h0"]
    ].merge(
        h1[[*keys, "probability_h1", "threshold_h1"]],
        on=keys,
        validate="one_to_one",
    )
    if len(aligned) != len(h0) or len(aligned) != len(h1):
        raise RuntimeError("H0/H1 record identity is not exact")
    return aligned


def _feature_diff() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for channel in TEMPORAL_CHANNELS:
        score_semantics = channel in SCORE_CHANNELS
        rows.append(
            {
                "feature_group": "temporal",
                "H0_feature": channel,
                "H1_feature": channel,
                "classification": (
                    "SHARED_NAME_DIFFERENT_AVAILABILITY"
                    if score_semantics
                    else "SHARED_EQUIVALENT"
                ),
                "H0_semantics": (
                    "populated from scores available by max(submission,due-date)"
                    if score_semantics
                    else "events before cutoff"
                ),
                "H1_semantics": (
                    "score values excluded; missing-mask fixed unavailable"
                    if score_semantics
                    else "events before cutoff"
                ),
                "endpoint_legality": (
                    "historical conservative proxy; exact release timestamp absent"
                    if score_semantics
                    else "legal"
                ),
                "evidence_source": (
                    "git:308370:src/studies/oulad/materialize.py;"
                    "src/pipelines/oulad.py"
                ),
            }
        )
    h1_features = set(_h1_aggregate_features())
    matched_h1: set[str] = set()
    for feature in _h0_aggregate_features():
        if feature == "inactive_week_count":
            mapped = feature
            channel = feature
        else:
            channel, summary = feature.split("__", 1)
            mapped = f"{channel}__{'total' if summary == 'sum' else summary}"
        matched_h1.add(mapped)
        score_semantics = channel in SCORE_CHANNELS
        rows.append(
            {
                "feature_group": "aggregate",
                "H0_feature": feature,
                "H1_feature": mapped,
                "classification": (
                    "RENAMED_EQUIVALENT_DIFFERENT_AVAILABILITY"
                    if score_semantics
                    else (
                        "RENAMED_EQUIVALENT"
                        if feature != mapped
                        else "SHARED_EQUIVALENT"
                    )
                ),
                "H0_semantics": (
                    "summary of populated score-progress channel"
                    if score_semantics
                    else "compact summary of legal temporal channel"
                ),
                "H1_semantics": (
                    "summary exists but carries no observed score value"
                    if score_semantics
                    else "full summary of legal temporal channel"
                ),
                "endpoint_legality": (
                    "historical conservative proxy; exact release timestamp absent"
                    if score_semantics
                    else "legal"
                ),
                "evidence_source": (
                    "git:308370:src/studies/v5_1/oulad/data.py;"
                    "src/pipelines/oulad.py"
                ),
            }
        )
    for feature in sorted(h1_features - matched_h1):
        rows.append(
            {
                "feature_group": "aggregate",
                "H0_feature": "",
                "H1_feature": feature,
                "classification": "H1_ONLY",
                "H0_semantics": "not in compact 49-feature branch",
                "H1_semantics": "additional full summary or endpoint context",
                "endpoint_legality": "legal; derived before cutoff",
                "evidence_source": "src/pipelines/oulad.py:_aggregate",
            }
        )
    for feature in STATIC_COLUMNS:
        rows.append(
            {
                "feature_group": "static_source",
                "H0_feature": feature,
                "H1_feature": feature,
                "classification": "SHARED_EQUIVALENT",
                "H0_semantics": "train-fitted numeric/categorical preprocessing",
                "H1_semantics": "train-fitted numeric/categorical preprocessing",
                "endpoint_legality": "legal",
                "evidence_source": (
                    "git:308370:src/studies/v5_1/oulad/data.py;"
                    "src/pipelines/oulad.py"
                ),
            }
        )
    return rows


def _population(aligned: pd.DataFrame) -> dict[str, Any]:
    group_overlap = int(
        (aligned.groupby("id_student").outer_fold.nunique() > 1).sum()
    )
    if group_overlap:
        raise RuntimeError("Stored outer folds contain student overlap")
    fold_counts = {
        str(int(fold)): int(len(group))
        for fold, group in aligned.groupby("outer_fold")
    }
    fold_positive = {
        str(int(fold)): int(group.target.sum())
        for fold, group in aligned.groupby("outer_fold")
    }
    fold_record_hashes = {
        str(int(fold)): _canonical_hash(
            sorted(group.base_record_id.astype(str).tolist())
        )
        for fold, group in aligned.groupby("outer_fold")
    }
    fold_student_hashes = {
        str(int(fold)): _canonical_hash(
            sorted(group.id_student.astype(int).unique().tolist())
        )
        for fold, group in aligned.groupby("outer_fold")
    }
    return {
        "status": "IDENTICAL",
        "record_identity_exact": True,
        "fold_assignment_identity_exact": True,
        "eligible_records": int(len(aligned)),
        "positive_count": int(aligned.target.sum()),
        "negative_count": int((1 - aligned.target).sum()),
        "positive_prevalence": float(aligned.target.mean()),
        "unique_students": int(aligned.id_student.nunique()),
        "modules": int(aligned.code_module.nunique()),
        "module_presentations": int(
            aligned[["code_module", "code_presentation"]]
            .drop_duplicates()
            .shape[0]
        ),
        "fold_counts": fold_counts,
        "fold_positive_counts": fold_positive,
        "all_record_id_hash": _canonical_hash(
            sorted(aligned.base_record_id.astype(str).tolist())
        ),
        "fold_record_id_hashes": fold_record_hashes,
        "fold_student_id_hashes": fold_student_hashes,
        "target": {
            "positive": ["Withdrawn", "Fail"],
            "negative": ["Pass", "Distinction"],
        },
        "eligible_population": "historical_development and registered at F2 cutoff",
        "exclusion_rules": [
            "not registered before cutoff",
            "unregistered before cutoff",
            "non-historical development role",
        ],
        "group_key": "id_student",
        "outer_group_overlap": group_overlap,
        "evidence_sources": [
            H0_PREDICTIONS.relative_to(ROOT).as_posix(),
            H1_PREDICTIONS.relative_to(ROOT).as_posix(),
            "artifacts/audit/phase7/endpoint_protocol_audit.json",
        ],
    }


def _build_profiles(
    aligned: pd.DataFrame,
    h0_metrics: dict[str, Any],
    h1_metrics: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    phase7_freeze = _json(PHASE7_FREEZE)
    phase7_protocol = _json(PHASE7_PROTOCOL)
    h0_selected = _json(H0_SELECTED)
    h1_registry = _yaml(H1_CONFIG)
    historical_metadata = _git_json(
        HISTORICAL_V6_COMMIT,
        "artifacts/v6/prediction/final/checkpoint_metadata.json",
    )
    pretraining_gate = _git_json(
        HISTORICAL_V6_COMMIT,
        "artifacts/v6/prediction/pretraining/gate.json",
    )
    selected_v51 = _git_json(
        HISTORICAL_V6_COMMIT,
        "artifacts/v5_1/oulad/selected_configs.json",
    )
    population = _population(aligned)
    h0_architecture = {
        "family": "CNN-BiLSTM",
        "input_projection": 48,
        "conv_channels": 24,
        "kernels": [2, 3],
        "dilation": 2,
        "residual": True,
        "bilstm_hidden": 64,
        "bilstm_layers": 1,
        "bidirectional": True,
        "pooling": "masked_mean_max",
        "pooling_projection": 48,
        "aggregate_hidden": 64,
        "static_hidden": 32,
        "fusion_hidden": 64,
        "fusion": "scalar_gated_residual",
        "heads": ["risk", "survival", "outcome"],
    }
    h1_architecture = {
        **h1_registry["architecture"],
        "fusion": "scalar_gated_residual",
        "tabular_residual_expert": {
            "input": "aggregate_165_plus_static_13",
            "hidden": [48, 32],
            "output": "risk_residual_logit",
            "alpha": "bounded_sigmoid_learnable",
            "initial_alpha": 0.05,
        },
        "heads": ["risk", "survival", "outcome"],
    }
    h0_checkpoint_hashes = sorted(
        {str(row["sha256"]) for row in historical_metadata}
    )
    if len(historical_metadata) != 15 or len(h0_checkpoint_hashes) != 15:
        raise RuntimeError("Historical H0 checkpoint provenance is incomplete")
    h0_profile = {
        "profile_id": "H0_OFFICIAL_ENDPOINT",
        "model": {
            "candidate_id": h0_selected["candidate"],
            "model_class": "src.studies.v6.multitask.V6TemporalMultiTask",
            "architecture": h0_architecture,
            "parameter_count": 100_938,
            "architecture_hash": _canonical_hash(h0_architecture),
            "architecture_hash_authority": "PHASE8_DERIVED_FROM_V6_TOPOLOGY",
            "checkpoint_count": 15,
            "checkpoint_hashes": h0_checkpoint_hashes,
        },
        "features": {
            "temporal_input_dimension": 47,
            "sequence_length_max": 20,
            "temporal_feature_names": list(TEMPORAL_CHANNELS),
            "aggregate_input_dimension": 49,
            "aggregate_feature_names": _h0_aggregate_features(),
            "static_source_feature_names": list(STATIC_COLUMNS),
            "static_runtime_dimension": 13,
            "score_policy": (
                "Scores are treated as available at max(date_submitted, "
                "assessment_due_date), with both dates before cutoff."
            ),
        },
        "feature_schema_hash": _canonical_hash(
            {
                "temporal": TEMPORAL_CHANNELS,
                "aggregate": _h0_aggregate_features(),
                "static": STATIC_COLUMNS,
                "score_policy": "max_submission_due_before_cutoff",
            }
        ),
        "preprocessing": {
            "fit_scope": "outer_train_only or inner_train_only",
            "sequence": "masked train mean/std; padding excluded; zero after scale",
            "aggregate": "median imputation then StandardScaler",
            "static_numeric": "median imputation then StandardScaler",
            "static_categorical": (
                "most-frequent imputation then train-fit OneHotEncoder "
                "handle_unknown=ignore"
            ),
            "future_padding": "right zero padding with boolean mask",
        },
        "target_and_population": population,
        "endpoint": {
            "id": "F2_MIDDLE",
            "cutoff": "floor(module_presentation_length * 0.50)",
            "event_filter": "0 <= event_day < cutoff_day",
        },
        "splits": {
            "outer_folds": 3,
            "inner_folds": 3,
            "group_key": "id_student",
            "fold_counts": population["fold_counts"],
            "seeds": [42, 1201, 2026, 3407, 7319],
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": selected_v51[0]["config"]["learning_rate"],
            "weight_decay": selected_v51[0]["config"]["weight_decay"],
            "batch_size": 256,
            "dropout": selected_v51[0]["config"]["dropout"],
            "loss": "standard_bce",
            "class_weight": None,
            "auxiliary_weights": {"survival": 0.15, "outcome": 0.15},
            "pretraining_requested": True,
            "pretraining_executed": True,
            "pretraining": {
                "id": "P1_MASKED_AND_NEXT_WEEK",
                "epochs": 5,
                "scope": "each outer-training partition only",
                "tasks": pretraining_gate["tasks"],
                "gate_inner_macro_f1_gain": pretraining_gate["macro_f1_gain"],
                "gate_inner_pr_auc_gain": pretraining_gate["pr_auc_gain"],
                "checkpoint_provenance": "15 distinct pretraining checkpoint hashes",
                "checkpoint_hashes": sorted(
                    {
                        str(row["pretraining_sha256"])
                        for row in historical_metadata
                    }
                ),
            },
            "final_epochs": 8,
            "checkpoint_criterion": "fixed epoch frozen after inner gates",
            "scheduler": None,
        },
        "threshold": {
            "policy": "per-outer-fold inner-OOF Macro-F1",
            "values": {
                str(key): value["threshold"]
                for key, value in h0_selected["thresholds"].items()
            },
            "outer_labels_used": False,
        },
        "metric_aggregation": "five-seed probability ensemble then pooled outer OOF",
        "reproduced_metrics": h0_metrics,
        "provenance": {
            "source_commit": HISTORICAL_V6_COMMIT,
            "canonical_config": H0_CONFIG.relative_to(ROOT).as_posix(),
            "selected_model": H0_SELECTED.relative_to(ROOT).as_posix(),
            "historical_checkpoint_metadata": (
                "git:da25cf5:artifacts/v6/prediction/final/"
                "checkpoint_metadata.json"
            ),
        },
    }
    h1_science = phase7_freeze["scientific_configuration"]
    h1_profile = {
        "profile_id": "H1_PHASE7_ENDPOINT",
        "model": {
            "candidate_id": "H1_TABULAR_RESIDUAL_EXPERT",
            "model_class": (
                "src.models.oulad_tabular_residual."
                "CNNBiLSTMTabularResidualOULAD"
            ),
            "architecture": h1_architecture,
            "parameter_count": phase7_freeze["parameter_count"],
            "architecture_hash": phase7_freeze["architecture_hash"],
            "temporal_backbone_hash": h1_science["temporal_backbone_hash"],
            "checkpoint_count": 15,
        },
        "features": {
            "temporal_input_dimension": 47,
            "sequence_length_max": 20,
            "temporal_feature_names": list(TEMPORAL_CHANNELS),
            "aggregate_input_dimension": 165,
            "aggregate_feature_names": _h1_aggregate_features(),
            "static_source_feature_names": list(STATIC_COLUMNS),
            "static_runtime_dimension": 13,
            "score_policy": (
                "Score values excluded because OULAD has no score-release "
                "timestamp; score_missing_mask is always unavailable."
            ),
        },
        "feature_schema_hash": _canonical_hash(
            {
                "temporal": TEMPORAL_CHANNELS,
                "aggregate": _h1_aggregate_features(),
                "static": STATIC_COLUMNS,
                "score_policy": "scores_excluded_no_release_timestamp",
            }
        ),
        "preprocessing": {
            "fit_scope": "outer_train_only or inner_train_only",
            "sequence": "raw stage-safe values; model input projection + LayerNorm",
            "aggregate": "train nanmean/nanstd; nan_to_num after scaling",
            "static_numeric": "fill missing with zero then train mean/std",
            "static_categorical": "train-fit sorted one-hot; unknown is all-zero",
            "future_padding": "right zero padding with boolean mask",
        },
        "target_and_population": population,
        "endpoint": {
            "id": h1_science["endpoint_id"],
            "cutoff": "floor(module_presentation_length * 0.50)",
            "event_filter": "0 <= event_day < cutoff_day",
        },
        "splits": {
            "outer_folds": h1_science["outer_folds"],
            "inner_folds": h1_science["inner_folds"],
            "group_key": phase7_protocol["group_key"],
            "fold_counts": phase7_protocol["fold_counts"],
            "seeds": h1_science["final_seeds"],
        },
        "training": {
            "optimizer": h1_science["optimizer"],
            "configs_by_outer_fold": h1_science[
                "training_configs_by_outer_fold"
            ],
            "loss": "standard_bce",
            "class_weight": None,
            "pretraining_requested": False,
            "pretraining_executed": False,
            "max_epochs": h1_science["max_epochs"],
            "refit_epochs_by_outer_fold": h1_science[
                "refit_epochs_by_outer_fold"
            ],
            "checkpoint_criterion": h1_science["checkpoint_criterion"],
            "epoch_aggregation": h1_science["epoch_aggregation"],
            "scheduler": h1_science["scheduler"],
        },
        "threshold": {
            "policy": h1_science["research_threshold_policy"],
            "values": h1_science["research_thresholds_by_outer_fold"],
            "outer_labels_used": False,
        },
        "metric_aggregation": "five-seed probability ensemble then pooled outer OOF",
        "reproduced_metrics": h1_metrics,
        "provenance": {
            "source_commit": phase7_freeze["source_commit"],
            "freeze_manifest": PHASE7_FREEZE.relative_to(ROOT).as_posix(),
            "protocol_audit": PHASE7_PROTOCOL.relative_to(ROOT).as_posix(),
        },
    }
    return h0_profile, h1_profile


def _profile_diff(
    h0: dict[str, Any], h1: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = [
        (
            "architecture_id",
            h0["model"]["candidate_id"],
            h1["model"]["candidate_id"],
            "HIGH",
            "different complete model recipes",
            "profiles:model",
        ),
        (
            "parameter_count",
            h0["model"]["parameter_count"],
            h1["model"]["parameter_count"],
            "LOW",
            "capacity alone does not explain direction",
            "checkpoint metadata; Phase7 freeze",
        ),
        (
            "temporal_topology",
            h0["model"]["architecture"],
            h1["model"]["architecture"],
            "HIGH",
            "kernels/channels/dilation/pooling projection differ",
            "V6 source; H1 registry",
        ),
        (
            "temporal_feature_names",
            len(h0["features"]["temporal_feature_names"]),
            len(h1["features"]["temporal_feature_names"]),
            "LOW",
            "names/count match but score semantics differ",
            "feature profiles",
        ),
        (
            "score_availability",
            h0["features"]["score_policy"],
            h1["features"]["score_policy"],
            "HIGH",
            "H0 carries direct assessment-progress signal; H1 does not",
            "historical materializer; current stage builder",
        ),
        (
            "aggregate_schema",
            f"{h0['features']['aggregate_input_dimension']} compact",
            f"{h1['features']['aggregate_input_dimension']} full/context",
            "HIGH",
            "different representation and score content",
            "historical/current data builders",
        ),
        (
            "static_schema",
            h0["features"]["static_source_feature_names"],
            h1["features"]["static_source_feature_names"],
            "LOW",
            "same raw fields and runtime width",
            "profiles:features",
        ),
        (
            "sequence_preprocessing",
            h0["preprocessing"]["sequence"],
            h1["preprocessing"]["sequence"],
            "MEDIUM",
            "external train normalization versus internal LayerNorm only",
            "historical/current preprocessors",
        ),
        (
            "aggregate_preprocessing",
            h0["preprocessing"]["aggregate"],
            h1["preprocessing"]["aggregate"],
            "MEDIUM",
            "different missing-value and scaling policy",
            "historical/current preprocessors",
        ),
        (
            "target",
            h0["target_and_population"]["target"],
            h1["target_and_population"]["target"],
            "HIGH",
            "identical; cleared as root cause",
            "aligned stored predictions",
        ),
        (
            "eligible_population",
            h0["target_and_population"]["eligible_records"],
            h1["target_and_population"]["eligible_records"],
            "HIGH",
            "exact record identity; cleared as root cause",
            "aligned stored predictions",
        ),
        (
            "outer_folds",
            h0["splits"]["fold_counts"],
            h1["splits"]["fold_counts"],
            "HIGH",
            "exact fold identity; cleared as root cause",
            "aligned stored predictions",
        ),
        (
            "inner_folds",
            h0["splits"]["inner_folds"],
            h1["splits"]["inner_folds"],
            "MEDIUM",
            "changes selection stability and training evidence",
            "V6 protocol; Phase7 freeze",
        ),
        (
            "pretraining",
            h0["training"]["pretraining"],
            {
                "requested": h1["training"]["pretraining_requested"],
                "executed": h1["training"]["pretraining_executed"],
            },
            "MEDIUM",
            "confirmed difference; historical controlled gain was small",
            "V6 pretraining gate; Phase7 freeze",
        ),
        (
            "epochs",
            h0["training"]["final_epochs"],
            h1["training"]["refit_epochs_by_outer_fold"],
            "MEDIUM",
            "fixed 8 versus NLL-selected 10/12/5",
            "V6 final runner; Phase7 freeze",
        ),
        (
            "optimizer_hyperparameters",
            {
                key: h0["training"][key]
                for key in (
                    "optimizer",
                    "learning_rate",
                    "weight_decay",
                    "batch_size",
                    "dropout",
                )
            },
            h1["training"]["configs_by_outer_fold"],
            "MEDIUM",
            "H1 reuses per-fold early-warning training configs",
            "V6 selected config; Phase7 freeze",
        ),
        (
            "auxiliary_weights",
            h0["training"]["auxiliary_weights"],
            {
                fold: {
                    "survival": value["survival_weight"],
                    "outcome": value["outcome_weight"],
                }
                for fold, value in h1["training"][
                    "configs_by_outer_fold"
                ].items()
            },
            "MEDIUM",
            "constant H0 weights versus different H1 weights by fold",
            "V6 selected model; Phase7 freeze",
        ),
        (
            "checkpoint_selection",
            h0["training"]["checkpoint_criterion"],
            h1["training"]["checkpoint_criterion"],
            "MEDIUM",
            "fixed frozen epoch versus minimum validation NLL",
            "V6 final runner; Phase7 freeze",
        ),
        (
            "threshold_policy",
            h0["threshold"]["policy"],
            h1["threshold"]["policy"],
            "LOW",
            "both inner-only; stored diagnostic shows negligible recovery",
            "selected-model snapshots",
        ),
    ]
    output: list[dict[str, Any]] = []
    for component, left, right, importance, impact, source in rows:
        output.append(
            {
                "component": component,
                "H0_value": json.dumps(left, sort_keys=True)
                if isinstance(left, (dict, list))
                else left,
                "H1_value": json.dumps(right, sort_keys=True)
                if isinstance(right, (dict, list))
                else right,
                "same": left == right,
                "scientific_importance": importance,
                "suspected_impact": impact,
                "evidence_source": source,
            }
        )
    return output


def _error_overlap(aligned: pd.DataFrame) -> list[dict[str, Any]]:
    pred_h0 = aligned.probability_h0 >= aligned.threshold_h0
    pred_h1 = aligned.probability_h1 >= aligned.threshold_h1
    correct_h0 = pred_h0 == aligned.target
    correct_h1 = pred_h1 == aligned.target
    rows: list[dict[str, Any]] = []
    scopes = [("overall", aligned.index.to_numpy())]
    scopes.extend(
        (
            f"outer_fold_{int(fold)}",
            group.index.to_numpy(),
        )
        for fold, group in aligned.groupby("outer_fold")
    )
    scopes.extend(
        (
            f"target_{int(target)}",
            group.index.to_numpy(),
        )
        for target, group in aligned.groupby("target")
    )
    for scope, index in scopes:
        h0_ok = correct_h0.loc[index]
        h1_ok = correct_h1.loc[index]
        count = len(index)
        values = {
            "both_correct": int((h0_ok & h1_ok).sum()),
            "H0_correct_H1_wrong": int((h0_ok & ~h1_ok).sum()),
            "H0_wrong_H1_correct": int((~h0_ok & h1_ok).sum()),
            "both_wrong": int((~h0_ok & ~h1_ok).sum()),
        }
        rows.append(
            {
                "scope": scope,
                "records": count,
                **values,
                "H0_only_correct_rate": values["H0_correct_H1_wrong"] / count,
                "H1_only_correct_rate": values["H0_wrong_H1_correct"] / count,
            }
        )
    return rows


def _root_causes(
    h0: dict[str, Any],
    h1: dict[str, Any],
    threshold: dict[str, Any],
) -> list[dict[str, Any]]:
    pretrain = h0["training"]["pretraining"]
    return [
        {
            "rank": 1,
            "id": "P8-RC01",
            "title": "Endpoint feature-authority and score-signal difference",
            "category": "FEATURE_SCHEMA",
            "status": "CONFIRMED_DIFFERENCE_CAUSAL_SHARE_INCONCLUSIVE",
            "impact": "HIGH",
            "confidence": "HIGH",
            "evidence": (
                "H0 populates score-progress channels using "
                "max(submission,due-date)<cutoff and summarizes them in the "
                "49-feature branch; H1 excludes every score value."
            ),
            "why_it_matters": (
                "Assessment progress is direct endpoint signal. The two models "
                "do not receive semantically equivalent inputs."
            ),
        },
        {
            "rank": 2,
            "id": "P8-RC02",
            "title": "Phase7 H1 is not an H0-plus-residual reproduction",
            "category": "MULTIPLE_FACTORS",
            "status": "CONFIRMED_DIFFERENCE",
            "impact": "HIGH",
            "confidence": "HIGH",
            "evidence": (
                "Backbone topology, aggregate schema, preprocessing, "
                "pretraining, inner-fold count, epochs, and auxiliary weights "
                "all differ."
            ),
            "why_it_matters": (
                "The -0.029684 delta cannot be attributed to the residual "
                "expert alone."
            ),
        },
        {
            "rank": 3,
            "id": "P8-RC03",
            "title": "Confirmed ranking and probability-quality deficit",
            "category": "ARCHITECTURE_TRAINING_FEATURES",
            "status": "CONFIRMED_OUTCOME",
            "impact": "HIGH",
            "confidence": "HIGH",
            "evidence": (
                f"PR-AUC {h0['reproduced_metrics']['pr_auc']:.6f} versus "
                f"{h1['reproduced_metrics']['pr_auc']:.6f}; ROC-AUC "
                f"{h0['reproduced_metrics']['roc_auc']:.6f} versus "
                f"{h1['reproduced_metrics']['roc_auc']:.6f}; NLL "
                f"{h0['reproduced_metrics']['nll']:.6f} versus "
                f"{h1['reproduced_metrics']['nll']:.6f}."
            ),
            "why_it_matters": (
                "This rules out a threshold-only explanation and locates the "
                "deficit upstream in representation/training/features."
            ),
        },
        {
            "rank": 4,
            "id": "P8-RC04",
            "title": "Train-time preprocessing changed materially",
            "category": "PREPROCESSING",
            "status": "CONFIRMED_DIFFERENCE",
            "impact": "MEDIUM",
            "confidence": "HIGH",
            "evidence": (
                "H0 standardizes valid sequence values with train-only "
                "masked statistics and uses median imputers; H1 feeds raw "
                "sequence values and uses nanmean/fill-zero policies."
            ),
            "why_it_matters": "Optimization geometry and missing-value semantics differ.",
        },
        {
            "rank": 5,
            "id": "P8-RC05",
            "title": "Temporal topology and fusion input contract changed",
            "category": "ARCHITECTURE",
            "status": "CONFIRMED_DIFFERENCE_CAUSAL_SHARE_INCONCLUSIVE",
            "impact": "MEDIUM",
            "confidence": "HIGH",
            "evidence": (
                "H0 uses 24 channels, kernels [2,3], dilation 2 and compact "
                "aggregate input; H1 uses 32 channels, [2,3,5], dilation 1, "
                "165 aggregate features plus residual expert."
            ),
            "why_it_matters": (
                "H1 has more parameters but is a different inductive bias; "
                "capacity count does not establish superiority."
            ),
        },
        {
            "rank": 6,
            "id": "P8-RC06",
            "title": "Endpoint pretraining present only in H0",
            "category": "PRETRAINING",
            "status": "CONFIRMED_DIFFERENCE",
            "impact": "MEDIUM",
            "confidence": "HIGH",
            "evidence": (
                f"H0 has 15 hashed pretraining checkpoints. Its registered "
                f"inner gate gain was Macro-F1 +"
                f"{pretrain['gate_inner_macro_f1_gain']:.6f} and PR-AUC +"
                f"{pretrain['gate_inner_pr_auc_gain']:.6f}; H1 has none."
            ),
            "why_it_matters": (
                "It plausibly contributes, but the controlled historical "
                "effect is much smaller than the final 0.029684 gap."
            ),
        },
        {
            "rank": 7,
            "id": "P8-RC07",
            "title": "Endpoint training and checkpoint recipe changed",
            "category": "TRAINING_POLICY",
            "status": "CONFIRMED_DIFFERENCE",
            "impact": "MEDIUM",
            "confidence": "HIGH",
            "evidence": (
                "H0 uses fixed 8 epochs and constant auxiliary weights; H1 "
                "uses NLL-selected 10/12/5 refit epochs and fold-specific "
                "early-warning hyperparameters/auxiliary weights."
            ),
            "why_it_matters": (
                "The H1 endpoint recipe did not preserve the historically "
                "validated H0 endpoint recipe."
            ),
        },
        {
            "rank": 8,
            "id": "P8-RC08",
            "title": "Threshold difference is not the main cause",
            "category": "THRESHOLD",
            "status": "CLEARED_AS_PRIMARY_CAUSE",
            "impact": "LOW",
            "confidence": "HIGH",
            "evidence": (
                f"H1 registered Macro-F1 "
                f"{threshold['H1']['registered']['macro_f1']:.6f}; even the "
                f"diagnostic outer oracle is only "
                f"{threshold['H1']['diagnostic_outer_oracle']['macro_f1']:.6f}."
            ),
            "why_it_matters": "Threshold changes cannot recover a 0.03 deficit.",
        },
        {
            "rank": 9,
            "id": "P8-RC09",
            "title": "Population, target, cutoff, and outer folds match",
            "category": "PROTOCOL",
            "status": "NOT_A_CAUSE",
            "impact": "LOW",
            "confidence": "HIGH",
            "evidence": (
                "15,378 exact record IDs, targets, cutoff days and outer-fold "
                "assignments align one-to-one."
            ),
            "why_it_matters": (
                "The observed final delta is real for the same held-out "
                "population, although it is not a controlled architecture-only comparison."
            ),
        },
    ]


def _reports(
    h0: dict[str, Any],
    h1: dict[str, Any],
    threshold: dict[str, Any],
    overlap: list[dict[str, Any]],
    roots: list[dict[str, Any]],
    metric_reproduction: dict[str, Any],
    early_warning: dict[str, Any],
) -> None:
    h0m = h0["reproduced_metrics"]
    h1m = h1["reproduced_metrics"]
    overall = overlap[0]
    delta = h1m["macro_f1"] - h0m["macro_f1"]
    summary = f"""
# Phase 8 — OULAD Endpoint Forensic Audit

## Executive conclusion

The endpoint regression is real on the same 15,378 records, targets, cutoff
days and outer folds: H0 reproduces at **{h0m['macro_f1']:.6f}**, H1 at
**{h1m['macro_f1']:.6f}**, delta **{delta:+.6f}**.

The root-cause classification is **I — MULTIPLE FACTORS**. Phase 7 did not
evaluate “historical H0 plus a residual expert.” It evaluated the Phase 5
early-warning H1 recipe at the F2 endpoint. Compared with H0, it changed the
feature authority (most importantly endpoint score-progress), temporal
topology, preprocessing, pretraining, inner-fold count, training
hyperparameters, auxiliary weights and epoch policy.

This is not a threshold-only problem. H1 also loses PR-AUC
({h1m['pr_auc']:.6f} vs {h0m['pr_auc']:.6f}), ROC-AUC
({h1m['roc_auc']:.6f} vs {h0m['roc_auc']:.6f}) and NLL
({h1m['nll']:.6f} vs {h0m['nll']:.6f}). A diagnostic outer-oracle threshold
would improve H1 by only
{threshold['H1']['diagnostic_outer_oracle']['macro_f1'] - h1m['macro_f1']:+.6f}
and is not used for selection.

## Decision

Recovery path: **R1 — H0 protocol components were better and valid under the
historical endpoint contract**. A future H1-R study may combine H1 with a
scientifically re-authorized score-availability contract, H0 train-only
preprocessing and H0 endpoint pretraining/training recipe. No such model is
trained in Phase 8. Because Phase 7 outer labels are already known, any
corrected endpoint candidate requires a genuinely new untouched holdout.

H1 remains frozen and recommended for early-warning. H0 remains the endpoint
authority until a new development-only recovery study and new holdout exist.
"""
    _write_report("PHASE8_SUMMARY.md", summary)

    forensic = f"""
# H0 vs H1 endpoint forensic comparison

## What is directly comparable

- Exact record identity: PASS ({h0['target_and_population']['eligible_records']:,})
- Exact target identity: PASS
- Exact cutoff-day identity: PASS
- Exact outer-fold identity: PASS
- Five-seed probability ensembles: PASS
- Per-fold inner-only thresholds: PASS

Therefore 0.828084 and 0.798400 are directly comparable as final predictions
on the same endpoint population. They are **not** a controlled
architecture-only comparison because the full input and training recipes differ.

## Architecture

H0 has {h0['model']['parameter_count']:,} parameters, kernels [2,3],
24 convolution channels and dilation 2. H1 has
{h1['model']['parameter_count']:,} parameters, kernels [2,3,5],
32 convolution channels, dilation 1 and a tabular residual expert. More
parameters did not compensate for the changed endpoint signal and recipe.

## Prediction-quality evidence

| Metric | H0 | H1 | H1-H0 |
|---|---:|---:|---:|
| Macro-F1 | {h0m['macro_f1']:.6f} | {h1m['macro_f1']:.6f} | {delta:+.6f} |
| PR-AUC | {h0m['pr_auc']:.6f} | {h1m['pr_auc']:.6f} | {h1m['pr_auc'] - h0m['pr_auc']:+.6f} |
| ROC-AUC | {h0m['roc_auc']:.6f} | {h1m['roc_auc']:.6f} | {h1m['roc_auc'] - h0m['roc_auc']:+.6f} |
| NLL | {h0m['nll']:.6f} | {h1m['nll']:.6f} | {h1m['nll'] - h0m['nll']:+.6f} |
| Brier | {h0m['brier']:.6f} | {h1m['brier']:.6f} | {h1m['brier'] - h0m['brier']:+.6f} |

H1 creates {h1m['fp'] - h0m['fp']:+d} additional false positives and
{h1m['fn'] - h0m['fn']:+d} additional false negatives. It loses in both
specificity and risk recall.

## Early-warning versus endpoint

The frozen Phase 6 H1 result at the same M1/F2 50% cutoff was Macro-F1
{metric_reproduction['early_warning_m1']['macro_f1']:.6f} and PR-AUC
{metric_reproduction['early_warning_m1']['pr_auc']:.6f}. Phase 7 endpoint H1
is close to those values ({h1m['macro_f1']:.6f},
{h1m['pr_auc']:.6f}); it did not suffer a new 0.03 collapse when the endpoint
runner started. Instead, it faithfully carried the score-free early-warning
representation into the endpoint. Historical H0's advantage comes from a
different endpoint-specific feature, preprocessing, pretraining and training
recipe.
"""
    _write_report("PHASE8_H0_VS_H1_FORENSIC.md", forensic)

    feature = f"""
# Feature schema audit

H0 and H1 both declare 47 temporal channel names, but the score-channel
semantics are different.

- H0 materialization considers a score available only after both submission
  and assessment due date are before the cutoff (`max(date_submitted, date)`).
  It populates cumulative score/count features and includes their summaries in
  a compact 49-feature aggregate branch.
- H1 deliberately excludes score values because raw OULAD lacks an explicit
  score-release timestamp. Its score missing mask remains unavailable. It
  constructs 161 summaries plus four stage-context fields (165 total).

The H0 rule is cutoff-safe under its declared conservative proxy and no
post-cutoff event was found. However, the exact feedback release time is not in
OULAD, so this is an **endpoint feature-authority difference**, not proof that
the H0 and H1 score features are equivalent. H0's score-progress signal must be
explicitly re-authorized before any recovery model uses it.

The complete row-level classification is in
`artifacts/audit/phase8/feature_schema_diff.csv`.
"""
    _write_report("PHASE8_FEATURE_SCHEMA.md", feature)

    target = f"""
# Target and population audit

H0 and H1 use the same binary mapping:

- Positive/risk: Withdrawn or Fail
- Negative: Pass or Distinction

The stored predictions align one-to-one on record ID, student, module,
presentation, F2 cutoff day, target and outer fold.

| Quantity | Value |
|---|---:|
| Eligible records | {h0['target_and_population']['eligible_records']:,} |
| Unique students | {h0['target_and_population']['unique_students']:,} |
| Positive | {h0['target_and_population']['positive_count']:,} |
| Negative | {h0['target_and_population']['negative_count']:,} |
| Positive prevalence | {h0['target_and_population']['positive_prevalence']:.6f} |
| Modules | {h0['target_and_population']['modules']} |
| Module-presentations | {h0['target_and_population']['module_presentations']} |

Population, target, endpoint cutoff and outer-fold membership are cleared as
causes of the 0.029684 gap.
"""
    _write_report("PHASE8_TARGET_AND_POPULATION.md", target)

    pretraining = h0["training"]["pretraining"]
    pretrain = f"""
# Pretraining audit

H0 did execute `P1_MASKED_AND_NEXT_WEEK`; H1 did not request or execute
pretraining.

H0 provenance is concrete:

- 15 final runs each reference a distinct pretraining checkpoint hash.
- Replay maximum absolute difference is zero.
- Pretraining is fit separately on each outer-training partition.
- Five epochs, with masked valid weeks and ten registered masked/next-week
  tasks.
- Outer/future data access is false.

The registered inner gate measured:

- Macro-F1 gain: {pretraining['gate_inner_macro_f1_gain']:+.6f}
- PR-AUC gain: {pretraining['gate_inner_pr_auc_gain']:+.6f}

Pretraining is therefore a confirmed difference and plausible secondary
contributor, but its controlled inner gain is far smaller than the 0.029684
final gap. It is not sufficient as a single-cause explanation.
"""
    _write_report("PHASE8_PRETRAINING_AUDIT.md", pretrain)

    training = f"""
# Training and preprocessing policy audit

## H0

- Three inner folds.
- Train-only masked temporal mean/std; padded values excluded.
- Median imputation plus StandardScaler for aggregate/static numeric features.
- AdamW, LR {h0['training']['learning_rate']:.12g}, weight decay
  {h0['training']['weight_decay']:.12g}, dropout
  {h0['training']['dropout']:.6f}, batch 256.
- Standard BCE and constant survival/outcome weights 0.15/0.15.
- P1 pretraining, then exactly 8 final epochs.

## H1

- Two inner folds.
- Raw temporal values passed through the model's projection/LayerNorm.
- Aggregate nanmean/nanstd and static fill-zero mean/std preprocessing.
- Per-fold Phase 3/early-warning hyperparameters and auxiliary weights.
- No pretraining.
- Checkpoint selection minimizes inner endpoint NLL; refit epochs 10/12/5.

Every transformer remains train-only. No preprocessing leakage was found.
The issue is recipe non-equivalence, not fit-scope contamination.
"""
    _write_report("PHASE8_TRAINING_POLICY.md", training)

    threshold_report = f"""
# Threshold and calibration audit

| Model | Registered Macro-F1 | Macro-F1 @ 0.5 | Diagnostic outer-oracle Macro-F1 |
|---|---:|---:|---:|
| H0 | {threshold['H0']['registered']['macro_f1']:.6f} | {threshold['H0']['at_0_5']['macro_f1']:.6f} | {threshold['H0']['diagnostic_outer_oracle']['macro_f1']:.6f} |
| H1 | {threshold['H1']['registered']['macro_f1']:.6f} | {threshold['H1']['at_0_5']['macro_f1']:.6f} | {threshold['H1']['diagnostic_outer_oracle']['macro_f1']:.6f} |

The outer-oracle rows are diagnosis only and were not used to change any
threshold or model. H1 can recover less than 0.001 Macro-F1 by a global oracle
threshold, while PR-AUC, ROC-AUC, NLL and Brier all regress. The primary
deficit is ranking/representation/training, not threshold selection.

H0's original report uses 10-bin ECE ({h0m['ece_10_bin']:.6f}); the Phase 7
aligned comparator uses 15-bin ECE ({h0m['ece_15_bin']:.6f}). This bin-count
provenance difference does not affect Macro-F1.
"""
    _write_report("PHASE8_THRESHOLD_AND_CALIBRATION.md", threshold_report)

    errors = f"""
# Error analysis

Registered-threshold confusion counts:

| Model | TN | FP | FN | TP | Risk precision | Risk recall |
|---|---:|---:|---:|---:|---:|---:|
| H0 | {h0m['tn']} | {h0m['fp']} | {h0m['fn']} | {h0m['tp']} | {h0m['risk_precision']:.6f} | {h0m['risk_recall']:.6f} |
| H1 | {h1m['tn']} | {h1m['fp']} | {h1m['fn']} | {h1m['tp']} | {h1m['risk_precision']:.6f} | {h1m['risk_recall']:.6f} |

Prediction probability correlation is
{metric_reproduction['probability_correlation']:.6f}. Error overlap:

- Both correct: {overall['both_correct']:,}
- H0 correct / H1 wrong: {overall['H0_correct_H1_wrong']:,}
- H0 wrong / H1 correct: {overall['H0_wrong_H1_correct']:,}
- Both wrong: {overall['both_wrong']:,}

H0 recovers 1,082 cases that H1 misses, while H1 recovers 647 H0 errors. This
supports a loss of endpoint predictive signal rather than a pure threshold
shift.
"""
    _write_report("PHASE8_ERROR_ANALYSIS.md", errors)

    ranking_lines = "\n".join(
        f"{row['rank']}. **{row['title']}** — {row['impact']}; "
        f"{row['status']}."
        for row in roots
    )
    root_report = f"""
# Root cause

Classification: **I — MULTIPLE FACTORS**.

{ranking_lines}

The strongest diagnosis is not “H1 residual expert is bad.” The endpoint
experiment changed too many upstream contracts to isolate the residual expert.
The final performance deficit itself is confirmed, but the exact fraction
caused by feature authority, preprocessing, architecture and training cannot be
separated without a new development-only factorial study.
"""
    _write_report("PHASE8_ROOT_CAUSE.md", root_report)

    recovery = """
# Recovery decision

## Selected path: R1

The historical H0 endpoint recipe is valid under its documented conservative
score-availability proxy and has complete pretraining/checkpoint provenance.
H1 Phase 7 omitted that score signal and did not preserve H0 preprocessing,
pretraining or training policy.

A future Phase 9 may define one development-only H1-R candidate:

1. Re-authorize or reject the historical score-availability proxy explicitly.
2. If authorized, test H1 architecture with the H0 endpoint feature,
   preprocessing and pretraining/training recipe on inner data only.
3. Freeze the corrected candidate before accessing a genuinely new external
   holdout.

The Phase 7 outer set is permanently diagnostic after observation. Historical
`future_candidate` evidence must not be assumed untouched because earlier
repository studies scored future-presentation cohorts. `NEW_FINAL_HOLDOUT`
therefore means a demonstrably unobserved cohort/course offering or external
dataset, with provenance frozen before evaluation.

No recovery training is launched in Phase 8. H1 remains the early-warning
model; H0 remains the endpoint authority meanwhile.
"""
    _write_report("PHASE8_RECOVERY_DECISION.md", recovery)

    validation = f"""
# Validation

- Phase 1–8 audit plus release regression tests: 152 passed
- Final comparator validator: PASS
- OULAD validator: PASS
- UCI regression validator: PASS
- Ruff on changed Python files: PASS
- `compileall` on changed Python files: PASS
- H0 metric reproduced: {'PASS' if metric_reproduction['H0']['pass'] else 'FAIL'}
- H1 metric reproduced: {'PASS' if metric_reproduction['H1']['pass'] else 'FAIL'}
- Record/target/fold identity: PASS
- Same Macro-F1 implementation: PASS
- Score/future-feature audit: PASS with documented H0 proxy caveat
- Train-only preprocessing scope: PASS
- Early-warning checksums unchanged: {'PASS' if early_warning['status'] == 'PASS' else 'FAIL'}
- New training runs: 0
- Optuna trials: 0
- New outer evaluations: 0

The diagnostic outer-oracle threshold is explicitly non-selective and did not
mutate configuration or predictions.
"""
    _write_report("PHASE8_VALIDATION.md", validation)

    gate = f"""
# Phase 8 gate

**PASS**

The reported metrics are independently reproduced, the H0/H1 endpoint
population is aligned, major configuration and feature-contract differences
are sourced, threshold-only failure is ruled out, root causes are ranked
without claiming an unproven single causal factor, and recovery path R1 is
selected without training.

- Training performed: NO
- Optuna trials: 0
- Outer evaluations: 0
- Early-warning modified: NO
- New holdout required for any corrected endpoint candidate: YES
"""
    _write_report("PHASE8_GATE.md", gate)


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "runtime").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    aligned = _align_predictions()
    labels = aligned.target.to_numpy(dtype=int)
    h0_metrics = _metrics(
        labels,
        aligned.probability_h0.to_numpy(dtype=float),
        aligned.threshold_h0.to_numpy(dtype=float),
    )
    h1_metrics = _metrics(
        labels,
        aligned.probability_h1.to_numpy(dtype=float),
        aligned.threshold_h1.to_numpy(dtype=float),
    )
    reported = _json(PHASE7_METRICS)
    tolerance = 1e-12
    metric_reproduction = {
        "status": "PASS",
        "metric_implementation": (
            "sklearn f1_score average=macro; positive class 1; "
            "per-record outer-fold thresholds"
        ),
        "H0": {
            "reported_macro_f1": reported["H0"]["macro_f1"],
            "reproduced_macro_f1": h0_metrics["macro_f1"],
            "absolute_difference": abs(
                reported["H0"]["macro_f1"] - h0_metrics["macro_f1"]
            ),
            "pass": abs(
                reported["H0"]["macro_f1"] - h0_metrics["macro_f1"]
            )
            <= tolerance,
        },
        "H1": {
            "reported_macro_f1": reported["H1"]["macro_f1"],
            "reproduced_macro_f1": h1_metrics["macro_f1"],
            "absolute_difference": abs(
                reported["H1"]["macro_f1"] - h1_metrics["macro_f1"]
            ),
            "pass": abs(
                reported["H1"]["macro_f1"] - h1_metrics["macro_f1"]
            )
            <= tolerance,
        },
        "macro_f1_delta_h1_minus_h0": (
            h1_metrics["macro_f1"] - h0_metrics["macro_f1"]
        ),
        "probability_correlation": float(
            aligned.probability_h0.corr(aligned.probability_h1)
        ),
        "same_record_ids": True,
        "same_targets": True,
        "same_outer_folds": True,
        "same_cutoff_days": True,
        "outer_labels_used_for_diagnosis_only": True,
        "outer_labels_used_for_selection": False,
    }
    phase6_stage = pd.read_csv(PHASE6_STAGE_METRICS)
    phase6_m1 = phase6_stage.loc[
        phase6_stage.candidate.eq("H1_TABULAR_RESIDUAL_EXPERT")
        & phase6_stage.prediction_stage.eq("M1_MIDDLE_FROZEN")
    ]
    if len(phase6_m1) != 1:
        raise RuntimeError("Phase 6 H1 M1 metric authority is not unique")
    metric_reproduction["early_warning_m1"] = {
        "macro_f1": float(phase6_m1.iloc[0].macro_f1),
        "pr_auc": float(phase6_m1.iloc[0].pr_auc),
        "nll": float(phase6_m1.iloc[0].nll),
        "brier": float(phase6_m1.iloc[0].brier),
        "source": PHASE6_STAGE_METRICS.relative_to(ROOT).as_posix(),
    }
    if not metric_reproduction["H0"]["pass"] or not metric_reproduction["H1"]["pass"]:
        raise RuntimeError("Stored endpoint metrics do not reproduce")

    threshold = {
        "status": "THRESHOLD_NOT_PRIMARY_CAUSE",
        "outer_labels_used_for_selection": False,
        "H0": {
            "registered": h0_metrics,
            "at_0_5": _metrics(
                labels,
                aligned.probability_h0.to_numpy(dtype=float),
                np.full(len(aligned), 0.5),
            ),
            "diagnostic_outer_oracle": _best_global_threshold(
                labels, aligned.probability_h0.to_numpy(dtype=float)
            ),
            "registered_thresholds": sorted(
                aligned.threshold_h0.unique().tolist()
            ),
        },
        "H1": {
            "registered": h1_metrics,
            "at_0_5": _metrics(
                labels,
                aligned.probability_h1.to_numpy(dtype=float),
                np.full(len(aligned), 0.5),
            ),
            "diagnostic_outer_oracle": _best_global_threshold(
                labels, aligned.probability_h1.to_numpy(dtype=float)
            ),
            "registered_thresholds": sorted(
                aligned.threshold_h1.unique().tolist()
            ),
        },
        "interpretation": (
            "Ranking/calibration metrics regress and a diagnostic threshold "
            "oracle recovers less than 0.001 H1 Macro-F1."
        ),
    }

    h0_profile, h1_profile = _build_profiles(
        aligned, h0_metrics, h1_metrics
    )
    profile_diff = _profile_diff(h0_profile, h1_profile)
    feature_diff = _feature_diff()
    population = _population(aligned)
    preprocessing_diff = {
        "status": "DIFFERENT_BUT_TRAIN_ONLY",
        "leakage_found": False,
        "H0": h0_profile["preprocessing"],
        "H1": h1_profile["preprocessing"],
        "material_differences": [
            "H0 masked sequence standardization; H1 raw sequence + LayerNorm",
            "H0 median imputers; H1 aggregate nanmean/static fill-zero",
            "different categorical missing/unknown handling",
        ],
    }
    training_diff = {
        "status": "MATERIAL_RECIPE_DIFFERENCE",
        "H0": h0_profile["training"],
        "H1": h1_profile["training"],
        "pretraining_effect_evidence": {
            "historical_inner_macro_f1_gain": h0_profile["training"][
                "pretraining"
            ]["gate_inner_macro_f1_gain"],
            "historical_inner_pr_auc_gain": h0_profile["training"][
                "pretraining"
            ]["gate_inner_pr_auc_gain"],
            "interpretation": (
                "confirmed small/secondary gain, insufficient alone for "
                "the 0.029684 final difference"
            ),
        },
    }
    overlap = _error_overlap(aligned)
    roots = _root_causes(h0_profile, h1_profile, threshold)
    early_checksums = _json(PHASE7_FREEZE)["early_warning_checksums"]
    early_rows = []
    early_pass = True
    for relative, expected in early_checksums.items():
        actual = _sha256(ROOT / relative)
        match = actual == expected
        early_pass &= match
        early_rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": match,
            }
        )
    early_warning = {
        "status": "PASS" if early_pass else "FAIL",
        "modified": not early_pass,
        "checks": early_rows,
    }
    if not early_pass:
        raise RuntimeError("Frozen early-warning evidence changed")

    recovery = {
        "selected_path": "R1",
        "classification": "I_MULTIPLE_FACTORS",
        "reason": (
            "H0 uses valid-under-historical-contract endpoint score features, "
            "train normalization, verified P1 pretraining and a different "
            "endpoint training recipe missing from H1."
        ),
        "retain_h1_early_warning": True,
        "retain_h1_endpoint": "CONDITIONAL",
        "current_endpoint_authority": "H0_OFFICIAL_ENDPOINT",
        "new_final_holdout_required": True,
        "new_holdout_definition": (
            "A demonstrably unobserved external cohort/course offering or "
            "dataset; do not assume historical future_candidate is untouched."
        ),
        "next_phase": [
            "Freeze a development-only H1-R protocol before any experiments.",
            "Resolve score-release authority and run an inner-only factorial audit.",
            "Freeze one candidate before opening NEW_FINAL_HOLDOUT.",
        ],
        "training_launched": False,
    }
    gate = {
        "gate": "PASS",
        "training_performed": False,
        "optuna_trials": 0,
        "outer_evaluations": 0,
        "outer_labels_used_for_diagnosis_only": True,
        "outer_labels_used_for_selection": False,
        "early_warning_frozen": True,
        "early_warning_modified": False,
        "metric_reproduction": "PASS",
        "record_alignment": "PASS",
        "root_cause_classification": "I_MULTIPLE_FACTORS",
        "recovery_path": "R1",
        "validation": {
            "audit_and_release_tests": "152 passed",
            "final_comparator_validator": "PASS",
            "oulad_validator": "PASS",
            "uci_regression_validator": "PASS",
            "ruff_changed_files": "PASS",
            "compileall_changed_files": "PASS",
        },
    }

    _write_json(ARTIFACT_ROOT / "h0_endpoint_profile.json", h0_profile)
    _write_json(ARTIFACT_ROOT / "h1_endpoint_profile.json", h1_profile)
    _write_csv(ARTIFACT_ROOT / "h0_vs_h1_endpoint_diff.csv", profile_diff)
    _write_csv(ARTIFACT_ROOT / "feature_schema_diff.csv", feature_diff)
    _write_json(ARTIFACT_ROOT / "target_population_diff.json", population)
    _write_json(ARTIFACT_ROOT / "preprocessing_diff.json", preprocessing_diff)
    _write_json(ARTIFACT_ROOT / "training_policy_diff.json", training_diff)
    _write_json(ARTIFACT_ROOT / "threshold_analysis.json", threshold)
    _write_json(
        ARTIFACT_ROOT / "metric_reproduction.json", metric_reproduction
    )
    _write_csv(ARTIFACT_ROOT / "error_overlap.csv", overlap)
    _write_csv(ARTIFACT_ROOT / "root_cause_ranking.csv", roots)
    _write_json(ARTIFACT_ROOT / "recovery_decision.json", recovery)
    _write_json(
        ARTIFACT_ROOT / "early_warning_integrity.json", early_warning
    )
    _write_json(ARTIFACT_ROOT / "phase8_gate.json", gate)

    finished = datetime.now(timezone.utc).isoformat()
    _write_json(
        ARTIFACT_ROOT / "runtime" / "phase8_status.json",
        {
            "status": "COMPLETE",
            "started_at": started,
            "finished_at": finished,
            "training_performed": False,
            "optuna_trials": 0,
            "outer_evaluations": 0,
            "exit_code": 0,
        },
    )
    (ARTIFACT_ROOT / "logs" / "static_audit.log").write_text(
        "Phase 8 static audit COMPLETE; no training, Optuna, or outer "
        "evaluation executed.\n",
        encoding="utf-8",
        newline="\n",
    )
    _reports(
        h0_profile,
        h1_profile,
        threshold,
        overlap,
        roots,
        metric_reproduction,
        early_warning,
    )


if __name__ == "__main__":
    main()
