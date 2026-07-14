"""Read-only Benchmark V2.1.1 validation and reporting primitives.

Expected coverage is defined by the protocol registry and fold manifest.  No
function in this module infers an expectation from prediction or metric output.
"""
from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.evaluation.metrics import CLASS_ORDER, classification_metrics
from src.evaluation.protocol import canonical_json, file_checksum

PATCH_VERSION = "v2.1.1"
LABEL_NAMES = ("Low", "Medium", "High")
SEEDS = (42, 52, 62, 72, 82)
JOB_COLUMNS = ("scenario", "model_name", "feature_set_id", "outer_fold", "training_seed")
SCALAR_METRICS = (
    "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy",
    "quadratic_weighted_kappa", "ordinal_mae", "brier_score",
    "pr_auc_macro", "ece_top_label_equal_width_10",
)


@dataclass(frozen=True)
class ModelContract:
    scenario: str
    model_name: str
    feature_set_id: str
    n_seeds: int
    estimator_group: str
    preprocessing: str = "fold_train_fit"
    scaler: str = "none"


MODEL_REGISTRY = (
    ModelContract("late_stage", "majority", "G1+G2", 1, "deterministic", "none"),
    ModelContract("late_stage", "g2_rule", "G2", 1, "deterministic", "none"),
    ModelContract("late_stage", "logistic_g2", "G2", 1, "single_seed", scaler="standard_scaler_train_only"),
    ModelContract("late_stage", "logistic_g1_g2", "G1+G2", 1, "single_seed", scaler="standard_scaler_train_only"),
    ModelContract("late_stage", "ordinal_logistic", "G1+G2", 1, "single_seed", scaler="standard_scaler_train_only"),
    ModelContract("late_stage", "ridge_regression", "G1+G2", 1, "single_seed", scaler="standard_scaler_train_only"),
    ModelContract("late_stage", "hgb_g1_g2", "G1+G2", 1, "single_seed"),
    ModelContract("late_stage", "small_mlp", "G1+G2", 5, "five_seed_fold_mean", scaler="standard_scaler_train_only"),
    ModelContract("early_warning", "majority", "G1", 1, "deterministic", "none"),
    ModelContract("early_warning", "g1_rule", "G1", 1, "deterministic", "none"),
    ModelContract("early_warning", "logistic_g1", "G1", 1, "single_seed", scaler="standard_scaler_train_only"),
    ModelContract("early_warning", "ordinal_logistic", "G1", 1, "single_seed", scaler="standard_scaler_train_only"),
    ModelContract("early_warning", "ridge_regression", "G1", 1, "single_seed", scaler="standard_scaler_train_only"),
    ModelContract("early_warning", "hgb_g1", "G1", 1, "single_seed"),
    ModelContract("early_warning", "small_mlp", "G1", 5, "five_seed_fold_mean", scaler="standard_scaler_train_only"),
    ModelContract("late_stage", "cnn_only", "G1+G2", 5, "five_seed_fold_mean", scaler="standard_scaler_train_only"),
    ModelContract("late_stage", "bilstm_only", "G1+G2", 5, "five_seed_fold_mean", scaler="standard_scaler_train_only"),
    ModelContract("late_stage", "cnn_bilstm_legacy_config_v2_refit", "G1+G2", 5, "five_seed_fold_mean", scaler="standard_scaler_train_only"),
    ModelContract("late_stage", "cnn_bilstm_v2_tuned", "G1+G2", 5, "five_seed_fold_mean", scaler="standard_scaler_train_only"),
)

REQUIRED_PAIRS = (
    ("late_stage", "cnn_bilstm_v2_tuned", "g2_rule"),
    ("late_stage", "cnn_bilstm_v2_tuned", "logistic_g2"),
    ("late_stage", "cnn_bilstm_v2_tuned", "hgb_g1_g2"),
    ("late_stage", "cnn_bilstm_v2_tuned", "small_mlp"),
    ("late_stage", "cnn_bilstm_v2_tuned", "bilstm_only"),
    ("late_stage", "cnn_bilstm_v2_tuned", "cnn_bilstm_legacy_config_v2_refit"),
    ("late_stage", "cnn_bilstm_v2_tuned", "cnn_only"),
    ("late_stage", "g2_rule", "hgb_g1_g2"),
    ("late_stage", "g2_rule", "small_mlp"),
    ("early_warning", "g1_rule", "small_mlp"),
    ("early_warning", "g1_rule", "hgb_g1"),
    ("early_warning", "g1_rule", "logistic_g1"),
    ("early_warning", "g1_rule", "ridge_regression"),
)


def semantic_checksum(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def validation_records_by_fold(fold_manifest: dict[str, Any]) -> dict[int, set[str]]:
    result = {fold: set() for fold in range(int(fold_manifest["outer_folds"]))}
    for row in fold_manifest["assignments"]:
        if row["outer_role"] == "validation":
            result[int(row["outer_fold"])].add(row["source_record_identity"])
    return result


def build_expected_job_contract(fold_manifest: dict[str, Any]) -> dict[str, Any]:
    """Build the independent expected universe from registry + fold manifest."""
    validation = validation_records_by_fold(fold_manifest)
    jobs = []
    for model in MODEL_REGISTRY:
        seeds = SEEDS if model.n_seeds == 5 else (42,)
        for fold, records in sorted(validation.items()):
            for seed in seeds:
                jobs.append({
                    "scenario": model.scenario, "model_name": model.model_name,
                    "feature_set_id": model.feature_set_id, "outer_fold": fold,
                    "training_seed": seed, "estimator_group": model.estimator_group,
                    "expected_record_count": len(records),
                })
    payload = {"contract_version": PATCH_VERSION, "source": "protocol_model_registry_plus_fold_manifest", "jobs": jobs}
    payload["semantic_checksum"] = semantic_checksum(payload)
    return payload


def compare_expected_jobs(contract: dict[str, Any], predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    expected = {tuple(row[c] for c in JOB_COLUMNS) for row in contract["jobs"]}
    counts = predictions.groupby(list(JOB_COLUMNS), dropna=False).size().reset_index(name="prediction_rows")
    actual = {tuple(row[c] for c in JOB_COLUMNS) for _, row in counts.iterrows()}
    duplicate_job_keys = int(counts.duplicated(list(JOB_COLUMNS)).sum())
    rows = []
    for key in sorted(expected | actual):
        rows.append({**dict(zip(JOB_COLUMNS, key)), "expected": key in expected, "actual": key in actual,
                     "status": "ok" if key in expected and key in actual else ("missing" if key in expected else "unexpected")})
    summary = {"expected": len(expected), "actual": len(actual), "missing": len(expected-actual),
               "unexpected": len(actual-expected), "duplicate_jobs": duplicate_job_keys}
    return pd.DataFrame(rows), summary


def validate_record_coverage(contract: dict[str, Any], predictions: pd.DataFrame, fold_manifest: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    validation = validation_records_by_fold(fold_manifest)
    expected_lookup = {tuple(row[c] for c in JOB_COLUMNS): row for row in contract["jobs"]}
    rows, duplicate_rows, invalid_jobs = [], 0, 0
    for key, group in predictions.groupby(list(JOB_COLUMNS), dropna=False):
        fold = int(key[3]); expected_ids = validation.get(fold, set()); ids = set(group["record_id"])
        duplicates = int(group.duplicated("record_id").sum()); duplicate_rows += duplicates
        missing = len(expected_ids - ids); outside = len(ids - expected_ids)
        ok = key in expected_lookup and not duplicates and not missing and not outside and len(group) == len(expected_ids)
        invalid_jobs += int(not ok)
        rows.append({**dict(zip(JOB_COLUMNS, key)), "expected_record_count": len(expected_ids),
                     "actual_record_rows": len(group), "unique_record_count": group.record_id.nunique(),
                     "duplicate_record_rows": duplicates, "missing_records": missing,
                     "records_outside_outer_validation": outside, "status": "valid" if ok else "invalid"})
    return pd.DataFrame(rows), {"duplicate_prediction_rows": duplicate_rows, "invalid_coverage_jobs": invalid_jobs}


def _stored_per_class(value: str) -> dict[str, Any]:
    return ast.literal_eval(value) if isinstance(value, str) else value


def recompute_metrics(predictions: pd.DataFrame, stored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scalar, confusion, per_class, scalar_bad, structured_bad = [], [], [], [], []
    stored_keys = ["scenario", "model_name", "outer_fold", "training_seed"]
    for key, group in predictions.groupby(list(JOB_COLUMNS), sort=True):
        job = dict(zip(JOB_COLUMNS, key)); match = stored
        for c in stored_keys:
            match = match[match[c] == job[c]]
        if len(match) != 1:
            structured_bad.append({**job, "metric": "stored_metric_row", "status": "missing_or_duplicate"})
            continue
        source = match.iloc[0]
        probs = group[["probability_low", "probability_medium", "probability_high"]].to_numpy(float)
        metrics = classification_metrics(group.true_label.to_numpy(), group.predicted_label.to_numpy(), probs)
        for name in SCALAR_METRICS:
            old, new = float(source[name]), float(metrics[name]); delta = abs(old-new); ok = delta <= 1e-6
            row = {**job, "metric": name, "stored": old, "recomputed": new, "absolute_difference": delta, "match": ok}
            scalar.append(row)
            if not ok: scalar_bad.append(row)
        stored_cm = json.loads(source["confusion_matrix"]); new_cm = metrics["confusion_matrix"]; cm_ok = stored_cm == new_cm
        confusion.append({**job, "stored_confusion_matrix": json.dumps(stored_cm), "recomputed_confusion_matrix": json.dumps(new_cm), "match": cm_ok})
        if not cm_ok: structured_bad.append({**job, "metric": "confusion_matrix", "status": "mismatch"})
        old_pc = _stored_per_class(source["per_class_f1"])
        for index, label in enumerate(LABEL_NAMES):
            new = metrics["per_class"][str(index)]; old = old_pc.get(str(index), {})
            f1_ok = abs(float(old.get("f1", np.nan))-new["f1"]) <= 1e-6
            support_ok = int(old.get("support", -1)) == new["support"]
            row = {**job, "class_index": index, "class_label": label,
                   "stored_precision": np.nan, "recomputed_precision": new["precision"], "precision_comparison": "not_stored",
                   "stored_recall": np.nan, "recomputed_recall": new["recall"], "recall_comparison": "not_stored",
                   "stored_f1": old.get("f1"), "recomputed_f1": new["f1"], "f1_match": f1_ok,
                   "stored_support": old.get("support"), "recomputed_support": new["support"], "support_match": support_ok}
            per_class.append(row)
            if not (f1_ok and support_ok): structured_bad.append({**job, "metric": f"per_class_{label}", "status": "mismatch"})
    structured_frame = pd.DataFrame(structured_bad, columns=[*JOB_COLUMNS, "metric", "status"])
    return pd.DataFrame(scalar), pd.DataFrame(confusion), pd.DataFrame(per_class), pd.DataFrame(scalar_bad), structured_frame


def feature_contracts(fold_manifest: dict[str, Any], dataset_version: int) -> list[dict[str, Any]]:
    cutoff = {"late_stage": "before_outcome_after_g2", "early_warning": "after_g1_before_g2"}
    contracts = []
    seen = set()
    for model in MODEL_REGISTRY:
        key = (model.scenario, model.feature_set_id, model.preprocessing, model.scaler)
        if key in seen: continue
        seen.add(key)
        ordered = model.feature_set_id.split("+")
        payload = {"contract_version": PATCH_VERSION, "scenario": model.scenario, "cutoff": cutoff[model.scenario],
                   "feature_set_id": model.feature_set_id, "ordered_features": ordered,
                   "preprocessing_contract": model.preprocessing, "scaler_contract": model.scaler,
                   "target_excluded": True, "temporal_availability_status": "verified_by_scenario_allowlist",
                   "class_order": list(LABEL_NAMES), "dataset_version": dataset_version,
                   "fold_manifest_checksum": fold_manifest["manifest_checksum"]}
        payload["semantic_checksum"] = semantic_checksum(payload)
        contracts.append(payload)
    return contracts


def checksum_validation(base: Path, expected: dict[str, str]) -> pd.DataFrame:
    rows = []
    for relative, checksum in expected.items():
        path = base / relative; actual = file_checksum(path) if path.is_file() else None
        rows.append({"path": relative, "expected_checksum": checksum, "actual_checksum": actual,
                     "missing": not path.is_file(), "valid": actual == checksum})
    return pd.DataFrame(rows)


def fold_metric_estimator(predictions: pd.DataFrame, scenario: str, model: str, metric: str = "macro_f1") -> dict[int, float]:
    frame = predictions[(predictions.scenario == scenario) & (predictions.model_name == model)]
    if frame.empty: raise ValueError(f"No predictions for {scenario}/{model}.")
    values: dict[int, float] = {}
    for fold, fold_frame in frame.groupby("outer_fold"):
        seed_scores = []
        for _, group in fold_frame.groupby("training_seed"):
            p = group[["probability_low", "probability_medium", "probability_high"]].to_numpy(float)
            seed_scores.append(classification_metrics(group.true_label, group.predicted_label, p)[metric])
        values[int(fold)] = float(np.mean(seed_scores))
    return values


def paired_comparisons(predictions: pd.DataFrame, pairs: Iterable[tuple[str, str, str]] = REQUIRED_PAIRS) -> pd.DataFrame:
    rows = []
    registry = {(m.scenario, m.model_name): m for m in MODEL_REGISTRY}
    for scenario, model_a, model_b in pairs:
        a = fold_metric_estimator(predictions, scenario, model_a); b = fold_metric_estimator(predictions, scenario, model_b)
        folds = sorted(set(a) & set(b)); differences = [a[f]-b[f] for f in folds]
        ca, cb = registry[(scenario, model_a)], registry[(scenario, model_b)]
        rows.append({"scenario": scenario, "model_a": model_a, "model_b": model_b,
                     "estimator_a": ca.estimator_group, "estimator_b": cb.estimator_group,
                     "feature_set_compatibility": "same_scenario_controlled_information",
                     "fold_aggregation": "mean_across_training_seeds_within_fold",
                     "metric": "macro_f1", "foldwise_scores_a": json.dumps([a[f] for f in folds]),
                     "foldwise_scores_b": json.dumps([b[f] for f in folds]), "foldwise_differences": json.dumps(differences),
                     "mean_difference": float(np.mean(differences)), "sd_difference": float(np.std(differences, ddof=1)),
                     "wins": sum(x > 1e-12 for x in differences), "ties": sum(abs(x) <= 1e-12 for x in differences),
                     "losses": sum(x < -1e-12 for x in differences), "record_level_comparison_available": True})
    return pd.DataFrame(rows)


def render_paired_markdown(frame: pd.DataFrame) -> str:
    keys = frame[["scenario", "model_a", "model_b"]].astype(str).agg(" / ".join, axis=1).tolist()
    selected = frame[["scenario", "model_a", "model_b", "mean_difference", "wins", "ties", "losses"]]
    header = "| " + " | ".join(selected.columns) + " |\n|" + "|".join(["---"] * len(selected.columns)) + "|\n"
    body = header + "".join("| " + " | ".join(str(value) for value in row) + " |\n" for row in selected.itertuples(index=False, name=None))
    return "# Paired comparisons V2.1.1\n\nSource keys: " + "; ".join(keys) + "\n\n" + body + "\n"


def aggregate_ece_corrections(scalar_mismatches: pd.DataFrame) -> pd.DataFrame:
    ece = scalar_mismatches[scalar_mismatches.metric == "ece_top_label_equal_width_10"].copy()
    if ece.empty: return pd.DataFrame()
    ece["correction"] = ece.recomputed-ece.stored
    rows=[]
    for (scenario, model), group in ece.groupby(["scenario", "model_name"]):
        rows.append({"scenario": scenario, "model": model, "affected_jobs": len(group), "old_ece_mean": group.stored.mean(),
                     "corrected_ece_mean": group.recomputed.mean(), "mean_absolute_correction": group.correction.abs().mean(),
                     "max_correction": group.correction.abs().max(), "probability_representation": "deterministic_one_hot",
                     "interpretation_warning": "Hard fallback is not a calibrated probability model."})
    return pd.DataFrame(rows)
