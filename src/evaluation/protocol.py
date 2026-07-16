"""Scientific protocol V2 guards, fold manifests, and record-level results.

This module is deliberately model-agnostic.  New model runners must consume
the shared outer-fold manifest and emit one prediction row per outer-validation
record so runs can be compared without touching the observed legacy holdout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.config import ROOT_DIR


PROTOCOL_ROOT = ROOT_DIR / "artifacts" / "student_mat" / "development_splits"
DEFAULT_FOLD_MANIFEST_PATH = PROTOCOL_ROOT / "student_mat_development_outer_folds.json"
LEGACY_MANIFEST_PATH = ROOT_DIR / "artifacts" / "archive" / "student_mat" / "legacy_dataset" / "legacy_manifest.json"
SCENARIO_CONFIG_PATHS = {
    "pre_assessment": ROOT_DIR / "config" / "features_pre_assessment.yaml",
    "early_warning": ROOT_DIR / "config" / "features_early_warning.yaml",
    "late_stage": ROOT_DIR / "config" / "features_late_stage.yaml",
}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def semantic_checksum(payload: dict[str, Any], *, checksum_key: str = "manifest_checksum") -> str:
    content = {key: value for key, value in payload.items() if key != checksum_key}
    return hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record_identity(dataset_version_id: int, source_row_number: int) -> str:
    """Stable source lineage key, not a bare mutable dataframe index."""
    return f"student-mat:dataset-version:{int(dataset_version_id)}:source-row:{int(source_row_number)}"


# Neural softmax output is produced in float32 before record-level values are
# serialized as float64.  1e-6 is a strict numerical tolerance for that path;
# it remains three orders of magnitude tighter than the rejected 0.001 defect.
PROBABILITY_TOLERANCE = 1e-6


def hard_label_probabilities(predicted_labels: Iterable[int]) -> np.ndarray:
    """Deterministic probabilities for rules without a probabilistic model."""
    labels = np.asarray(list(predicted_labels), dtype=int)
    if np.any((labels < 0) | (labels > 2)):
        raise ValueError("Predicted labels must be Low/Medium/High encoded as 0/1/2.")
    return np.eye(3, dtype=np.float64)[labels]


def validate_probability_matrix(
    probabilities: np.ndarray,
    predicted_labels: Iterable[int] | None = None,
    *,
    tolerance: float = PROBABILITY_TOLERANCE,
) -> None:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Probability matrix must have shape (n_records, 3) in Low/Medium/High order.")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Probabilities must be finite values in [0, 1].")
    sum_errors = np.abs(values.sum(axis=1) - 1.0)
    max_sum_error = float(np.max(sum_errors)) if len(sum_errors) else 0.0
    if max_sum_error > tolerance:
        raise ValueError(
            "Probability rows must sum to 1 within the strict tolerance "
            f"({tolerance:g}); observed maximum error={max_sum_error:.17g}."
        )
    if predicted_labels is not None and not np.array_equal(values.argmax(axis=1), np.asarray(list(predicted_labels), dtype=int)):
        raise ValueError("Predicted labels must equal argmax(probabilities).")


def load_json_yaml(path: Path) -> dict[str, Any]:
    """Load our YAML files, intentionally encoded as JSON (a YAML subset)."""
    return json.loads(path.read_text(encoding="utf-8"))


def allowed_features_for_scenario(scenario: str) -> set[str]:
    if scenario not in SCENARIO_CONFIG_PATHS:
        raise ValueError(f"Unknown scenario '{scenario}'.")
    payload = load_json_yaml(SCENARIO_CONFIG_PATHS[scenario])
    return set(payload["allowed_features"])


def _is_g3_or_derived(feature: str) -> bool:
    normalized = feature.lower().replace("-", "_")
    return normalized == "g3" or normalized.startswith("g3_") or "g3" in normalized.split("_")


def validate_scenario_features(
    feature_names: Iterable[str],
    scenario: str,
    *,
    override_reason: str | None = None,
    override_log_path: Path | None = None,
) -> None:
    """Reject unavailable, unknown, target, and target-derived model features.

    An override is intentionally not supported by this function: availability
    exceptions require a separately approved data-contract update, rather than
    silently weakening a training invocation.
    """
    features = [str(feature) for feature in feature_names]
    forbidden_target = [feature for feature in features if _is_g3_or_derived(feature)]
    if forbidden_target:
        raise ValueError(f"G3 or a G3-derived feature is forbidden: {forbidden_target}")
    allowed = allowed_features_for_scenario(scenario)
    disallowed = sorted(set(features) - allowed)
    if disallowed:
        if override_reason or override_log_path:
            raise ValueError("Scenario overrides are not implemented; update the availability contract first.")
        raise ValueError(f"Features are not allowed for {scenario}: {disallowed}")


def load_fold_manifest(path: Path = DEFAULT_FOLD_MANIFEST_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_fold_manifest(payload)
    return payload


def validate_fold_manifest(manifest: dict[str, Any]) -> None:
    expected = semantic_checksum(manifest)
    if manifest.get("manifest_checksum") != expected:
        raise ValueError("Fold manifest semantic checksum mismatch.")
    n_folds = int(manifest["outer_folds"])
    assignments = manifest["assignments"]
    development = manifest["development_records"]
    development_ids = {row["source_record_identity"] for row in development}
    expected_ids = set(development_ids)
    by_fold: dict[int, dict[str, set[str]]] = {
        fold: {"train": set(), "validation": set()} for fold in range(n_folds)
    }
    validation_counts = {identity: 0 for identity in development_ids}
    for row in assignments:
        fold = int(row["outer_fold"])
        role = row["outer_role"]
        identity = row["source_record_identity"]
        if fold not in by_fold or role not in {"train", "validation"} or identity not in development_ids:
            raise ValueError("Fold manifest contains an invalid assignment.")
        by_fold[fold][role].add(identity)
        if role == "validation":
            validation_counts[identity] += 1
    for fold, roles in by_fold.items():
        if roles["train"] & roles["validation"]:
            raise ValueError(f"Outer fold {fold} has train/validation overlap.")
        if roles["train"] | roles["validation"] != expected_ids:
            raise ValueError(f"Outer fold {fold} does not cover development records.")
    if any(count != 1 for count in validation_counts.values()):
        raise ValueError("Each development record must occur in exactly one outer-validation fold.")
    if len(development) != 316:
        raise ValueError("Protocol V2 is pinned to the 316-record legacy development cohort.")


def outer_folds_from_manifest(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    source_column: str = "__source_row_number",
) -> list[tuple[np.ndarray, np.ndarray]]:
    dataset_version_id = int(manifest["dataset_version_id"])
    by_identity = {
        source_record_identity(dataset_version_id, int(row)): position
        for position, row in enumerate(frame[source_column].astype(int).tolist())
    }
    expected = {record["source_record_identity"] for record in manifest["development_records"]}
    if set(by_identity) != expected:
        raise ValueError("Input records do not exactly match the V2 development cohort.")
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(int(manifest["outer_folds"])):
        rows = [row for row in manifest["assignments"] if int(row["outer_fold"]) == fold]
        train = [by_identity[row["source_record_identity"]] for row in rows if row["outer_role"] == "train"]
        validation = [by_identity[row["source_record_identity"]] for row in rows if row["outer_role"] == "validation"]
        folds.append((np.asarray(sorted(train), dtype=int), np.asarray(sorted(validation), dtype=int)))
    return folds


def assert_no_legacy_records(record_ids: Iterable[str], legacy_manifest_path: Path = LEGACY_MANIFEST_PATH) -> None:
    legacy = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
    observed = set(legacy["current_79_record_ids"])
    overlap = observed & set(record_ids)
    if overlap:
        raise ValueError("legacy_heldout_observed records are forbidden from V2 model selection.")


def build_fold_prediction_rows(
    *,
    run_metadata: dict[str, Any],
    manifest: dict[str, Any],
    outer_fold: int,
    record_ids: Iterable[str],
    y_true: Iterable[int],
    y_pred: Iterable[int],
    probabilities: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    required = {"model_name", "scenario", "feature_set_id", "training_seed", "hyperparameter_trial_id", "config_checksum", "code_commit", "dataset_checksum"}
    missing = required - set(run_metadata)
    if missing:
        raise ValueError(f"Run metadata missing required fields: {sorted(missing)}")
    if run_metadata["fold_manifest_checksum"] != manifest["manifest_checksum"]:
        raise ValueError("Run metadata fold checksum does not match fold manifest.")
    validation_ids = {
        row["source_record_identity"]
        for row in manifest["assignments"]
        if int(row["outer_fold"]) == int(outer_fold) and row["outer_role"] == "validation"
    }
    ids, truth, predicted = list(record_ids), list(y_true), list(y_pred)
    if set(ids) != validation_ids or len(ids) != len(validation_ids):
        raise ValueError("Predictions must contain exactly this outer fold's validation records.")
    rows = []
    for index, record_id in enumerate(ids):
        row = {**run_metadata, "outer_fold": int(outer_fold), "record_id": record_id, "true_label": int(truth[index]), "predicted_label": int(predicted[index])}
        if probabilities is not None:
            row.update({f"prob_{label}": float(probabilities[index, label]) for label in range(probabilities.shape[1])})
        rows.append(row)
    return rows


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int], probabilities: np.ndarray | None = None) -> dict[str, Any]:
    truth, predicted = np.asarray(list(y_true), dtype=int), np.asarray(list(y_pred), dtype=int)
    _, _, per_class_f1, support = precision_recall_fscore_support(truth, predicted, labels=[0, 1, 2], zero_division=0)
    payload: dict[str, Any] = {
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(truth, predicted, weights="quadratic")),
        "ordinal_mae": float(np.mean(np.abs(truth - predicted))),
        "one_step_errors": int(np.sum(np.abs(truth - predicted) == 1)),
        "two_step_errors": int(np.sum(np.abs(truth - predicted) >= 2)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1, 2]).tolist(),
        "per_class_f1": {str(label): {"f1": float(per_class_f1[label]), "support": int(support[label])} for label in range(3)},
    }
    if probabilities is not None:
        one_hot = np.eye(probabilities.shape[1])[truth]
        payload["brier_score"] = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    return payload
