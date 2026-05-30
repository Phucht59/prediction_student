from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_selection import chi2


def canonicalize_name(value: str) -> str:
    """Normalize raw/processed column names for case-insensitive matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def resolve_columns_by_aliases(
    columns: Iterable[str],
    aliases: Iterable[str],
    *,
    required: bool = True,
    error_label: str = "columns",
) -> list[str]:
    """Resolve desired column aliases to actual dataframe column names."""
    actual_by_key = {canonicalize_name(column): column for column in columns}
    resolved: list[str] = []
    missing: list[str] = []
    for alias in aliases:
        key = canonicalize_name(alias)
        actual = actual_by_key.get(key)
        if actual is None:
            missing.append(str(alias))
        elif actual not in resolved:
            resolved.append(actual)

    if missing and required:
        raise ValueError(f"Missing {error_label}: {missing}")
    return resolved


def feature_refers_to_raw_column(feature_name: str, raw_column: str) -> bool:
    """Return True when a processed feature belongs to a raw column."""
    suffix = str(feature_name).split("__", 1)[-1]
    feature_key = canonicalize_name(suffix)
    column_key = canonicalize_name(raw_column)
    return feature_key == column_key or feature_key.startswith(column_key)


@dataclass(frozen=True)
class FeatureSelectionResult:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    selected_indices: list[int]
    selected_feature_names: list[str]
    metadata: dict


def _pearson_abs_scores(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    X_centered = X - X.mean(axis=0, keepdims=True)
    y_centered = y - y.mean()
    numerator = np.sum(X_centered * y_centered[:, None], axis=0)
    denominator = np.sqrt(np.sum(X_centered**2, axis=0) * np.sum(y_centered**2))
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.abs(numerator / denominator)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def _chi2_scores(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    X = np.asarray(X_train, dtype=np.float64)
    min_values = X.min(axis=0, keepdims=True)
    X_non_negative = np.where(min_values < 0, X - min_values, X)
    scores, _ = chi2(X_non_negative, y_train)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0)
    max_score = float(np.max(scores)) if scores.size else 0.0
    if max_score <= 0:
        return np.zeros_like(scores, dtype=np.float64)
    return scores / max_score


def _forced_feature_indices(
    feature_names: list[str],
    force_raw_columns: Iterable[str] | None,
    force_feature_names: Iterable[str] | None,
) -> set[int]:
    forced: set[int] = set()
    raw_columns = list(force_raw_columns or [])
    feature_name_set = set(force_feature_names or [])
    for index, feature_name in enumerate(feature_names):
        if feature_name in feature_name_set:
            forced.add(index)
            continue
        if any(feature_refers_to_raw_column(feature_name, column) for column in raw_columns):
            forced.add(index)
    return forced


def select_features_supervised(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    max_features: int | None = None,
    min_features: int = 1,
    pearson_threshold: float = 0.3,
    numeric_raw_columns: Iterable[str] | None = None,
    force_raw_columns: Iterable[str] | None = None,
    force_feature_names: Iterable[str] | None = None,
) -> FeatureSelectionResult:
    """Select tabular features with train-only Pearson and Chi-square scores."""
    if X_train.ndim != 2 or X_val.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Feature selection expects 2D train/val/test arrays.")
    if X_train.shape[1] != len(feature_names):
        raise ValueError(
            f"feature_names length {len(feature_names)} does not match X_train width {X_train.shape[1]}."
        )
    if X_val.shape[1] != X_train.shape[1] or X_test.shape[1] != X_train.shape[1]:
        raise ValueError("Feature selection received inconsistent split feature widths.")

    n_features = int(X_train.shape[1])
    if n_features == 0:
        raise ValueError("Cannot select features from an empty feature matrix.")

    if max_features is None or max_features <= 0:
        target_count = n_features
    else:
        target_count = min(int(max_features), n_features)
    target_count = max(int(min_features), target_count)

    pearson_scores = _pearson_abs_scores(X_train, y_train)
    chi2_raw_scores = _chi2_scores(X_train, y_train)
    combined_scores = 0.5 * _normalize_scores(pearson_scores) + 0.5 * _normalize_scores(chi2_raw_scores)

    numeric_raw_columns = list(numeric_raw_columns or [])
    numeric_indices = {
        index
        for index, feature_name in enumerate(feature_names)
        if str(feature_name).startswith("numeric__")
        or any(feature_refers_to_raw_column(feature_name, column) for column in numeric_raw_columns)
    }
    pearson_selected_indices = {
        index
        for index in numeric_indices
        if float(pearson_scores[index]) >= float(pearson_threshold)
    }

    forced_indices = _forced_feature_indices(feature_names, force_raw_columns, force_feature_names)
    forced_indices.update(pearson_selected_indices)
    if len(forced_indices) > target_count:
        target_count = len(forced_indices)

    selected: set[int] = set(forced_indices)
    ranked_indices = np.argsort(-combined_scores, kind="mergesort")
    for index in ranked_indices:
        selected.add(int(index))
        if len(selected) >= target_count:
            break

    selected_indices = sorted(selected)
    selected_feature_names = [feature_names[index] for index in selected_indices]
    top_scores = [
        {
            "feature": feature_names[index],
            "combined_score": float(combined_scores[index]),
            "pearson_abs": float(pearson_scores[index]),
            "chi2": float(chi2_raw_scores[index]),
            "forced": bool(index in forced_indices),
            "selected_by_pearson_threshold": bool(index in pearson_selected_indices),
        }
        for index in ranked_indices[: min(25, len(ranked_indices))]
    ]
    metadata = {
        "method": "pearson_chi2",
        "pearson_threshold": float(pearson_threshold),
        "numeric_raw_columns": numeric_raw_columns,
        "n_features_before": n_features,
        "n_features_after": len(selected_indices),
        "max_features": int(target_count),
        "forced_raw_columns": list(force_raw_columns or []),
        "forced_feature_count": len(forced_indices),
        "pearson_threshold_feature_count": len(pearson_selected_indices),
        "pearson_threshold_feature_names": [
            feature_names[index] for index in sorted(pearson_selected_indices)
        ],
        "selected_feature_names": selected_feature_names,
        "top_scores": top_scores,
    }
    return FeatureSelectionResult(
        X_train=X_train[:, selected_indices].astype("float32"),
        X_val=X_val[:, selected_indices].astype("float32"),
        X_test=X_test[:, selected_indices].astype("float32"),
        selected_indices=selected_indices,
        selected_feature_names=selected_feature_names,
        metadata=metadata,
    )
