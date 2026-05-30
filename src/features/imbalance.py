from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, BorderlineSMOTE, RandomOverSampler
from sklearn.base import clone


RESAMPLING_STRATEGIES = ("none", "random_over", "smote", "borderline_smote", "adasyn")


def class_distribution(y) -> dict[int, int]:
    counts = Counter(np.asarray(y).tolist())
    return {class_id: int(counts.get(class_id, 0)) for class_id in (0, 1, 2)}


def _minority_count(y_train) -> int:
    distribution = class_distribution(y_train)
    present_counts = [count for count in distribution.values() if count > 0]
    if not present_counts:
        raise ValueError("y_train has no class samples.")
    return min(present_counts)


def _safe_neighbor_count(y_train, default: int = 5) -> int:
    min_class_count = _minority_count(y_train)
    if min_class_count < 2:
        raise ValueError(
            "SMOTE/ADASYN/BorderlineSMOTE require at least 2 samples in the minority class; "
            f"minority class count is {min_class_count}."
        )
    return min(default, min_class_count - 1)


def get_resampler(strategy: str, seed: int = 42, y_train=None):
    if strategy not in RESAMPLING_STRATEGIES:
        raise ValueError(f"Unknown imbalance strategy '{strategy}'.")
    if strategy == "none":
        return None
    if strategy == "random_over":
        return RandomOverSampler(random_state=seed)

    if y_train is None:
        raise ValueError(f"y_train is required to configure {strategy}.")

    k_neighbors = _safe_neighbor_count(y_train)
    if strategy == "smote":
        return SMOTE(random_state=seed, k_neighbors=k_neighbors)
    if strategy == "borderline_smote":
        m_neighbors = min(10, max(1, _minority_count(y_train) - 1))
        return BorderlineSMOTE(
            random_state=seed,
            k_neighbors=k_neighbors,
            m_neighbors=m_neighbors,
        )
    if strategy == "adasyn":
        return ADASYN(random_state=seed, n_neighbors=k_neighbors)

    raise ValueError(f"Unhandled imbalance strategy '{strategy}'.")


def resample_train_data(X_train, y_train, strategy: str, seed: int = 42):
    before_distribution = class_distribution(y_train)
    resampler = get_resampler(strategy, seed=seed, y_train=y_train)
    if resampler is None:
        return X_train, y_train, before_distribution, before_distribution

    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
    after_distribution = class_distribution(y_resampled)
    return X_resampled, y_resampled, before_distribution, after_distribution


def resample_raw_mixed_train_data(
    X_train_raw: pd.DataFrame,
    y_train,
    categorical_columns: list[str],
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, dict[int, int], dict[int, int]]:
    """Apply SMOTENC before one-hot encoding on a raw mixed-type training split."""
    if not isinstance(X_train_raw, pd.DataFrame):
        raise TypeError("SMOTENC raw resampling expects X_train_raw to be a pandas DataFrame.")
    missing = [column for column in categorical_columns if column not in X_train_raw.columns]
    if missing:
        raise ValueError(f"SMOTENC categorical columns missing from raw training data: {missing}")
    if not categorical_columns:
        raise ValueError("SMOTENC requires at least one categorical column; use SMOTE for all-numeric data.")

    before_distribution = class_distribution(y_train)
    categorical_indices = [X_train_raw.columns.get_loc(column) for column in categorical_columns]
    k_neighbors = _safe_neighbor_count(y_train)
    sampler = SMOTENC(
        categorical_features=categorical_indices,
        random_state=seed,
        k_neighbors=k_neighbors,
    )
    X_resampled, y_resampled = sampler.fit_resample(X_train_raw, y_train)
    if isinstance(X_resampled, pd.DataFrame):
        X_resampled_df = X_resampled.copy()
    else:
        X_resampled_df = pd.DataFrame(X_resampled, columns=X_train_raw.columns)

    numeric_columns = [column for column in X_train_raw.columns if column not in categorical_columns]
    for column in numeric_columns:
        X_resampled_df[column] = pd.to_numeric(X_resampled_df[column], errors="raise")
    for column in categorical_columns:
        X_resampled_df[column] = X_resampled_df[column].astype(X_train_raw[column].dtype, copy=False)

    y_resampled = np.asarray(y_resampled, dtype=np.int64)
    after_distribution = class_distribution(y_resampled)
    return X_resampled_df.reset_index(drop=True), y_resampled, before_distribution, after_distribution


def supports_class_weight(model) -> bool:
    try:
        return "class_weight" in model.get_params()
    except Exception:
        return False


def clone_with_class_weight(model):
    if not supports_class_weight(model):
        raise ValueError(f"Model {type(model).__name__} does not support class_weight.")
    weighted_model = clone(model)
    weighted_model.set_params(class_weight="balanced")
    return weighted_model
