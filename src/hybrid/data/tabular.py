"""Deterministic tabular baseline feature builders for UCI and OULAD."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.hybrid.contracts import HybridDataView
from src.hybrid.data.oulad import (
    OULAD_CATEGORICAL_CONTEXT,
    OULAD_NUMERIC_CONTEXT,
)
from src.hybrid.data.uci import (
    UCI_CATEGORICAL_CONTEXT,
    UCI_NUMERIC_CONTEXT,
)


def build_uci_tabular_baseline(
    uci_df: pd.DataFrame,
    stage: str,
) -> pd.DataFrame:
    """Build fair tabular representation for UCI baselines matching observable information per stage."""
    if stage not in {"S0", "S1", "S2"}:
        raise ValueError(f"Unknown stage {stage}")

    feature_cols = UCI_NUMERIC_CONTEXT + UCI_CATEGORICAL_CONTEXT
    tabular_df = uci_df[["record_id", "global_student_group", "target"] + feature_cols].copy()

    g1_vals = (uci_df["G1"].to_numpy(dtype=np.float32) / 20.0)
    g2_vals = (uci_df["G2"].to_numpy(dtype=np.float32) / 20.0)

    if stage == "S0":
        tabular_df["g1_norm"] = 0.0
        tabular_df["g2_norm"] = 0.0
        tabular_df["g1_available"] = 0
        tabular_df["g2_available"] = 0
    elif stage == "S1":
        tabular_df["g1_norm"] = g1_vals
        tabular_df["g2_norm"] = 0.0
        tabular_df["g1_available"] = 1
        tabular_df["g2_available"] = 0
    elif stage == "S2":
        tabular_df["g1_norm"] = g1_vals
        tabular_df["g2_norm"] = g2_vals
        tabular_df["g1_available"] = 1
        tabular_df["g2_available"] = 1

    return tabular_df


def build_oulad_tabular_baseline(
    view: HybridDataView,
    channel_names: list[str],
    context_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extract deterministic summary features per channel across valid observed weeks for ML baselines.
    
    Fast vectorized implementation: sum, mean, std, min, max, last, recent_2week_mean, slope.
    """
    view.validate()
    n_records, max_t, n_channels = view.temporal.shape
    if len(channel_names) != n_channels:
        raise ValueError(f"channel_names length ({len(channel_names)}) != n_channels ({n_channels})")

    records_dict: dict[str, np.ndarray] = {
        "record_id": view.record_id,
        "group_id": view.group_id,
        "target": view.target,
    }

    lengths = view.lengths  # [N]
    lengths_safe = np.maximum(1, lengths)[:, None]  # [N, 1]
    mask_3d = view.mask[:, :, None]  # [N, T, 1]

    # Temporal with padding replaced by nan for min/max/std
    masked_temp = np.where(mask_3d, view.temporal, np.nan)  # [N, T, D]

    # 1. Sum and Mean
    sums = np.sum(view.temporal, axis=1)  # [N, D]
    means = sums / lengths_safe  # [N, D]

    # 2. Min, Max, Std (ignoring NaNs where mask is False)
    # To avoid all-NaN slice warnings, replace all-NaN rows with 0.0
    nan_mask = np.all(np.isnan(masked_temp), axis=1)  # [N, D]
    clean_temp = np.where(nan_mask[:, None, :], 0.0, masked_temp)

    mins = np.nanmin(clean_temp, axis=1)
    maxs = np.nanmax(clean_temp, axis=1)
    stds = np.nanstd(clean_temp, axis=1)

    # 3. Last observed value and recent 2-week mean
    lasts = np.zeros((n_records, n_channels), dtype=np.float32)
    r2_means = np.zeros((n_records, n_channels), dtype=np.float32)
    slopes = np.zeros((n_records, n_channels), dtype=np.float32)

    # Row-indexed vectorization for last & recent 2
    row_idx = np.arange(n_records)
    last_pos = np.maximum(0, lengths - 1)
    lasts = view.temporal[row_idx, last_pos, :]

    # For recent 2: if len >= 2: mean of (T-1, T-2), else last
    prev_pos = np.maximum(0, lengths - 2)
    val_prev = view.temporal[row_idx, prev_pos, :]
    r2_means = np.where((lengths[:, None] >= 2), (lasts + val_prev) / 2.0, lasts)

    # Slopes: for records with lengths >= 2, simple linear slope
    # slope = (T*sum(t*y) - sum(t)*sum(y)) / (T*sum(t^2) - (sum(t))^2)
    t_indices = np.arange(max_t, dtype=np.float32)[None, :, None]  # [1, T, 1]
    t_masked = np.where(mask_3d, t_indices, 0.0)  # [N, T, 1]
    sum_t = np.sum(t_masked, axis=1)  # [N, 1]
    sum_t2 = np.sum(t_masked ** 2, axis=1)  # [N, 1]
    sum_ty = np.sum(t_masked * view.temporal, axis=1)  # [N, D]

    n_t = lengths_safe.astype(np.float32)
    denom = n_t * sum_t2 - sum_t ** 2
    numer = n_t * sum_ty - sum_t * sums
    slopes = np.where((lengths[:, None] >= 2) & (denom > 1e-6), numer / np.maximum(1e-6, denom), 0.0)

    for c_idx, ch_name in enumerate(channel_names):
        records_dict[f"{ch_name}__sum"] = sums[:, c_idx].astype(np.float32)
        records_dict[f"{ch_name}__mean"] = means[:, c_idx].astype(np.float32)
        records_dict[f"{ch_name}__std"] = stds[:, c_idx].astype(np.float32)
        records_dict[f"{ch_name}__min"] = mins[:, c_idx].astype(np.float32)
        records_dict[f"{ch_name}__max"] = maxs[:, c_idx].astype(np.float32)
        records_dict[f"{ch_name}__last"] = lasts[:, c_idx].astype(np.float32)
        records_dict[f"{ch_name}__recent2_mean"] = r2_means[:, c_idx].astype(np.float32)
        records_dict[f"{ch_name}__slope"] = slopes[:, c_idx].astype(np.float32)

    tabular_df = pd.DataFrame(records_dict)

    if context_df is not None:
        allowed_context = [
            c
            for c in (OULAD_NUMERIC_CONTEXT + OULAD_CATEGORICAL_CONTEXT)
            if c in context_df.columns
        ]
        tabular_df = pd.merge(
            tabular_df,
            context_df[["record_id"] + allowed_context],
            on="record_id",
            how="left",
        )

    return tabular_df
