"""Deterministic group-safe stratified cross-validation split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class SplitManifest:
    """Contains outer and inner fold assignments for records."""

    outer_assignments: pd.DataFrame  # record_id, group_id, outer_fold
    inner_assignments: pd.DataFrame  # record_id, group_id, outer_fold, inner_fold


def create_group_stratified_splits(
    records: pd.DataFrame,
    record_id_col: str,
    group_id_col: str,
    target_col: str,
    n_outer_splits: int = 5,
    n_inner_splits: int = 3,
    base_seed: int = 42,
) -> SplitManifest:
    """Generate deterministic outer and nested inner group-stratified fold assignments.
    
    Guarantees:
    - Group disjointness: same student group never appears across train and test/val.
    - Outer test records never appear in inner fold tuning sets.
    """
    df = records.copy().reset_index(drop=True)
    
    outer_sgkf = StratifiedGroupKFold(
        n_splits=n_outer_splits, shuffle=True, random_state=base_seed
    )

    df["outer_fold"] = -1
    y = df[target_col].values
    groups = df[group_id_col].values

    for fold_idx, (train_idx, test_idx) in enumerate(outer_sgkf.split(df, y, groups)):
        df.loc[test_idx, "outer_fold"] = fold_idx

    if (df["outer_fold"] < 0).any():
        raise RuntimeError("Some records were not assigned an outer fold")

    outer_df = df[[record_id_col, group_id_col, "outer_fold"]].copy()

    inner_records = []
    for outer_fold_idx in range(n_outer_splits):
        train_mask = df["outer_fold"] != outer_fold_idx
        train_subset = df[train_mask].copy().reset_index(drop=True)

        inner_seed = base_seed + outer_fold_idx
        inner_sgkf = StratifiedGroupKFold(
            n_splits=n_inner_splits, shuffle=True, random_state=inner_seed
        )

        train_subset["inner_fold"] = -1
        y_inner = train_subset[target_col].values
        groups_inner = train_subset[group_id_col].values

        for inner_idx, (in_train_idx, in_val_idx) in enumerate(
            inner_sgkf.split(train_subset, y_inner, groups_inner)
        ):
            train_subset.loc[in_val_idx, "inner_fold"] = inner_idx

        train_subset["outer_fold"] = outer_fold_idx
        inner_records.append(
            train_subset[[record_id_col, group_id_col, "outer_fold", "inner_fold"]]
        )

    inner_df = pd.concat(inner_records, ignore_index=True)

    # Validate split integrity immediately
    verify_split_disjointness(outer_df, fold_col="outer_fold", group_col=group_id_col)
    verify_inner_no_outer_test(outer_df, inner_df, record_id_col=record_id_col)
    verify_inner_group_disjointness(inner_df, group_col=group_id_col, record_id_col=record_id_col)

    return SplitManifest(outer_assignments=outer_df, inner_assignments=inner_df)


def verify_split_disjointness(
    df: pd.DataFrame, fold_col: str, group_col: str
) -> None:
    """Ensure no group appears in multiple test folds or both train and test."""
    folds = sorted(df[fold_col].unique())
    for f in folds:
        test_groups = set(df[df[fold_col] == f][group_col])
        train_groups = set(df[df[fold_col] != f][group_col])
        overlap = test_groups & train_groups
        if overlap:
            raise ValueError(
                f"Group leakage detected in fold {f}: {len(overlap)} groups overlap between train and test"
            )


def verify_inner_no_outer_test(
    outer_df: pd.DataFrame, inner_df: pd.DataFrame, record_id_col: str = "record_id"
) -> None:
    """Ensure outer test records never participate in inner fold splits for that outer fold."""
    for outer_f in outer_df["outer_fold"].unique():
        outer_test_ids = set(outer_df[outer_df["outer_fold"] == outer_f][record_id_col])
        inner_fold_ids = set(inner_df[inner_df["outer_fold"] == outer_f][record_id_col])
        overlap = outer_test_ids & inner_fold_ids
        if overlap:
            raise ValueError(
                f"Outer test records present in inner split for outer fold {outer_f}: {len(overlap)} overlapping records"
            )


def verify_inner_group_disjointness(
    inner_df: pd.DataFrame,
    group_col: str,
    record_id_col: str = "record_id",
) -> None:
    """Ensure inner validation groups never overlap with inner train groups within any outer fold."""
    for outer_f in sorted(inner_df["outer_fold"].unique()):
        sub_df = inner_df[inner_df["outer_fold"] == outer_f]
        # Assert each record in outer-train receives exactly one inner fold
        if sub_df[record_id_col].duplicated().any():
            raise ValueError(f"Duplicate record IDs found in inner fold for outer fold {outer_f}")

        for inner_f in sorted(sub_df["inner_fold"].unique()):
            val_groups = set(sub_df[sub_df["inner_fold"] == inner_f][group_col])
            train_groups = set(sub_df[sub_df["inner_fold"] != inner_f][group_col])
            overlap = val_groups & train_groups
            if overlap:
                raise ValueError(
                    f"Inner group leakage in outer fold {outer_f}, inner fold {inner_f}: {len(overlap)} overlapping groups"
                )
