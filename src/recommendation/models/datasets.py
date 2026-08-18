"""Build per-action Phase 8 training tables from frozen Phase 7 silver labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..weak_supervision.matrix import FINAL_ACTIONS
from .features import ACTION_TO_KEY, APPROVED_FEATURES, IDENTITY_COLUMNS, METADATA_COLUMNS, encode_state_features

TRAINING_STATUSES = frozenset({"VALID", "REVIEW"})


def eligible_training_mask(silver: pd.DataFrame) -> pd.Series:
    status_ok = silver["silver_status"].isin(TRAINING_STATUSES)
    target = pd.to_numeric(silver["expected_relevance"], errors="coerce")
    finite = target.notna() & np.isfinite(target.to_numpy(dtype=float))
    in_range = target.between(0, 3)
    return status_ok & finite & in_range


def build_action_training(
    silver: pd.DataFrame,
    panel_a: pd.DataFrame,
    action_id: str,
    *,
    panel_b_ids: set[str],
) -> pd.DataFrame:
    rows = silver[silver["action_id"] == action_id].copy()
    rows["case_id"] = rows["case_id"].astype(str)
    if set(rows["case_id"]) & panel_b_ids:
        raise ValueError(f"{action_id} silver labels overlap Panel B")
    no_evidence = rows["silver_status"].eq("NO_WEAK_EVIDENCE")
    if no_evidence.any() and rows.loc[no_evidence, "expected_relevance"].notna().any():
        # NO_WEAK_EVIDENCE may store NaN targets; never coerce them to 0 here.
        pass
    keep = rows.loc[eligible_training_mask(rows)].copy()
    if keep.empty:
        raise ValueError(f"{action_id} has no eligible training rows")
    if keep["silver_status"].eq("NO_WEAK_EVIDENCE").any():
        raise ValueError("NO_WEAK_EVIDENCE rows must not enter training")
    state = panel_a.copy()
    state["case_id"] = state["case_id"].astype(str)
    if set(state["case_id"]) & panel_b_ids:
        raise ValueError("Panel A artifact overlaps Panel B")
    features = encode_state_features(state)
    features["case_id"] = state["case_id"].to_numpy()
    merged = keep.merge(features, on="case_id", how="inner", validate="one_to_one")
    if len(merged) != len(keep):
        raise ValueError(f"{action_id} training join lost silver rows")
    if set(merged["case_id"]) & panel_b_ids:
        raise ValueError(f"{action_id} training set contains Panel B")
    output = pd.DataFrame({"case_id": merged["case_id"].astype(str), "action_id": action_id})
    for column in APPROVED_FEATURES:
        output[column] = merged[column].to_numpy()
    output["expected_relevance"] = merged["expected_relevance"].astype(float)
    output["silver_confidence"] = merged["confidence"].astype(float)
    output["silver_entropy"] = merged["entropy"].astype(float)
    output["silver_status"] = merged["silver_status"].astype(str)
    output["aggregator_type"] = merged["aggregator_type"].astype(str)
    output["label_model_version"] = merged["label_model_version"].astype(str)
    output["phase6_source_manifest_version"] = merged["phase6_source_manifest_version"].astype(str)
    if output["silver_confidence"].isna().any() or (output["silver_confidence"] <= 0).any():
        raise ValueError(f"{action_id} has invalid sample-weight confidence")
    if set(output.columns) & IDENTITY_COLUMNS - {"case_id"}:
        raise ValueError("training table contains identity fields beyond case_id")
    return output.sort_values("case_id").reset_index(drop=True)


def build_all_training_sets(silver: pd.DataFrame, panel_a: pd.DataFrame, panel_b_ids: set[str], output_dir: Path) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for action_id in FINAL_ACTIONS:
        table = build_action_training(silver, panel_a, action_id, panel_b_ids=panel_b_ids)
        key = ACTION_TO_KEY[action_id]
        table.to_parquet(output_dir / f"{key}_training.parquet", index=False)
        tables[action_id] = table
    return tables


def feature_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = frame[list(APPROVED_FEATURES)].to_numpy(dtype=float)
    target = frame["expected_relevance"].to_numpy(dtype=float)
    weights = frame["silver_confidence"].to_numpy(dtype=float)
    return features, target, weights
