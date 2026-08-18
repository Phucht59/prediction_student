"""Construct and validate Phase 7 silver labels."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .matrix import FINAL_ACTIONS, FINAL_TO_LEGACY, FORBIDDEN_COLUMNS

SILVER_COLUMNS = (
    "case_id",
    "action_id",
    "p_r0",
    "p_r1",
    "p_r2",
    "p_r3",
    "expected_relevance",
    "hard_label",
    "confidence",
    "entropy",
    "majority_label",
    "aggregator_majority_same",
    "feasibility_status",
    "silver_status",
    "aggregator_type",
    "label_model_version",
    "phase6_source_manifest_version",
)
PROBABILITY_COLUMNS = ("p_r0", "p_r1", "p_r2", "p_r3")
VALID_STATUSES = frozenset({"VALID", "REVIEW"})
SILVER_STATUS_DOMAIN = frozenset({"VALID", "NO_WEAK_EVIDENCE", "REVIEW"})
AGGREGATOR_DOMAIN = frozenset({"SNORKEL", "TWO_SOURCE_CONSENSUS"})
FEASIBILITY_DOMAIN = frozenset({"FEASIBLE", "INFEASIBLE", "UNKNOWN"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def attach_feasibility(frame: pd.DataFrame, feasibility_path: Path, panel_a_ids: set[str]) -> pd.DataFrame:
    feasibility = pd.read_parquet(feasibility_path, columns=["case_id", "action_id", "feasibility_status"])
    if set(feasibility.columns) & FORBIDDEN_COLUMNS:
        raise ValueError("feasibility artifact contains forbidden leakage columns")
    feasibility = feasibility.copy()
    feasibility["case_id"] = feasibility["case_id"].astype(str)
    feasibility = feasibility[feasibility["case_id"].isin(panel_a_ids)].copy()
    reverse = {legacy: action for action, legacy in FINAL_TO_LEGACY.items()}
    feasibility["action_id"] = feasibility["action_id"].map(lambda value: reverse.get(str(value), str(value)))
    feasibility = feasibility[feasibility["action_id"].isin(FINAL_ACTIONS)]
    if feasibility.duplicated(["case_id", "action_id"]).any():
        raise ValueError("feasibility artifact has duplicate case-action rows for Panel A")
    merged = frame.merge(feasibility, on=["case_id", "action_id"], how="left", validate="one_to_one")
    if merged["feasibility_status"].isna().any():
        raise ValueError("silver rows are missing feasibility_status")
    if not set(merged["feasibility_status"]).issubset(FEASIBILITY_DOMAIN):
        raise ValueError("invalid feasibility_status values on silver rows")
    return merged


def apply_action_review_status(frame: pd.DataFrame, review_actions: set[str]) -> pd.DataFrame:
    output = frame.copy()
    evidence = output["silver_status"].isin(VALID_STATUSES)
    output.loc[evidence & output["action_id"].isin(review_actions), "silver_status"] = "REVIEW"
    output.loc[evidence & ~output["action_id"].isin(review_actions), "silver_status"] = "VALID"
    return output


def validate_silver(frame: pd.DataFrame, panel_a_ids: set[str], panel_b_ids: set[str]) -> None:
    required = set(SILVER_COLUMNS)
    if not required.issubset(frame.columns):
        raise ValueError(f"silver missing columns: {sorted(required - set(frame.columns))}")
    if len(frame) != 2500 or frame.duplicated(["case_id", "action_id"]).any():
        raise ValueError("silver must contain 2,500 unique case-action rows")
    if set(frame["action_id"]) != set(FINAL_ACTIONS) or any(len(group) != 500 for _, group in frame.groupby("action_id")):
        raise ValueError("silver action coverage gate failed")
    case_ids = set(frame["case_id"].astype(str))
    if case_ids != panel_a_ids:
        raise ValueError("silver case identity gate failed")
    overlap = case_ids & panel_b_ids
    if overlap:
        raise ValueError(f"silver contains Panel-B case_ids: {len(overlap)}")
    if set(frame.columns) & FORBIDDEN_COLUMNS:
        raise ValueError("silver contains forbidden leakage columns")
    if not set(frame["silver_status"]).issubset(SILVER_STATUS_DOMAIN):
        raise ValueError("invalid silver_status values")
    if not set(frame["aggregator_type"]).issubset(AGGREGATOR_DOMAIN):
        raise ValueError("invalid aggregator_type values")
    if not set(frame["feasibility_status"]).issubset(FEASIBILITY_DOMAIN):
        raise ValueError("invalid feasibility_status values")

    evidence = frame[frame["silver_status"].isin(VALID_STATUSES)].copy()
    empty = frame[frame["silver_status"].eq("NO_WEAK_EVIDENCE")].copy()
    if evidence.empty and empty.empty:
        raise ValueError("silver contains no classifiable rows")

    if not empty.empty:
        if not empty[list(PROBABILITY_COLUMNS)].isna().all().all():
            raise ValueError("NO_WEAK_EVIDENCE rows must not fabricate probabilities")
        if not empty[["expected_relevance", "confidence", "entropy"]].isna().all().all():
            raise ValueError("NO_WEAK_EVIDENCE rows must not fabricate summary scores")
        if empty["hard_label"].notna().any():
            raise ValueError("NO_WEAK_EVIDENCE rows must not invent a hard label")
        if (empty["hard_label"].fillna(-999) == 0).any():
            raise ValueError("all-abstain rows must not be mapped to class 0")

    if evidence.empty:
        return
    probabilities = evidence[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("VALID/REVIEW silver probabilities contain non-finite values")
    if (probabilities < -1e-8).any() or (probabilities > 1 + 1e-8).any():
        raise ValueError("VALID/REVIEW silver probabilities are outside [0,1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("VALID/REVIEW silver probabilities do not sum to one")
    if not evidence["expected_relevance"].between(0, 3).all():
        raise ValueError("expected relevance outside [0,3]")
    if not evidence["hard_label"].isin([0, 1, 2, 3]).all():
        raise ValueError("invalid silver hard label")
    if not evidence["confidence"].between(0, 1).all():
        raise ValueError("confidence outside [0,1]")
    entropy = evidence["entropy"].to_numpy(dtype=float)
    if (~np.isfinite(entropy)).any() or (entropy < -1e-12).any():
        raise ValueError("entropy must be finite and >= 0")
    if not evidence["majority_label"].isin([-1, 0, 1, 2, 3]).all():
        raise ValueError("invalid majority label")
    reconstructed = (
        0 * evidence["p_r0"] + 1 * evidence["p_r1"] + 2 * evidence["p_r2"] + 3 * evidence["p_r3"]
    )
    if not np.allclose(reconstructed.to_numpy(dtype=float), evidence["expected_relevance"].to_numpy(dtype=float), atol=1e-8):
        raise ValueError("expected_relevance does not match the probability-weighted class index")
