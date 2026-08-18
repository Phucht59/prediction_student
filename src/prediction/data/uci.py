"""Frozen UCI Combined binary-risk adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import assert_binary_target, uci_risk_target
from .common import UnifiedHybridData


UCI_QUASI_IDENTITY_FIELDS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "nursery", "internet",
]
UCI_CATEGORICAL_CONTEXT = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason",
    "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher",
    "internet", "romantic", "subject",
]
UCI_NUMERIC_CONTEXT = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
    "freetime", "goout", "Dalc", "Walc", "health",
]
UCI_FORBIDDEN_PREDICTORS = ["G1", "G2", "G3", "absences"]


def _stable_id(*parts: Any, length: int = 24) -> str:
    import hashlib
    return hashlib.sha256("|".join(str(x).strip() for x in parts).encode()).hexdigest()[:length]


def load_raw_uci(mat_path: str | Path = "data/raw/student-mat.csv", por_path: str | Path = "data/raw/student-por.csv"):
    mat_df = pd.read_csv(mat_path, sep=";")
    por_df = pd.read_csv(por_path, sep=";")
    if len(mat_df) != 395 or len(por_df) != 649:
        raise ValueError(f"UCI Combined requires 395 MAT + 649 POR, got {len(mat_df)} + {len(por_df)}")
    if set(mat_df.columns) != set(por_df.columns):
        raise ValueError("MAT/POR column mismatch")
    mat_df, por_df = mat_df.copy(), por_df.copy()
    mat_df["subject"], por_df["subject"] = "math", "portuguese"
    mat_df["source_row_idx"], por_df["source_row_idx"] = np.arange(len(mat_df)), np.arange(len(por_df))
    return mat_df, por_df, pd.concat([mat_df, por_df], ignore_index=True)


def build_uci_combined(mat_path: str | Path = "data/raw/student-mat.csv", por_path: str | Path = "data/raw/student-por.csv") -> tuple[pd.DataFrame, dict[str, Any]]:
    mat_df, por_df, frame = load_raw_uci(mat_path, por_path)
    frame = frame.copy()
    frame["target"] = uci_risk_target(frame["G3"].to_numpy())
    frame["identity_signature"] = frame.apply(lambda row: "|".join(str(row[c]).strip() for c in UCI_QUASI_IDENTITY_FIELDS), axis=1)
    frame["global_student_group"] = frame["identity_signature"].map(lambda x: _stable_id("uci_group", x, length=20))
    frame["record_id"] = frame.apply(lambda row: _stable_id("uci_record", row.subject, row.source_row_idx, row.identity_signature), axis=1)
    assert_binary_target(frame["target"], name="UCI target")
    summary = {
        "dataset": "uci_combined", "total_mat_records": len(mat_df), "total_por_records": len(por_df),
        "total_combined_records": len(frame), "unique_global_student_groups": int(frame.global_student_group.nunique()),
        "cross_subject_groups": len(
            set(frame.loc[frame.subject == "math", "identity_signature"])
            & set(frame.loc[frame.subject == "portuguese", "identity_signature"])
        ),
        "target_rule": "G3 < 10",
    }
    return frame, summary


def build_uci_stage_view(uci_df: pd.DataFrame, stage: str, static: np.ndarray | None = None) -> UnifiedHybridData:
    if stage not in {"S0", "S1", "S2"}:
        raise ValueError("stage must be S0, S1, or S2")
    n = len(uci_df)
    temporal = np.zeros((n, 2, 1), dtype=np.float32)
    mask = np.zeros((n, 2), dtype=bool)
    aggregate = np.zeros((n, 5), dtype=np.float32)
    available = np.zeros(n, dtype=np.int8)
    g1, g2 = uci_df.G1.to_numpy(np.float32) / 20.0, uci_df.G2.to_numpy(np.float32) / 20.0
    if stage in {"S1", "S2"}:
        temporal[:, 0, 0], mask[:, 0] = g1, True
        aggregate[:, 0], aggregate[:, 1], aggregate[:, 2], available[:] = g1, g1, 0.5, 1
    if stage == "S2":
        temporal[:, 1, 0], mask[:, 1] = g2, True
        aggregate[:, 0], aggregate[:, 1], aggregate[:, 2], aggregate[:, 3], aggregate[:, 4] = g2, (g1 + g2) / 2, 1.0, g2 - g1, 1.0
    view = UnifiedHybridData(
        static=np.zeros((n, 0), dtype=np.float32) if static is None else np.asarray(static, dtype=np.float32),
        temporal=temporal, temporal_mask=mask, lengths=mask.sum(1).astype(np.int64), aggregate=aggregate,
        aggregate_available=available, progress=np.full(n, {"S0": 0.0, "S1": 0.5, "S2": 1.0}[stage], np.float32),
        target=uci_df.target.to_numpy(np.int64), record_id=uci_df.record_id.to_numpy(str),
        group_id=uci_df.global_student_group.to_numpy(str), metadata={"dataset": "uci_combined", "stage": stage, "target_rule": UCI_RISK_RULE},
    )
    view.validate()
    return view


def verify_group_disjoint(train: pd.DataFrame, test: pd.DataFrame, group_col: str = "global_student_group") -> None:
    overlap = set(train[group_col].astype(str)) & set(test[group_col].astype(str))
    if overlap:
        raise ValueError(f"UCI grouped split leakage: {len(overlap)} groups overlap")


from ..contracts import UCI_RISK_RULE

__all__ = ["load_raw_uci", "build_uci_combined", "build_uci_stage_view", "verify_group_disjoint", "UCI_FORBIDDEN_PREDICTORS"]
