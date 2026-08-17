"""UCI Student Performance Combined dataset loader, student grouping, and stage view builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.hybrid.contracts import HybridDataView
from src.hybrid.data.common import make_deterministic_id

UCI_QUASI_IDENTITY_FIELDS: list[str] = [
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "nursery",
    "internet",
]

UCI_CATEGORICAL_CONTEXT: list[str] = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "subject",
]

UCI_NUMERIC_CONTEXT: list[str] = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
]

UCI_FORBIDDEN_PREDICTORS: list[str] = [
    "G1",  # Temporal only
    "G2",  # Temporal only
    "G3",  # Target only
    "absences",  # Forbidden: non-timestamped total absences
]


def load_raw_uci(
    mat_path: str | Path = "data/raw/student-mat.csv",
    por_path: str | Path = "data/raw/student-por.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and concatenate raw Student-Mat and Student-Por with subject annotation."""
    mat_df = pd.read_csv(mat_path, sep=";")
    por_df = pd.read_csv(por_path, sep=";")

    if len(mat_df) != 395:
        raise ValueError(f"Expected 395 rows for student-mat.csv, got {len(mat_df)}")
    if len(por_df) != 649:
        raise ValueError(f"Expected 649 rows for student-por.csv, got {len(por_df)}")

    # Verify column parity
    if set(mat_df.columns) != set(por_df.columns):
        raise ValueError("Column mismatch between student-mat.csv and student-por.csv")

    mat_df["subject"] = "math"
    por_df["subject"] = "portuguese"

    # Add source row index for deterministic ID generation
    mat_df["source_row_idx"] = np.arange(len(mat_df))
    por_df["source_row_idx"] = np.arange(len(por_df))

    combined_df = pd.concat([mat_df, por_df], ignore_index=True)
    return mat_df, por_df, combined_df


def build_uci_combined(
    mat_path: str | Path = "data/raw/student-mat.csv",
    por_path: str | Path = "data/raw/student-por.csv",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build unified UCI Combined dataframe with binary risk target and student grouping."""
    mat_df, por_df, df = load_raw_uci(mat_path, por_path)

    # 1. Target: risk = 1 if G3 < 10 else 0
    df["target"] = (df["G3"] < 10).astype(np.int64)

    # 2. Quasi-identity signature and global_student_group
    def _create_sig(row: pd.Series) -> str:
        return "|".join(str(row[col]).strip() for col in UCI_QUASI_IDENTITY_FIELDS)

    df["identity_signature"] = df.apply(_create_sig, axis=1)
    df["global_student_group"] = df["identity_signature"].apply(
        lambda sig: make_deterministic_id("uci_group", sig, length=20)
    )

    # 3. Deterministic subject record_id
    df["record_id"] = df.apply(
        lambda row: make_deterministic_id(
            "uci_record", row["subject"], row["source_row_idx"], row["identity_signature"], length=24
        ),
        axis=1,
    )

    # 4. Audit statistics
    mat_sigs = set(df[df["subject"] == "math"]["identity_signature"])
    por_sigs = set(df[df["subject"] == "portuguese"]["identity_signature"])
    cross_subject_sigs = mat_sigs & por_sigs

    mat_counts = df[df["subject"] == "math"]["identity_signature"].value_counts()
    por_counts = df[df["subject"] == "portuguese"]["identity_signature"].value_counts()
    mat_repeats = (mat_counts > 1).sum()
    por_repeats = (por_counts > 1).sum()

    group_sizes = df["global_student_group"].value_counts()

    audit_summary = {
        "total_mat_records": int(len(mat_df)),
        "total_por_records": int(len(por_df)),
        "total_combined_records": int(len(df)),
        "total_risk_records": int(df["target"].sum()),
        "risk_prevalence": float(df["target"].mean()),
        "mat_risk_records": int(df[df["subject"] == "math"]["target"].sum()),
        "mat_risk_prevalence": float(df[df["subject"] == "math"]["target"].mean()),
        "por_risk_records": int(df[df["subject"] == "portuguese"]["target"].sum()),
        "por_risk_prevalence": float(df[df["subject"] == "portuguese"]["target"].mean()),
        "unique_identity_signatures": int(df["identity_signature"].nunique()),
        "unique_global_student_groups": int(df["global_student_group"].nunique()),
        "cross_subject_signatures_count": int(len(cross_subject_sigs)),
        "repeated_signatures_within_mat": int(mat_repeats),
        "repeated_signatures_within_por": int(por_repeats),
        "max_group_size": int(group_sizes.max()),
        "min_group_size": int(group_sizes.min()),
    }

    return df, audit_summary


def build_uci_stage_view(
    uci_df: pd.DataFrame,
    stage: str,
    context_matrix: np.ndarray | None = None,
) -> HybridDataView:
    """Construct HybridDataView for a specific early-warning stage (S0, S1, S2)."""
    if stage not in {"S0", "S1", "S2"}:
        raise ValueError(f"Unknown UCI stage {stage}, must be one of {{'S0', 'S1', 'S2'}}")

    n_records = len(uci_df)
    record_ids = uci_df["record_id"].to_numpy(dtype=str)
    group_ids = uci_df["global_student_group"].to_numpy(dtype=str)
    targets = uci_df["target"].to_numpy(dtype=np.int64)

    # Base sequence [N, 2, 1] normalized by / 20.0
    g1_norm = (uci_df["G1"].to_numpy(dtype=np.float32) / 20.0)[:, None]  # [N, 1]
    g2_norm = (uci_df["G2"].to_numpy(dtype=np.float32) / 20.0)[:, None]  # [N, 1]
    zeros = np.zeros_like(g1_norm)

    temporal = np.zeros((n_records, 2, 1), dtype=np.float32)
    mask = np.zeros((n_records, 2), dtype=bool)

    if stage == "S0":
        # mask = [False, False], lengths = 0
        pass
    elif stage == "S1":
        # mask = [True, False], lengths = 1
        temporal[:, 0, :] = g1_norm
        mask[:, 0] = True
    elif stage == "S2":
        # mask = [True, True], lengths = 2
        temporal[:, 0, :] = g1_norm
        temporal[:, 1, :] = g2_norm
        mask[:, 0] = True
        mask[:, 1] = True

    lengths = np.sum(mask, axis=1).astype(np.int64)

    view = HybridDataView(
        record_id=record_ids,
        group_id=group_ids,
        target=targets,
        temporal=temporal,
        mask=mask,
        lengths=lengths,
        context=context_matrix,
        metadata={"stage": stage, "domain": "uci_combined"},
    )
    view.validate()
    return view
