from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.studies.common.hashing import stable_record_id


IDENTITY_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "nursery", "internet",
]


def encode_g3(values: pd.Series) -> np.ndarray:
    raw = pd.to_numeric(values, errors="raise").to_numpy()
    if np.any((raw < 0) | (raw > 20)):
        raise ValueError("G3 outside the frozen 0..20 contract")
    return np.where(raw <= 9, 0, np.where(raw <= 14, 1, 2)).astype(int)


def load_student_csv(path: str | Path, dataset_id: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";")
    required = {*IDENTITY_COLUMNS, "G1", "G2", "G3"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if frame.duplicated().any():
        raise ValueError(f"Exact duplicate rows in {dataset_id}")
    frame = frame.copy()
    frame["raw_g3"] = pd.to_numeric(frame["G3"], errors="raise")
    frame["G3"] = encode_g3(frame["G3"])
    frame["source_row_number"] = np.arange(len(frame), dtype=int)
    frame["source_record_id"] = [stable_record_id(dataset_id, int(row)) for row in frame["source_row_number"]]
    return frame


def overlap_membership(mat: pd.DataFrame, por: pd.DataFrame) -> tuple[pd.Series, dict[str, object]]:
    mat_counts = mat.groupby(IDENTITY_COLUMNS, dropna=False).size()
    por_counts = por.groupby(IDENTITY_COLUMNS, dropna=False).size()
    shared = set(mat_counts.index) & set(por_counts.index)
    one_to_one = {key for key in shared if int(mat_counts.loc[key]) == 1 and int(por_counts.loc[key]) == 1}
    ambiguous = shared - one_to_one

    def key_for(row: pd.Series) -> tuple[object, ...]:
        return tuple(row[column] for column in IDENTITY_COLUMNS)

    labels = []
    for _, row in por.iterrows():
        key = key_for(row)
        if key in one_to_one:
            labels.append("conservative_matched")
        elif key in ambiguous:
            labels.append("ambiguous_shared_key")
        else:
            labels.append("conservative_unmatched")
    membership = pd.Series(labels, index=por.index, name="overlap_partition")
    standard_join_rows = len(mat.merge(por, on=IDENTITY_COLUMNS, how="inner", suffixes=("_mat", "_por")))
    audit = {
        "identity_columns": IDENTITY_COLUMNS,
        "standard_inner_join_rows": standard_join_rows,
        "shared_unique_quasi_identity_keys": len(shared),
        "unambiguous_one_to_one_keys": len(one_to_one),
        "ambiguous_shared_keys": len(ambiguous),
        "portuguese_partition_counts": membership.value_counts().to_dict(),
        "claim_boundary": "Quasi-identity matching is not verified student identity; transfer is not independent external validation.",
    }
    return membership, audit
