from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CONTEXT_FEATURES = [
    "failures",
    "absences",
    "studytime",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "internet",
    "higher",
    "traveltime",
    "freetime",
    "goout",
    "health",
]
CONTEXT_CATEGORICAL = ["schoolsup", "famsup", "paid", "activities", "internet", "higher"]
CONTEXT_NUMERIC = [column for column in CONTEXT_FEATURES if column not in CONTEXT_CATEGORICAL]
QUASI_IDENTITY = [
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


@dataclass(frozen=True)
class UCIData:
    dataset_id: str
    frame: pd.DataFrame
    sequence: np.ndarray
    context: pd.DataFrame
    target: np.ndarray
    raw_g3: np.ndarray
    record_ids: np.ndarray
    quasi_groups: np.ndarray


def encode_target(values: pd.Series) -> np.ndarray:
    raw = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if ((raw < 0) | (raw > 20)).any():
        raise ValueError("G3 outside 0..20")
    return np.where(raw <= 9, 0, np.where(raw <= 14, 1, 2)).astype(np.int64)


def _stable_id(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()[:24]


def load_uci(path: str | Path, dataset_id: str) -> UCIData:
    frame = pd.read_csv(path, sep=";")
    required = {"G1", "G2", "G3", *CONTEXT_FEATURES, *QUASI_IDENTITY}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing UCI columns: {missing}")
    if frame.duplicated().any():
        raise ValueError(f"Exact duplicate rows in {dataset_id}")
    raw_g3 = pd.to_numeric(frame["G3"], errors="raise").to_numpy(dtype=np.float32)
    sequence = frame[["G1", "G2"]].to_numpy(dtype=np.float32)[..., None]
    groups = np.array(
        [_stable_id("quasi", *(row[column] for column in QUASI_IDENTITY)) for _, row in frame.iterrows()]
    )
    record_ids = np.array([_stable_id(dataset_id, index) for index in range(len(frame))])
    return UCIData(
        dataset_id=dataset_id,
        frame=frame.copy(),
        sequence=sequence,
        context=frame.loc[:, CONTEXT_FEATURES].copy(),
        target=encode_target(frame["G3"]),
        raw_g3=raw_g3,
        record_ids=record_ids,
        quasi_groups=groups,
    )


def context_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                CONTEXT_NUMERIC,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CONTEXT_CATEGORICAL,
            ),
        ],
        sparse_threshold=0.0,
    )


def fit_transform_partition(
    data: UCIData,
    fit_indices: np.ndarray,
    transform_indices: np.ndarray,
    fitted: ColumnTransformer | None = None,
) -> tuple[np.ndarray, np.ndarray, ColumnTransformer, np.ndarray, np.ndarray]:
    preprocessor = fitted or context_preprocessor()
    if fitted is None:
        preprocessor.fit(data.context.iloc[fit_indices])
    context = preprocessor.transform(data.context.iloc[transform_indices]).astype(np.float32)
    mean = data.sequence[fit_indices].mean(axis=(0, 1), keepdims=True)
    std = data.sequence[fit_indices].std(axis=(0, 1), keepdims=True).clip(1e-6)
    sequence = ((data.sequence[transform_indices] - mean) / std).astype(np.float32)
    return sequence, context, preprocessor, mean.astype(np.float32), std.astype(np.float32)


__all__ = [
    "CONTEXT_CATEGORICAL",
    "CONTEXT_FEATURES",
    "CONTEXT_NUMERIC",
    "QUASI_IDENTITY",
    "UCIData",
    "context_preprocessor",
    "encode_target",
    "fit_transform_partition",
    "load_uci",
]

