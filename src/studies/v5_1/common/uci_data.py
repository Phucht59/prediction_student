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


TEMPORAL_CHANNELS = (
    "normalized_grade",
    "stage_indicator",
    "signed_change_from_G1",
    "absolute_change_from_G1",
    "signed_distance_to_boundary_10",
    "signed_distance_to_boundary_15",
    "change_direction",
)
PRIMARY_CONTEXT_FEATURES = (
    "failures",
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
)
SENSITIVITY_CONTEXT_FEATURES = (*PRIMARY_CONTEXT_FEATURES, "absences")
CONTEXT_CATEGORICAL = ("schoolsup", "famsup", "paid", "activities", "internet", "higher")
QUASI_IDENTITY = (
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
)


@dataclass(frozen=True)
class UCIDataV51:
    dataset_id: str
    frame: pd.DataFrame
    temporal: np.ndarray
    context: pd.DataFrame
    target: np.ndarray
    raw_g3: np.ndarray
    record_ids: np.ndarray
    quasi_groups: np.ndarray
    context_contract: str


def _stable_id(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()[:24]


def encode_target(values: pd.Series) -> np.ndarray:
    raw = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all() or ((raw < 0) | (raw > 20)).any():
        raise ValueError("G3 must be finite and inside 0..20")
    return np.where(raw <= 9, 0, np.where(raw <= 14, 1, 2)).astype(np.int64)


def build_temporal_features(frame: pd.DataFrame) -> np.ndarray:
    """Construct the frozen two-step V5.1 contract using G1/G2 only."""

    missing = sorted({"G1", "G2"} - set(frame.columns))
    if missing:
        raise ValueError(f"Missing temporal columns: {missing}")
    grades = frame.loc[:, ["G1", "G2"]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(grades).all() or ((grades < 0) | (grades > 20)).any():
        raise ValueError("G1/G2 must be finite and inside 0..20")
    delta = grades[:, 1] - grades[:, 0]
    result = np.zeros((len(frame), 2, len(TEMPORAL_CHANNELS)), dtype=np.float32)
    result[:, :, 0] = grades / 20.0
    result[:, 0, 1] = -1.0
    result[:, 1, 1] = 1.0
    result[:, 1, 2] = delta / 20.0
    result[:, 1, 3] = np.abs(delta) / 20.0
    result[:, :, 4] = (grades - 10.0) / 20.0
    result[:, :, 5] = (grades - 15.0) / 20.0
    result[:, 1, 6] = np.sign(delta)
    return result


def context_preprocessor(*, include_absences: bool = False) -> ColumnTransformer:
    features = SENSITIVITY_CONTEXT_FEATURES if include_absences else PRIMARY_CONTEXT_FEATURES
    numeric = [column for column in features if column not in CONTEXT_CATEGORICAL]
    categorical = [column for column in features if column in CONTEXT_CATEGORICAL]
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
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0.0,
    )


def load_uci_v5_1(path: str | Path, dataset_id: str, *, include_absences: bool = False) -> UCIDataV51:
    frame = pd.read_csv(path, sep=";")
    context_features = SENSITIVITY_CONTEXT_FEATURES if include_absences else PRIMARY_CONTEXT_FEATURES
    required = {"G1", "G2", "G3", *context_features, *QUASI_IDENTITY}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing UCI columns: {missing}")
    if frame.duplicated().any():
        raise ValueError(f"Exact duplicate rows in {dataset_id}")
    raw_g3 = pd.to_numeric(frame["G3"], errors="raise").to_numpy(dtype=np.float32)
    groups = np.array(
        [_stable_id("quasi", *(row[column] for column in QUASI_IDENTITY)) for _, row in frame.iterrows()]
    )
    record_ids = np.array([_stable_id(dataset_id, index) for index in range(len(frame))])
    return UCIDataV51(
        dataset_id=dataset_id,
        frame=frame.copy(),
        temporal=build_temporal_features(frame),
        context=frame.loc[:, list(context_features)].copy(),
        target=encode_target(frame["G3"]),
        raw_g3=raw_g3,
        record_ids=record_ids,
        quasi_groups=groups,
        context_contract="sensitivity_with_absences" if include_absences else "primary_safe",
    )


__all__ = [
    "CONTEXT_CATEGORICAL",
    "PRIMARY_CONTEXT_FEATURES",
    "QUASI_IDENTITY",
    "SENSITIVITY_CONTEXT_FEATURES",
    "TEMPORAL_CHANNELS",
    "UCIDataV51",
    "build_temporal_features",
    "context_preprocessor",
    "encode_target",
    "load_uci_v5_1",
]

