"""Train-only preprocessing and deterministic prefix sampling."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class HybridBatch:
    temporal: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor
    context: torch.Tensor
    target: torch.Tensor
    record_id: list[str]
    group_id: list[str]
    stage: list[str]


class ContextPreprocessor:
    def __init__(self, numeric: list[str], categorical: list[str]):
        self.columns = numeric + categorical
        self.transformer = ColumnTransformer([
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ], remainder="drop")
        self.fit_record_ids: tuple[str, ...] = ()
        self.fit_record_sha256 = ""
        self.output_dim = 0

    def fit(self, frame: pd.DataFrame) -> "ContextPreprocessor":
        ids = tuple(sorted(frame.record_id.astype(str).unique()))
        if len(ids) != len(frame):
            raise ValueError("Context preprocessor requires unique base training records")
        self.transformer.fit(frame[self.columns])
        self.fit_record_ids = ids
        self.fit_record_sha256 = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        self.output_dim = int(self.transformer.transform(frame.iloc[:1][self.columns]).shape[1])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.transformer.transform(frame[self.columns]), dtype=np.float32)


def sample_prefixes(record_ids: list[str] | np.ndarray, eligible: list[list[str]], seed: int, epoch: int) -> list[str]:
    selected = []
    for record_id, choices in zip(record_ids, eligible, strict=True):
        if not choices:
            raise ValueError(f"No eligible prefix for {record_id}")
        digest = hashlib.sha256(f"{seed}:{epoch}:{record_id}".encode()).digest()
        selected.append(choices[int.from_bytes(digest[:8], "big") % len(choices)])
    return selected


def sample_prefixes_stage_balanced(record_ids, eligible, seed: int, epoch: int) -> list[str]:
    """Deterministic inverse-availability weighting to approximately balance stages."""
    counts: dict[str, int] = {}
    for choices in eligible:
        for stage in choices:
            counts[stage] = counts.get(stage, 0) + 1
    selected = []
    for record_id, choices in zip(record_ids, eligible, strict=True):
        weights = np.asarray([1.0 / counts[stage] for stage in choices], dtype=np.float64)
        weights /= weights.sum()
        digest = hashlib.sha256(f"balanced:{seed}:{epoch}:{record_id}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        selected.append(choices[min(int(np.searchsorted(np.cumsum(weights), value, side="right")), len(choices)-1)])
    return selected
