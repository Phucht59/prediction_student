"""FIT-only context and masked-tensor scalers used by the frozen Phase 4 path."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ContextPreprocessor:
    """Phase 4 context encoder: median+scale numerics, constant-Unknown OHE categoricals."""

    def __init__(self, numeric: list[str], categorical: list[str]):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.columns = self.numeric + self.categorical
        self.transformer = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                    self.numeric,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    self.categorical,
                ),
            ],
            remainder="drop",
        )
        self.fit_record_ids: tuple[str, ...] = ()
        self.output_dim = 0

    def fit(self, frame: pd.DataFrame) -> "ContextPreprocessor":
        ids = tuple(sorted(frame.record_id.astype(str).unique()))
        if len(ids) != len(frame):
            raise ValueError("Context preprocessor requires unique FIT records")
        self.transformer.fit(frame[self.columns])
        self.fit_record_ids = ids
        self.output_dim = int(self.transformer.transform(frame.iloc[:1][self.columns]).shape[1])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fit_record_ids:
            raise RuntimeError("ContextPreprocessor must be fit on FIT records first")
        return np.asarray(self.transformer.transform(frame[self.columns]), dtype=np.float32)


@dataclass
class MaskedStandardScaler:
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    std: np.ndarray = field(default_factory=lambda: np.ones(0, np.float32))

    def fit(self, values: np.ndarray, mask: np.ndarray) -> "MaskedStandardScaler":
        kept = values[mask]
        if kept.size == 0:
            width = values.shape[-1]
            self.mean = np.zeros(width, np.float32)
            self.std = np.ones(width, np.float32)
            return self
        self.mean = kept.mean(0).astype(np.float32)
        self.std = kept.std(0).astype(np.float32)
        self.std = np.where(self.std < 1e-6, 1.0, self.std).astype(np.float32)
        return self

    def transform(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = (values - self.mean) / self.std
        out = out.astype(np.float32)
        out[~mask] = 0.0
        return out


__all__ = ["ContextPreprocessor", "MaskedStandardScaler"]
