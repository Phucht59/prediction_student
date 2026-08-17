"""Train-only context preprocessors and feature transformation pipelines."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.hybrid.contracts import MaskedStandardScaler, assert_train_only_fit


class TabularContextPreprocessor:
    """Preprocesses mixed numeric and categorical context features strictly fit on training data."""

    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
    ):
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)

        self.num_imputer = SimpleImputer(strategy="median")
        self.num_scaler = StandardScaler()
        self.cat_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop=None,
        )

        self.fitted_record_ids: set[str] | None = None
        self.feature_names_: list[str] = []
        self._is_fitted: bool = False

    def fit(
        self, df: pd.DataFrame, fit_record_ids: Iterable[str] | None = None
    ) -> TabularContextPreprocessor:
        """Fit imputation, scaling, and one-hot encoding on train frame only."""
        if fit_record_ids is not None:
            self.fitted_record_ids = set(map(str, fit_record_ids))

        # Numeric fitting
        if self.numeric_features:
            num_data = df[self.numeric_features].to_numpy(dtype=np.float64)
            num_imputed = self.num_imputer.fit_transform(num_data)
            self.num_scaler.fit(num_imputed)

        # Categorical fitting
        if self.categorical_features:
            cat_data = df[self.categorical_features].fillna("Unknown").astype(str).to_numpy()
            self.cat_encoder.fit(cat_data)

        # Build output feature names
        names: list[str] = []
        if self.numeric_features:
            names.extend(self.numeric_features)
        if self.categorical_features:
            encoded_names = self.cat_encoder.get_feature_names_out(self.categorical_features)
            names.extend(list(encoded_names))

        self.feature_names_ = names
        self._is_fitted = True
        return self

    def transform(
        self, df: pd.DataFrame, test_record_ids: Iterable[str] | None = None
    ) -> np.ndarray:
        """Transform context features into a float32 array, validating train-only fit."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        if self.fitted_record_ids is not None and test_record_ids is not None:
            assert_train_only_fit(self.fitted_record_ids, test_record_ids)

        parts: list[np.ndarray] = []

        if self.numeric_features:
            num_data = df[self.numeric_features].to_numpy(dtype=np.float64)
            num_imputed = self.num_imputer.transform(num_data)
            num_scaled = self.num_scaler.transform(num_imputed)
            parts.append(num_scaled)

        if self.categorical_features:
            cat_data = df[self.categorical_features].fillna("Unknown").astype(str).to_numpy()
            cat_encoded = self.cat_encoder.transform(cat_data)
            parts.append(cat_encoded)

        if not parts:
            return np.empty((len(df), 0), dtype=np.float32)

        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit_transform(
        self, df: pd.DataFrame, fit_record_ids: Iterable[str] | None = None
    ) -> np.ndarray:
        return self.fit(df, fit_record_ids=fit_record_ids).transform(
            df, test_record_ids=None
        )
