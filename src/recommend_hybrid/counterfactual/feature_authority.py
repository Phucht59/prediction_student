"""Canonical OULAD feature rebuilding with the frozen fold preprocessor."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from src.recommend_hybrid.exceptions import ContractValidationError

from .oulad_tensor import BASE_CHANNELS


class AggregatePreprocessor(Protocol):
    def transform_aggregate(self, raw_aggregate: np.ndarray) -> np.ndarray:
        """Apply the frozen outer-fold aggregate transformation."""

    def inverse_transform_aggregate(
        self,
        transformed_aggregate: np.ndarray,
    ) -> np.ndarray:
        """Recover raw aggregate values using the same frozen state."""


class PreprocessedOULADFeatureAuthority:
    """Rebuild raw temporal features, then restore checkpoint model space."""

    base_channels = BASE_CHANNELS

    def __init__(
        self,
        aggregate_preprocessor: AggregatePreprocessor,
        *,
        feature_module: Any | None = None,
    ) -> None:
        if feature_module is None:
            from src.pipelines import oulad as feature_module

        if tuple(feature_module.BASE_CHANNELS) != BASE_CHANNELS:
            raise ContractValidationError(
                "canonical OULAD base-channel authority changed"
            )
        for method in (
            "transform_aggregate",
            "inverse_transform_aggregate",
        ):
            if not callable(getattr(aggregate_preprocessor, method, None)):
                raise ContractValidationError(
                    "frozen aggregate preprocessor authority is required"
                )
        self._features = feature_module
        self._preprocessor = aggregate_preprocessor

    def rebuild(
        self,
        base_sequence: np.ndarray,
        lengths: np.ndarray,
        mask: np.ndarray,
        baseline_aggregate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        dynamic = self._features._dynamic(
            base_sequence,
            mask.astype(bool),
        )
        aggregate_base = self._features._aggregate(
            base_sequence,
            lengths,
        )
        raw_baseline = self._preprocessor.inverse_transform_aggregate(
            baseline_aggregate
        )
        roundtrip = self._preprocessor.transform_aggregate(raw_baseline)
        if not np.allclose(
            roundtrip,
            baseline_aggregate,
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ContractValidationError(
                "baseline aggregate does not match frozen preprocessor"
            )
        raw_aggregate = np.column_stack(
            [aggregate_base, raw_baseline[:, 161:]]
        ).astype(np.float32)
        transformed = self._preprocessor.transform_aggregate(raw_aggregate)
        if dynamic.shape[2] != 47 or transformed.shape[1] != 165:
            raise ContractValidationError(
                "canonical OULAD feature dimensions changed"
            )
        return dynamic.astype(np.float32), transformed.astype(np.float32)


__all__ = [
    "AggregatePreprocessor",
    "PreprocessedOULADFeatureAuthority",
]
