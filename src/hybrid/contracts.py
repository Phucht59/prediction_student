"""Data contracts and validation for Hybrid scientific pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HybridDataView:
    """Neutral dataset contract representing temporal sequence, mask, context, and binary targets."""

    record_id: np.ndarray
    group_id: np.ndarray
    target: np.ndarray
    temporal: np.ndarray
    mask: np.ndarray
    lengths: np.ndarray
    context: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        n_records = len(self.record_id)
        if len(self.group_id) != n_records:
            raise ValueError(f"group_id length ({len(self.group_id)}) != records ({n_records})")
        if len(self.target) != n_records:
            raise ValueError(f"target length ({len(self.target)}) != records ({n_records})")
        if len(self.lengths) != n_records:
            raise ValueError(f"lengths length ({len(self.lengths)}) != records ({n_records})")

        # Temporal dimensions
        if self.temporal.ndim != 3:
            raise ValueError(f"temporal must have shape [N, T, D], got {self.temporal.shape}")
        if self.temporal.shape[0] != n_records:
            raise ValueError(f"temporal records ({self.temporal.shape[0]}) != records ({n_records})")

        # Mask dimensions
        if self.mask.ndim != 2:
            raise ValueError(f"mask must have shape [N, T], got {self.mask.shape}")
        if self.mask.shape != self.temporal.shape[:2]:
            raise ValueError(f"mask shape {self.mask.shape} != temporal shape {self.temporal.shape[:2]}")

        # Context dimensions if provided
        if self.context is not None:
            if self.context.ndim != 2:
                raise ValueError(f"context must have shape [N, D_ctx], got {self.context.shape}")
            if self.context.shape[0] != n_records:
                raise ValueError(f"context records ({self.context.shape[0]}) != records ({n_records})")

        # Target values {0, 1}
        unique_targets = set(np.unique(self.target))
        if not unique_targets.issubset({0, 1}):
            raise ValueError(f"target values must be binary in {{0, 1}}, got {unique_targets}")

        # Duplicate record IDs
        if len(np.unique(self.record_id)) != n_records:
            raise ValueError(f"Duplicate record_id detected in dataset view ({n_records} rows, {len(np.unique(self.record_id))} unique)")

        # Empty group IDs
        if any(str(g).strip() == "" for g in self.group_id):
            raise ValueError("Empty group_id detected")

        # Length validation
        expected_lengths = np.sum(self.mask, axis=1)
        if not np.array_equal(self.lengths, expected_lengths):
            raise ValueError("lengths array does not match sum of boolean mask")

        # Padded timesteps must be exactly 0
        padded_mask = ~self.mask
        if np.any(padded_mask):
            padded_values = self.temporal[padded_mask]
            if np.any(np.abs(padded_values) > 1e-6):
                raise ValueError("Non-zero values found in padded (mask=False) positions")


def assert_train_only_fit(
    fit_record_ids: Iterable[str], test_record_ids: Iterable[str]
) -> None:
    """Reject preprocessing fitted on records that appear in an evaluation set."""
    fit_set = set(map(str, fit_record_ids))
    test_set = set(map(str, test_record_ids))
    overlap = fit_set & test_set
    if overlap:
        raise ValueError(f"Leakage detected: Preprocessor fit set overlaps with test set by {len(overlap)} records")


class MaskedStandardScaler:
    """Standard scaler computed strictly on valid timesteps (mask == True)."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_observed_: np.ndarray | None = None

    def fit(self, temporal: np.ndarray, mask: np.ndarray) -> MaskedStandardScaler:
        """Compute mean and std per channel strictly where mask is True.
        
        Args:
            temporal: [N, T, D] array
            mask: [N, T] boolean array
        """
        if temporal.ndim != 3 or mask.ndim != 2:
            raise ValueError("temporal must be [N, T, D] and mask must be [N, T]")
        if temporal.shape[:2] != mask.shape:
            raise ValueError("temporal and mask shape mismatch")

        n_channels = temporal.shape[2]
        means = np.zeros(n_channels, dtype=np.float64)
        scales = np.ones(n_channels, dtype=np.float64)
        counts = np.zeros(n_channels, dtype=np.int64)

        for c in range(n_channels):
            channel_data = temporal[:, :, c]
            valid_vals = channel_data[mask]
            if len(valid_vals) > 0:
                m = float(np.mean(valid_vals))
                s = float(np.std(valid_vals))
                means[c] = m
                scales[c] = s if s > self.eps else 1.0
                counts[c] = len(valid_vals)
            else:
                means[c] = 0.0
                scales[c] = 1.0
                counts[c] = 0

        self.mean_ = means.astype(np.float32)
        self.scale_ = scales.astype(np.float32)
        self.n_observed_ = counts
        return self

    def transform(self, temporal: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Standardize valid timesteps, forcing padded timesteps to 0.0."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("MaskedStandardScaler must be fitted before transform")
        if temporal.ndim != 3 or mask.ndim != 2:
            raise ValueError("temporal must be [N, T, D] and mask must be [N, T]")
        if temporal.shape[:2] != mask.shape:
            raise ValueError("temporal and mask shape mismatch")

        out = np.zeros_like(temporal, dtype=np.float32)
        for c in range(temporal.shape[2]):
            channel_data = temporal[:, :, c]
            norm_channel = (channel_data - self.mean_[c]) / self.scale_[c]
            norm_channel[~mask] = 0.0
            out[:, :, c] = norm_channel

        return out

    def fit_transform(self, temporal: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.fit(temporal, mask).transform(temporal, mask)
