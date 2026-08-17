"""Dataset-neutral Phase 7 representation contract."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass(frozen=True)
class UnifiedHybridData:
    """One semantic input interface for UCI stages and OULAD cutoffs.

    ``static`` is deliberately a matrix even before train-only preprocessing;
    callers may use a zero-width matrix until its dataset adapter supplies the
    train-fitted static representation.  ``aggregate_available`` distinguishes
    an unavailable aggregate (UCI S0) from genuine all-zero observations.
    """

    static: np.ndarray
    temporal: np.ndarray
    temporal_mask: np.ndarray
    lengths: np.ndarray
    aggregate: np.ndarray
    aggregate_available: np.ndarray
    progress: np.ndarray
    target: np.ndarray
    record_id: np.ndarray
    group_id: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    temporal_exposure_normalized: np.ndarray | None = None

    @property
    def mask(self) -> np.ndarray:
        """Compatibility spelling for mask-safe utilities."""
        return self.temporal_mask

    def validate(self) -> None:
        n = len(self.record_id)
        arrays = (self.group_id, self.target, self.lengths, self.aggregate_available, self.progress)
        if any(len(value) != n for value in arrays):
            raise ValueError("Phase7 record-aligned fields have inconsistent lengths")
        if self.static.ndim != 2 or self.static.shape[0] != n:
            raise ValueError("static must have shape [N, D_static]")
        if self.temporal.ndim != 3 or self.temporal.shape[0] != n:
            raise ValueError("temporal must have shape [N, T, D_temporal]")
        if self.temporal_mask.shape != self.temporal.shape[:2] or self.temporal_mask.dtype != bool:
            raise ValueError("temporal_mask must be boolean [N, T]")
        if self.aggregate.ndim != 2 or self.aggregate.shape[0] != n:
            raise ValueError("aggregate must have shape [N, D_aggregate]")
        if not set(np.unique(self.target)).issubset({0, 1}):
            raise ValueError("target must be binary")
        if not np.array_equal(self.lengths, self.temporal_mask.sum(axis=1)):
            raise ValueError("lengths must equal temporal_mask sums")
        if not set(np.unique(self.aggregate_available)).issubset({0, 1, False, True}):
            raise ValueError("aggregate_available must be binary")
        if np.any(~np.isfinite(self.progress)) or np.any((self.progress < 0) | (self.progress > 1)):
            raise ValueError("progress must be finite in [0, 1]")
        if np.any(np.abs(self.temporal[~self.temporal_mask]) > 1e-6):
            raise ValueError("padded temporal positions must be strict zero")
        if self.temporal_exposure_normalized is not None:
            if self.temporal_exposure_normalized.shape != self.temporal.shape:
                raise ValueError("temporal_exposure_normalized must match temporal shape")
            if np.any(np.abs(self.temporal_exposure_normalized[~self.temporal_mask]) > 1e-6):
                raise ValueError("normalized padded positions must be strict zero")
        if len(np.unique(self.record_id.astype(str))) != n:
            raise ValueError("record_id must be unique")
