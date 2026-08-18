"""Cutoff-safe input contract for CNN-BiLSTM OULAD."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OULADInputContract:
    sequence: np.ndarray
    lengths: np.ndarray
    mask: np.ndarray
    aggregate: np.ndarray
    static: np.ndarray

    def validate(self) -> None:
        records = len(self.sequence)
        if self.sequence.ndim != 3 or self.sequence.shape[2] != 47:
            raise ValueError("OULAD sequence must have 47 temporal channels")
        if any(len(value) != records for value in (self.lengths, self.mask, self.aggregate, self.static)):
            raise ValueError("OULAD branches must have aligned records")
        if np.any(self.lengths <= 0):
            raise ValueError("sequence lengths must be positive")


@dataclass(frozen=True)
class UCIInputContract:
    """Validated temporal and context inputs for the UCI models."""

    temporal: np.ndarray
    context: np.ndarray

    def validate(self) -> None:
        if self.temporal.ndim != 3:
            raise ValueError("temporal must have shape [records, steps, channels]")
        if self.context.ndim != 2:
            raise ValueError("context must have shape [records, features]")
        if len(self.temporal) != len(self.context):
            raise ValueError("temporal/context record counts differ")


def assert_train_only_fit(
    fit_record_ids: Iterable[str], test_record_ids: Iterable[str]
) -> None:
    """Reject preprocessing fitted on records that appear in an evaluation set."""

    overlap = set(map(str, fit_record_ids)) & set(map(str, test_record_ids))
    if overlap:
        raise ValueError(f"preprocessor fit/test overlap: {len(overlap)} records")
