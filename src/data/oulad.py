"""Cutoff-safe input contract for CNN-BiLSTM OULAD."""

from __future__ import annotations

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
