"""Input contract for CNN-BiLSTM MAT and CNN-BiLSTM POR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UCIInputContract:
    temporal: np.ndarray
    context: np.ndarray

    def validate(self) -> None:
        if self.temporal.ndim != 3:
            raise ValueError("temporal must have shape [records, steps, channels]")
        if self.context.ndim != 2:
            raise ValueError("context must have shape [records, features]")
        if len(self.temporal) != len(self.context):
            raise ValueError("temporal/context record counts differ")
