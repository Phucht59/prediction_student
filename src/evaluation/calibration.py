"""Calibration audits for frozen probabilities."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            value += float(mask.mean()) * abs(
                float((predicted[mask] == y_true[mask]).mean())
                - float(confidence[mask].mean())
            )
    return value
