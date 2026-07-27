"""Deterministic probability ensemble."""

import numpy as np


def mean_probability(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.ndim < 2:
        raise ValueError("probabilities must include an ensemble axis")
    return probabilities.mean(axis=0)
