"""Student Learning State identity and operational risk-band contract."""

from __future__ import annotations

import hashlib


def make_case_id(dataset: str, record_id: str, stage: str) -> str:
    """Create a stable key without introducing a new student identity."""
    value = f"{dataset}|{record_id}|{stage}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:32]


def operational_risk_band(
    probability: float,
    *,
    low_risk_max: float = 0.33,
    medium_risk_max: float = 0.66,
) -> str:
    """Map probability to configurable operational bands.

    These are recommendation-operational thresholds, not prediction
    thresholds, and remain configurable until a recommendation policy exists.
    """
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    if not 0 < low_risk_max < medium_risk_max < 1:
        raise ValueError("risk-band thresholds must be ordered in (0, 1)")
    if probability < low_risk_max:
        return "low"
    if probability < medium_risk_max:
        return "medium"
    return "high"
