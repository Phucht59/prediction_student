"""Final risk-profile contract."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskProfile:
    record_id: str
    probability: float
    risk_label: str
