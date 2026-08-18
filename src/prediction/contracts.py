"""Fail-closed binary target and prediction contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


UCI_RISK_RULE = "risk = 1 if G3 < 10 else 0"
OULAD_RISK_RULE = "risk = 1 if final_result in {'Fail', 'Withdrawn'} else 0"
OULAD_ENDPOINTS = ("20pct", "35pct", "50pct", "75pct", "FINAL-100")
UCI_STAGES = ("S0", "S1", "S2")


def uci_risk_target(g3: Iterable[float] | np.ndarray) -> np.ndarray:
    """Construct the frozen UCI binary target from G3 only."""

    values = np.asarray(g3)
    return (values < 10).astype(np.int64)


def oulad_risk_target(final_result: Iterable[str] | np.ndarray) -> np.ndarray:
    """Construct the frozen OULAD binary target from final_result only."""

    values = np.asarray(final_result).astype(str)
    allowed = {"Fail", "Withdrawn", "Pass", "Distinction"}
    unknown = set(np.unique(values)) - allowed
    if unknown:
        raise ValueError(f"unknown final_result values: {sorted(unknown)}")
    return np.isin(values, ("Fail", "Withdrawn")).astype(np.int64)


def assert_binary_target(values: Iterable[int] | np.ndarray, *, name: str = "target") -> None:
    unique = set(np.unique(np.asarray(values)))
    if not unique.issubset({0, 1}):
        raise ValueError(f"{name} must be binary {{0, 1}}, got {sorted(unique)}")


@dataclass(frozen=True)
class PredictionResult:
    """Canonical output consumed by recommendation without model inspection."""

    dataset: str
    record_id: str
    stage_or_endpoint: str
    risk_probability: float
    predicted_risk: int
    threshold: float
    uncertainty: float | None = None
    model_id: str = "hybrid"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_id != "hybrid":
            raise ValueError("only model_id='hybrid' is active")
        if not 0.0 <= float(self.risk_probability) <= 1.0:
            raise ValueError("risk_probability must be in [0, 1]")
        if int(self.predicted_risk) not in {0, 1}:
            raise ValueError("predicted_risk must be binary")
        if not 0.0 < float(self.threshold) < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if self.uncertainty is not None and not 0.0 <= float(self.uncertainty) <= 1.0:
            raise ValueError("uncertainty must be in [0, 1]")

    def recommendation_features(self) -> dict[str, Any]:
        """Return only stable, model-neutral fields for recommendation."""

        output = {
            "dataset": self.dataset,
            "student_key": self.record_id,
            "stage_or_endpoint": self.stage_or_endpoint,
            "risk_probability": float(self.risk_probability),
            "predicted_risk": int(self.predicted_risk),
            "threshold": float(self.threshold),
            "model_id": self.model_id,
        }
        if self.uncertainty is not None:
            output["hybrid_uncertainty"] = float(self.uncertainty)
        output.update(self.metadata)
        return output


__all__ = [
    "PredictionResult",
    "UCI_RISK_RULE",
    "OULAD_RISK_RULE",
    "OULAD_ENDPOINTS",
    "UCI_STAGES",
    "uci_risk_target",
    "oulad_risk_target",
    "assert_binary_target",
]
