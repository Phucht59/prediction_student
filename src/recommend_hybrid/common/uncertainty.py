"""Versioned uncertainty bands that can only reduce automation."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from .policy_contracts import PolicyPredictionContext


class UncertaintyDisposition(str, Enum):
    ACCEPTABLE = "ACCEPTABLE"
    CAUTION = "CAUTION"
    ABSTAIN = "ABSTAIN"


def uncertainty_disposition(
    prediction: PolicyPredictionContext, common_config: Mapping[str, Any]
) -> UncertaintyDisposition:
    policy = common_config["uncertainty"]
    entropy = policy["predictive_entropy"]
    disagreement = policy["seed_disagreement"]
    if (
        prediction.uncertainty >= float(entropy["abstain_at"])
        or prediction.seed_disagreement >= float(disagreement["abstain_at"])
    ):
        return UncertaintyDisposition.ABSTAIN
    if (
        prediction.uncertainty >= float(entropy["caution_at"])
        or prediction.seed_disagreement >= float(disagreement["caution_at"])
    ):
        return UncertaintyDisposition.CAUTION
    return UncertaintyDisposition.ACCEPTABLE


__all__ = ["UncertaintyDisposition", "uncertainty_disposition"]
