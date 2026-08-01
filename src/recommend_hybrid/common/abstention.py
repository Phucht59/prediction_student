"""Shared automation and abstention policy."""

from __future__ import annotations

from .policy_contracts import AutomationStatus, EvidenceAvailability, EvidenceItem
from .uncertainty import UncertaintyDisposition


def automation_status(
    *,
    uncertainty: UncertaintyDisposition,
    core_features: tuple[str, ...],
    evidence_by_name: dict[str, EvidenceItem],
    supported_actions: int,
) -> tuple[AutomationStatus, tuple[str, ...]]:
    if uncertainty is UncertaintyDisposition.ABSTAIN:
        return AutomationStatus.ABSTAIN, ("PREDICTION_UNCERTAINTY_EXCEEDS_POLICY",)
    if supported_actions == 0:
        return AutomationStatus.ABSTAIN, ("NO_EVIDENCE_SUPPORTED_ACTION",)
    missing = tuple(
        feature
        for feature in core_features
        if feature not in evidence_by_name
        or evidence_by_name[feature].availability is not EvidenceAvailability.AVAILABLE
    )
    if uncertainty is UncertaintyDisposition.CAUTION or missing:
        reasons = []
        if uncertainty is UncertaintyDisposition.CAUTION:
            reasons.append("UNCERTAINTY_REQUIRES_PARTIAL_AUTOMATION")
        if missing:
            reasons.append("CORE_EVIDENCE_PARTIALLY_MISSING")
        return AutomationStatus.PARTIAL, tuple(reasons)
    return AutomationStatus.FULL, ()


__all__ = ["automation_status"]
