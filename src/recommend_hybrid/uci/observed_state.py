"""Leakage-safe UCI academic observations for S0/S1/S2."""

from __future__ import annotations

from typing import Any, Mapping

from src.recommend_hybrid.common.policy_contracts import (
    EvidenceAvailability,
    EvidenceItem,
    EvidenceSeverity,
)

FORBIDDEN = {"G3", "final_outcome", "test_label", "outer_label"}


def _item(
    feature: str,
    value: Any,
    *,
    cutoff: float,
    applicable: bool = True,
) -> EvidenceItem:
    available = value is not None and applicable
    availability = (
        EvidenceAvailability.AVAILABLE
        if available
        else EvidenceAvailability.NOT_APPLICABLE
        if not applicable
        else EvidenceAvailability.MISSING
    )
    return EvidenceItem(
        evidence_id=f"uci:{feature}:{cutoff:g}",
        feature_name=feature,
        observed_value=value if available else None,
        severity=EvidenceSeverity.NONE if available else EvidenceSeverity.MISSING,
        availability=availability,
        source_lineage=f"UCI Student Performance/{feature}/pre-stage academic observation",
        observation_end=cutoff,
        cutoff=cutoff,
    )


def build_uci_observed_state(
    *,
    stage: str,
    cutoff: float,
    g1: float | None,
    g2: float | None,
    absences: int | None,
    study_time: int | None,
    previous_failures: int | None,
    next_assessment_available: bool | None,
    extra_features: Mapping[str, Any] | None = None,
) -> tuple[EvidenceItem, ...]:
    extra = dict(extra_features or {})
    forbidden = sorted(set(extra) & FORBIDDEN)
    if forbidden:
        raise ValueError(f"forbidden UCI policy feature(s): {', '.join(forbidden)}")
    if stage == "S0" and (g1 is not None or g2 is not None):
        raise ValueError("S0 cannot contain grade evidence")
    if stage == "S1" and (g1 is None or g2 is not None):
        raise ValueError("S1 requires G1 and prohibits G2")
    if stage == "S2" and (g1 is None or g2 is None):
        raise ValueError("S2 requires G1 and G2")
    decline = max(float(g1) - float(g2), 0.0) if stage == "S2" else None
    improvement = max(float(g2) - float(g1), 0.0) if stage == "S2" else None
    return (
        _item("absences", absences, cutoff=cutoff),
        _item("study_time", study_time, cutoff=cutoff),
        _item("previous_failures", previous_failures, cutoff=cutoff),
        _item("G1", g1, cutoff=cutoff, applicable=stage in {"S1", "S2"}),
        _item("G2", g2, cutoff=cutoff, applicable=stage == "S2"),
        _item("grade_decline", decline, cutoff=cutoff, applicable=stage == "S2"),
        _item("grade_improvement", improvement, cutoff=cutoff, applicable=stage == "S2"),
        _item(
            "next_assessment_available",
            next_assessment_available,
            cutoff=cutoff,
            applicable=stage in {"S1", "S2"},
        ),
    )


__all__ = ["FORBIDDEN", "build_uci_observed_state"]
