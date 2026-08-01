"""Cutoff-safe OULAD policy evidence at the user's actual requested cutoff."""

from __future__ import annotations

from typing import Any

from src.recommend_hybrid.common.policy_contracts import (
    EvidenceAvailability,
    EvidenceItem,
    EvidenceSeverity,
)


def _item(
    feature: str,
    value: Any,
    *,
    cutoff: float,
    observation_end: float | None,
    source: str,
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
        evidence_id=f"oulad:{feature}:{cutoff:g}",
        feature_name=feature,
        observed_value=value if available else None,
        severity=EvidenceSeverity.NONE if available else EvidenceSeverity.MISSING,
        availability=availability,
        source_lineage=source,
        observation_end=observation_end if available else None,
        cutoff=cutoff,
    )


def build_oulad_observed_state(
    *,
    requested_cutoff: float,
    max_observation_cutoff: float | None,
    activity_level: float | None,
    recent_activity_trend: float | None,
    inactivity_streak: int | None,
    assessment_progress: float | None,
    assessments_due: int | None,
    grade_trend: float | None,
    grade_release_verified: bool,
    knowledge_gap: str | None,
) -> tuple[EvidenceItem, ...]:
    time_derived_present = any(
        value is not None
        for value in (
            activity_level,
            recent_activity_trend,
            inactivity_streak,
            assessment_progress,
            assessments_due,
            grade_trend,
            knowledge_gap,
        )
    )
    if time_derived_present and max_observation_cutoff is None:
        raise ValueError("OULAD observed evidence requires an observation-end lineage")
    if max_observation_cutoff is not None and max_observation_cutoff >= requested_cutoff:
        raise ValueError("OULAD evidence must be strictly before requested cutoff")
    if grade_trend is not None and not grade_release_verified:
        raise ValueError("grade evidence requires a verified pre-cutoff release timestamp")
    if assessment_progress is not None and not 0 <= assessment_progress <= 1:
        raise ValueError("assessment progress must be in [0,1]")
    if assessments_due is not None and assessments_due < 0:
        raise ValueError("assessments_due must be non-negative")
    assessment_applicable = assessments_due is not None and assessments_due > 0
    return (
        _item(
            "activity_level",
            activity_level,
            cutoff=requested_cutoff,
            observation_end=max_observation_cutoff,
            source="studentVle/sum_click/pre-request aggregation",
        ),
        _item(
            "recent_activity_trend",
            recent_activity_trend,
            cutoff=requested_cutoff,
            observation_end=max_observation_cutoff,
            source="studentVle/date,sum_click/recent-minus-prior",
        ),
        _item(
            "inactivity_streak",
            inactivity_streak,
            cutoff=requested_cutoff,
            observation_end=max_observation_cutoff,
            source="studentVle/date/last-observed gap",
        ),
        _item(
            "assessment_progress",
            assessment_progress,
            cutoff=requested_cutoff,
            observation_end=max_observation_cutoff,
            source="studentAssessment+assessments/pre-request completion ratio",
            applicable=assessment_applicable,
        ),
        _item(
            "assessments_due",
            assessments_due,
            cutoff=requested_cutoff,
            observation_end=max_observation_cutoff,
            source="assessments/date/due-before-request count",
        ),
        _item(
            "course_progress",
            requested_cutoff / 100.0,
            cutoff=requested_cutoff,
            observation_end=None,
            source="policy_oulad.yaml/requested cutoff fraction",
        ),
        _item(
            "grade_trend",
            grade_trend,
            cutoff=requested_cutoff,
            observation_end=max_observation_cutoff,
            source="verified released assessment scores/pre-request trend",
            applicable=grade_release_verified,
        ),
        _item(
            "knowledge_gap",
            knowledge_gap,
            cutoff=requested_cutoff,
            observation_end=max_observation_cutoff,
            source="verified pre-request diagnostic evidence",
        ),
    )


__all__ = ["build_oulad_observed_state"]
