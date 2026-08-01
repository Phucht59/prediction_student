"""Cutoff-safe construction of observed learning state and feature lineage."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from .contracts import FeatureLineage, ObservedLearningState, Stage
from .exceptions import PostCutoffDataError, SensitiveFeatureError

PROHIBITED_FIELDS = frozenset(
    {
        "age_band",
        "disability",
        "gender",
        "region",
        "imd_band",
        "final_result",
        "target",
        "outer_label",
        "date_unregistration",
        "withdrawal_outcome",
    }
)


@dataclass(frozen=True)
class ActivityEvent:
    day: int
    clicks: float
    source_table: str = "studentVle"
    source_column: str = "sum_click"


@dataclass(frozen=True)
class AssessmentEvent:
    due_day: int
    submitted_day: int | None
    score: float | None = None
    score_release_day: int | None = None
    source_table: str = "studentAssessment+assessments"


def reject_sensitive_fields(field_names: Iterable[str]) -> None:
    invalid = sorted({field.lower() for field in field_names} & PROHIBITED_FIELDS)
    if invalid:
        raise SensitiveFeatureError(f"prohibited fields: {', '.join(invalid)}")


def _lineage(
    feature: str,
    source_table: str,
    source_column: str,
    aggregation: str,
    days: list[int],
    cutoff_day: int,
    missing: bool,
) -> FeatureLineage:
    return FeatureLineage(
        feature=feature,
        source_table=source_table,
        source_column=source_column,
        aggregation=aggregation,
        observation_start=min(days) if days else None,
        observation_end=max(days) if days else None,
        cutoff_day=cutoff_day,
        missing_status="MISSING" if missing else "AVAILABLE",
    )


class ObservedStateBuilder:
    """Build normalized evidence using only events with ``event_day < cutoff_day``."""

    _progress = {
        Stage.EARLY_20: 0.20,
        Stage.EARLY_35: 0.35,
        Stage.MIDDLE_50: 0.50,
        Stage.LATE_75: 0.75,
        Stage.FINAL_EVALUATION: 1.00,
    }

    def build(
        self,
        *,
        stage: Stage,
        cutoff_day: int,
        activity_events: Iterable[ActivityEvent],
        assessment_events: Iterable[AssessmentEvent],
        source_fields: Iterable[str] = (),
    ) -> ObservedLearningState:
        reject_sensitive_fields(source_fields)
        activity = tuple(activity_events)
        assessments = tuple(assessment_events)
        if any(event.day >= cutoff_day for event in activity):
            raise PostCutoffDataError("activity event is at or after cutoff")
        if any(
            day is not None and day >= cutoff_day
            for event in assessments
            for day in (event.submitted_day, event.score_release_day)
        ):
            raise PostCutoffDataError("assessment evidence is at or after cutoff")

        available: list[str] = ["course_progress"]
        missing: list[str] = []
        lineages: list[FeatureLineage] = []
        activity_days = [event.day for event in activity]
        if activity:
            clicks = [float(event.clicks) for event in activity]
            recent = [float(event.clicks) for event in activity if event.day >= cutoff_day - 14]
            earlier = [float(event.clicks) for event in activity if event.day < cutoff_day - 14]
            total_activity = sum(clicks)
            average_activity = mean(clicks)
            recent_activity = sum(recent)
            recent_trend = mean(recent) - mean(earlier) if recent and earlier else None
            inactivity = max(0, cutoff_day - 1 - max(activity_days))
            activity_level = average_activity
            for feature in (
                "activity_level",
                "total_activity",
                "average_activity",
                "recent_activity",
                "inactivity_streak",
            ):
                available.append(feature)
                lineages.append(
                    _lineage(feature, "studentVle", "date,sum_click", feature, activity_days, cutoff_day, False)
                )
            if recent_trend is None:
                missing.append("recent_activity_trend")
            else:
                available.append("recent_activity_trend")
            lineages.append(
                _lineage(
                    "recent_activity_trend",
                    "studentVle",
                    "date,sum_click",
                    "recent_14_day_mean_minus_prior_mean",
                    activity_days,
                    cutoff_day,
                    recent_trend is None,
                )
            )
        else:
            activity_level = total_activity = average_activity = recent_activity = None
            recent_trend = None
            inactivity = None
            for feature in (
                "activity_level",
                "total_activity",
                "average_activity",
                "recent_activity",
                "recent_activity_trend",
                "inactivity_streak",
            ):
                missing.append(feature)
                lineages.append(
                    _lineage(feature, "studentVle", "date,sum_click", feature, [], cutoff_day, True)
                )

        due = [event for event in assessments if event.due_day < cutoff_day]
        completed = [
            event for event in due if event.submitted_day is not None and event.submitted_day < cutoff_day
        ]
        assessment_progress = len(completed) / len(due) if due else None
        assessment_days = [event.due_day for event in due]
        if due:
            available.extend(
                ["assessment_progress", "assessments_due", "assessments_completed"]
            )
        else:
            missing.extend(["assessment_progress", "assessments_due", "assessments_completed"])
        for feature in ("assessment_progress", "assessments_due", "assessments_completed"):
            lineages.append(
                _lineage(
                    feature,
                    "studentAssessment+assessments",
                    "date,date_submitted",
                    feature,
                    assessment_days,
                    cutoff_day,
                    not due,
                )
            )

        released = sorted(
            (
                event.score_release_day,
                float(event.score),
            )
            for event in completed
            if event.score is not None and event.score_release_day is not None
        )
        grade_trend = released[-1][1] - released[0][1] if len(released) >= 2 else None
        if grade_trend is None:
            missing.append("grade_trend")
        else:
            available.append("grade_trend")
        lineages.append(
            _lineage(
                "grade_trend",
                "studentAssessment",
                "score,score_release_day",
                "last_released_score_minus_first_released_score",
                [item[0] for item in released],
                cutoff_day,
                grade_trend is None,
            )
        )
        lineages.append(
            _lineage(
                "course_progress",
                "configs/recommend_hybrid/model_authority.yaml",
                "stage_policy",
                "canonical_stage_fraction",
                [],
                cutoff_day,
                False,
            )
        )
        return ObservedLearningState(
            activity_level=activity_level,
            inactivity_streak=inactivity,
            assessment_progress=assessment_progress,
            grade_trend=grade_trend,
            course_progress=self._progress[stage],
            recent_activity_trend=recent_trend,
            available_evidence=tuple(sorted(set(available))),
            missing_evidence=tuple(sorted(set(missing))),
            feature_lineage=tuple(lineages),
            cutoff_day=cutoff_day,
            stage=stage,
            total_activity=total_activity,
            recent_activity=recent_activity,
            average_activity=average_activity,
            assessments_due=len(due) if due else None,
            assessments_completed=len(completed) if due else None,
        )


__all__ = [
    "ActivityEvent",
    "AssessmentEvent",
    "ObservedStateBuilder",
    "PROHIBITED_FIELDS",
    "reject_sensitive_fields",
]
