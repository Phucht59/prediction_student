"""Stage-aware target-trial protocol for recommendation-effect estimation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

STAGE_ORDER = ("EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75")


@dataclass(frozen=True)
class LandmarkStage:
    name: str
    cutoff_fraction: float
    treatment_end_fraction: float

    def __post_init__(self) -> None:
        if self.name not in STAGE_ORDER:
            raise ValueError(f"unknown landmark stage {self.name}")
        if not 0.0 < self.cutoff_fraction < self.treatment_end_fraction <= 1.0:
            raise ValueError("landmark windows must satisfy 0 < cutoff < end <= 1")

    def contains_baseline_progress(self, progress: float) -> bool:
        return 0.0 <= float(progress) <= self.cutoff_fraction

    def contains_treatment_progress(self, progress: float) -> bool:
        value = float(progress)
        return self.cutoff_fraction < value <= self.treatment_end_fraction


LANDMARK_STAGES: Mapping[str, LandmarkStage] = {
    "EARLY_20": LandmarkStage("EARLY_20", 0.20, 0.35),
    "EARLY_35": LandmarkStage("EARLY_35", 0.35, 0.50),
    "MIDDLE_50": LandmarkStage("MIDDLE_50", 0.50, 0.75),
    "LATE_75": LandmarkStage("LATE_75", 0.75, 1.00),
}


@dataclass(frozen=True)
class TargetTrialProtocol:
    """One emulated trial for one stage and one canonical action."""

    stage: str
    action_id: str
    student_id_column: str = "student_id"
    treatment_column: str = "treatment"
    outcome_column: str = "outcome_pass"
    group_column: str = "student_id"

    def __post_init__(self) -> None:
        if self.stage not in LANDMARK_STAGES:
            raise ValueError(f"unsupported stage {self.stage}")
        if not self.action_id or self.action_id.upper() != self.action_id:
            raise ValueError("action_id must be a non-empty canonical uppercase id")
        for value in (
            self.student_id_column,
            self.treatment_column,
            self.outcome_column,
            self.group_column,
        ):
            if not value:
                raise ValueError("column names must be non-empty")

    @property
    def landmark(self) -> LandmarkStage:
        return LANDMARK_STAGES[self.stage]

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "action_id": self.action_id,
            "cutoff_fraction": self.landmark.cutoff_fraction,
            "treatment_end_fraction": self.landmark.treatment_end_fraction,
            "student_id_column": self.student_id_column,
            "treatment_column": self.treatment_column,
            "outcome_column": self.outcome_column,
            "group_column": self.group_column,
        }


def stage_from_fraction(value: float) -> str:
    """Map an exact supported prediction cutoff to its landmark name."""

    fraction = float(value)
    for stage in STAGE_ORDER:
        if abs(LANDMARK_STAGES[stage].cutoff_fraction - fraction) <= 1.0e-9:
            return stage
    raise ValueError(
        f"unsupported prediction cutoff {fraction}; expected 0.20, 0.35, 0.50, or 0.75"
    )


def validate_temporal_columns(
    *,
    stage: str,
    maximum_baseline_progress: float,
    minimum_treatment_progress: float,
    maximum_treatment_progress: float,
) -> None:
    """Fail closed when a trial frame crosses its temporal boundaries."""

    landmark = LANDMARK_STAGES[stage]
    if float(maximum_baseline_progress) > landmark.cutoff_fraction + 1.0e-9:
        raise ValueError("baseline features contain post-cutoff information")
    if float(minimum_treatment_progress) <= landmark.cutoff_fraction:
        raise ValueError("treatment window overlaps the baseline window")
    if float(maximum_treatment_progress) > landmark.treatment_end_fraction + 1.0e-9:
        raise ValueError("treatment features exceed the stage-specific follow-up window")


__all__ = [
    "LANDMARK_STAGES",
    "STAGE_ORDER",
    "LandmarkStage",
    "TargetTrialProtocol",
    "stage_from_fraction",
    "validate_temporal_columns",
]
