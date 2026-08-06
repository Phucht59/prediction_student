"""Offline learning-plan construction with an explicit claim boundary."""
from __future__ import annotations

from dataclasses import dataclass

from .contracts import ActionScore, CanonicalAction, Stage


@dataclass(frozen=True)
class LearningPlan:
    primary_action: str
    secondary_actions: tuple[str, ...]
    rationale_vi: str
    evidence: tuple[tuple[str, float], ...]
    duration_days: int
    measurable_target: str
    review_stage: str
    claim_boundary: str = "Model-implied offline recommendation; not causal or deployment evidence."
    runtime_authorized: bool = False


def build_plan(ranked: tuple[ActionScore, ...], stage: Stage) -> LearningPlan:
    if not ranked:
        raise ValueError("a plan requires at least one ranked action")
    primary = ranked[0].action
    duration = 14 if primary is CanonicalAction.STUDY_REGULARITY else 7
    return LearningPlan(primary.value, tuple(x.action.value for x in ranked[1:3]), "Bằng chứng hành vi trước cutoff hỗ trợ ưu tiên hành động này.", ranked[0].explanation, duration, "Hoàn thành mục tiêu hành động trong thời lượng đã định.", stage.value, runtime_authorized=False)
