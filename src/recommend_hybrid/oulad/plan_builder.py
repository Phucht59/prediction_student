"""Remaining-course learning plan builder for OULAD."""

from __future__ import annotations

from typing import Any, Mapping

from src.recommend_hybrid.common.constraints import HybridConstraintSolver
from src.recommend_hybrid.common.explanation import build_plan_explanation
from src.recommend_hybrid.common.plan_contracts import (
    LearningPlan,
    PlanLineage,
    PlanStatus,
    deterministic_plan_id,
)
from src.recommend_hybrid.common.policy_contracts import AutomationStatus, PolicyRecommendationResult


class OULADLearningPlanBuilder:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.solver = HybridConstraintSolver(config)

    def build(
        self,
        policy_result: PolicyRecommendationResult,
        *,
        course_key: str,
        created_at: str,
        active_contraindications: tuple[str, ...] = (),
    ) -> LearningPlan:
        remaining = max(0.0, 100.0 - policy_result.requested_cutoff)
        periods = self._available_periods(remaining, policy_result.automation_status)
        action_limit = 1 if 0 < remaining < float(self.config["oulad"]["minimum_remaining_percent"]["SHORT_TERM"]) else None
        stage = policy_result.prediction_anchor.anchor_stage or "NO_ANCHOR"
        solved = self.solver.solve(
            policy_result,
            stage=stage,
            dataset_group="oulad",
            periods=periods,
            active_contraindications=active_contraindications,
            action_limit=action_limit,
        )
        status = _status(policy_result.automation_status, bool(solved.selected_actions), solved.truncated)
        explanation = build_plan_explanation(
            policy_result,
            solved.selected_actions,
            rejected_actions=solved.rejected_actions,
            constraint_reasons=solved.constraint_reasons,
        )
        identity = {
            "dataset_id": policy_result.dataset_id.value,
            "student_key": policy_result.student_key,
            "course_key": course_key,
            "requested_cutoff": policy_result.requested_cutoff,
            "anchor": policy_result.prediction_anchor.anchor_cutoff,
            "status": status.value,
            "actions": [item.to_dict() for item in solved.selected_actions],
            "policy_version": policy_result.policy_version,
            "planning_version": self.config["planning_version"],
        }
        return LearningPlan(
            plan_id=deterministic_plan_id(identity),
            dataset_id=policy_result.dataset_id.value,
            student_key=policy_result.student_key,
            course_key=course_key,
            requested_cutoff=policy_result.requested_cutoff,
            prediction_anchor=policy_result.prediction_anchor.anchor_cutoff,
            automation_status=status,
            selected_actions=solved.selected_actions,
            total_minutes=sum(item.weekly_minutes for item in solved.selected_actions),
            plan_periods=periods,
            explanation=explanation,
            lineage=_lineage(policy_result),
            model_authority="RECOMMEND_HYBRID_MODEL_AUTHORITY",
            policy_version=policy_result.policy_version,
            planning_version=self.config["planning_version"],
            created_at=created_at,
        )

    def _available_periods(
        self, remaining: float, automation: AutomationStatus
    ) -> tuple[str, ...]:
        if automation is AutomationStatus.EVALUATION_ONLY or remaining <= 0:
            return tuple(self.config["oulad"]["periods"])
        thresholds = self.config["oulad"]["minimum_remaining_percent"]
        result = ["IMMEDIATE"]
        if remaining >= float(thresholds["SHORT_TERM"]):
            result.append("SHORT_TERM")
        if remaining >= float(thresholds["FOLLOW_UP"]):
            result.append("FOLLOW_UP")
        return tuple(result)


def _status(source: AutomationStatus, has_actions: bool, truncated: bool) -> PlanStatus:
    if source is AutomationStatus.EVALUATION_ONLY:
        return PlanStatus.EVALUATION_ONLY
    if source is AutomationStatus.ABSTAIN or not has_actions:
        return PlanStatus.ABSTAIN
    return PlanStatus.PARTIAL if source is AutomationStatus.PARTIAL or truncated else PlanStatus.FULL


def _lineage(result: PolicyRecommendationResult) -> tuple[PlanLineage, ...]:
    rows = [PlanLineage("phase3_policy", result.policy_version)]
    rows.extend(PlanLineage("checkpoint", item) for item in result.prediction_anchor.checkpoint_lineage)
    return tuple(rows)


__all__ = ["OULADLearningPlanBuilder"]
