"""Immutable contracts for constrained, replayable learning plans."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Any

from src.recommend_hybrid.exceptions import ContractValidationError

from .policy_contracts import EvidenceItem, PolicyContract, Priority


class PlanStatus(str, Enum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    ABSTAIN = "ABSTAIN"
    EVALUATION_ONLY = "EVALUATION_ONLY"


@dataclass(frozen=True)
class SelectedAction(PolicyContract):
    action_id: str
    priority: Priority
    scheduled_period: str
    weekly_minutes: int
    reason_codes: tuple[str, ...]
    supporting_evidence: tuple[EvidenceItem, ...]
    success_criterion: str
    requires_human_contact: bool
    policy_version: str

    def __post_init__(self) -> None:
        if not self.action_id or self.priority is Priority.NOT_APPLICABLE:
            raise ContractValidationError("selected action requires an ordinal priority")
        if not self.scheduled_period or not 0 < self.weekly_minutes <= 180:
            raise ContractValidationError("selected action has invalid schedule/workload")
        if not self.reason_codes or not self.supporting_evidence:
            raise ContractValidationError("selected action requires evidence and reason codes")
        if not self.success_criterion or not self.policy_version:
            raise ContractValidationError("selected action requires criterion and policy lineage")


@dataclass(frozen=True)
class PlanExplanation(PolicyContract):
    current_state: tuple[str, ...]
    main_issues: tuple[str, ...]
    selected_reasons: tuple[str, ...]
    excluded_actions: tuple[str, ...]
    limitations: tuple[str, ...]
    routing: str


@dataclass(frozen=True)
class PlanLineage(PolicyContract):
    source: str
    reference: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.source or not self.reference:
            raise ContractValidationError("plan lineage source/reference are required")
        if self.sha256 is not None and len(self.sha256) != 64:
            raise ContractValidationError("lineage SHA-256 is invalid")


@dataclass(frozen=True)
class LearningPlan(PolicyContract):
    plan_id: str
    dataset_id: str
    student_key: str
    course_key: str
    requested_cutoff: float
    prediction_anchor: float | None
    automation_status: PlanStatus
    selected_actions: tuple[SelectedAction, ...]
    total_minutes: int
    plan_periods: tuple[str, ...]
    explanation: PlanExplanation
    lineage: tuple[PlanLineage, ...]
    model_authority: str
    policy_version: str
    planning_version: str
    created_at: str

    def __post_init__(self) -> None:
        if not all((self.plan_id, self.dataset_id, self.student_key, self.course_key)):
            raise ContractValidationError("plan identity is required")
        if self.automation_status in {PlanStatus.ABSTAIN, PlanStatus.EVALUATION_ONLY} and self.selected_actions:
            raise ContractValidationError("abstain/evaluation-only plan must have zero actions")
        if self.total_minutes != sum(action.weekly_minutes for action in self.selected_actions):
            raise ContractValidationError("plan workload total is inconsistent")
        if len({action.action_id for action in self.selected_actions}) != len(self.selected_actions):
            raise ContractValidationError("plan contains duplicate actions")
        if any(action.scheduled_period not in self.plan_periods for action in self.selected_actions):
            raise ContractValidationError("action references an undefined plan period")
        if not self.lineage or not all((self.model_authority, self.policy_version, self.planning_version)):
            raise ContractValidationError("plan authority lineage is required")
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ContractValidationError("created_at must be ISO-8601") from exc

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LearningPlan":
        actions = tuple(
            SelectedAction(
                action_id=row["action_id"],
                priority=Priority(row["priority"]),
                scheduled_period=row["scheduled_period"],
                weekly_minutes=int(row["weekly_minutes"]),
                reason_codes=tuple(row["reason_codes"]),
                supporting_evidence=tuple(
                    _evidence_from_dict(item) for item in row["supporting_evidence"]
                ),
                success_criterion=row["success_criterion"],
                requires_human_contact=bool(row["requires_human_contact"]),
                policy_version=row["policy_version"],
            )
            for row in payload["selected_actions"]
        )
        explanation = PlanExplanation(
            **{
                key: tuple(value) if isinstance(value, list) else value
                for key, value in payload["explanation"].items()
            }
        )
        lineage = tuple(PlanLineage(**row) for row in payload["lineage"])
        return cls(
            **{
                **payload,
                "automation_status": PlanStatus(payload["automation_status"]),
                "selected_actions": actions,
                "plan_periods": tuple(payload["plan_periods"]),
                "explanation": explanation,
                "lineage": lineage,
            }
        )


def _evidence_from_dict(payload: dict[str, Any]) -> EvidenceItem:
    from .policy_contracts import EvidenceAvailability, EvidenceSeverity

    return EvidenceItem(
        evidence_id=payload["evidence_id"],
        feature_name=payload["feature_name"],
        observed_value=payload["observed_value"],
        severity=EvidenceSeverity(payload["severity"]),
        availability=EvidenceAvailability(payload["availability"]),
        source_lineage=payload["source_lineage"],
        observation_end=payload["observation_end"],
        cutoff=payload["cutoff"],
    )


def deterministic_plan_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "rhp_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


__all__ = [
    "LearningPlan",
    "PlanExplanation",
    "PlanLineage",
    "PlanStatus",
    "SelectedAction",
    "deterministic_plan_id",
]
