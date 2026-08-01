"""Shared deterministic constraint solver for UCI and OULAD learning plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .plan_contracts import SelectedAction
from .policy_contracts import (
    AutomationStatus,
    PolicyActionDecision,
    PolicyRecommendationResult,
    Priority,
)
from .selector import DeterministicActionSelector


@dataclass(frozen=True)
class ConstraintResult:
    selected_actions: tuple[SelectedAction, ...]
    rejected_actions: tuple[str, ...]
    constraint_reasons: tuple[str, ...]
    truncated: bool


class HybridConstraintSolver:
    """Apply configured safety constraints without ranking or probability conversion."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.selector = DeterministicActionSelector(config)

    def solve(
        self,
        result: PolicyRecommendationResult,
        *,
        stage: str,
        dataset_group: str,
        periods: tuple[str, ...],
        active_contraindications: tuple[str, ...] = (),
        action_limit: int | None = None,
    ) -> ConstraintResult:
        if result.automation_status in {AutomationStatus.ABSTAIN, AutomationStatus.EVALUATION_ONLY}:
            excluded = tuple(item.action_id for item in result.action_decisions)
            return ConstraintResult((), excluded, tuple(result.abstention_reasons), False)
        if not periods:
            return ConstraintResult((), (), ("NO_SAFE_PLAN_PERIOD",), True)
        dataset_id = result.dataset_id.value
        allowed = set(self.config["dataset_actions"][dataset_id])
        stage_allowed = set(self.config[dataset_group]["stage_actions"].get(stage, ()))
        human_contact = set(self.config["human_contact_actions"])
        metadata = self.config["action_metadata"]
        ordered = self.selector.order(result.action_decisions, stage=stage, dataset_group=dataset_group)
        decisions_by_id = {item.action_id: item for item in ordered}
        rejected = [
            item.action_id
            for item in result.action_decisions
            if item.priority is Priority.NOT_APPLICABLE
        ]
        reasons = [
            f"{item.action_id}:POLICY_{item.eligibility_status.value}"
            for item in result.action_decisions
            if item.priority is Priority.NOT_APPLICABLE
        ]
        planning_truncated = False
        valid: dict[str, PolicyActionDecision] = {}
        for decision in ordered:
            action_id = decision.action_id
            action_meta = metadata.get(action_id)
            if action_id not in allowed or action_meta is None:
                rejected.append(action_id)
                reasons.append(f"{action_id}:DATASET_NOT_APPLICABLE")
                planning_truncated = True
            elif action_id not in stage_allowed:
                rejected.append(action_id)
                reasons.append(f"{action_id}:STAGE_NOT_APPLICABLE")
                planning_truncated = True
            elif (action_id in human_contact) != decision.requires_human_contact:
                rejected.append(action_id)
                reasons.append(f"{action_id}:HUMAN_CONTACT_REQUIREMENT_MISMATCH")
                planning_truncated = True
            elif not decision.supporting_evidence:
                rejected.append(action_id)
                reasons.append(f"{action_id}:MISSING_SUPPORTING_EVIDENCE")
                planning_truncated = True
            elif set(action_meta["contraindications"]) & set(active_contraindications):
                rejected.append(action_id)
                reasons.append(f"{action_id}:CONTRAINDICATED")
                planning_truncated = True
            else:
                valid[action_id] = decision
        for left, right in self.config.get("conflicting_action_pairs", ()):
            if left in valid and right in valid:
                loser = max(
                    (valid[left], valid[right]),
                    key=lambda item: self.selector.key(
                        item, stage=stage, dataset_group=dataset_group
                    ),
                )
                valid.pop(loser.action_id)
                rejected.append(loser.action_id)
                reasons.append(f"{loser.action_id}:CONFLICTING_ACTION")
                planning_truncated = True
        for action_id in tuple(valid):
            missing = [item for item in metadata[action_id]["prerequisites"] if item not in valid]
            if missing:
                valid.pop(action_id)
                rejected.append(action_id)
                reasons.append(f"{action_id}:PREREQUISITE_NOT_MET:{','.join(sorted(missing))}")
                planning_truncated = True

        stable = self._topological_order(valid, metadata, stage=stage, dataset_group=dataset_group)
        cap = min(action_limit or int(self.config["max_actions_per_plan"]), int(self.config["max_actions_per_plan"]))
        period_minutes = {period: 0 for period in periods}
        selected: list[SelectedAction] = []
        selected_ids: set[str] = set()
        for decision in stable:
            if len(selected) >= cap:
                rejected.append(decision.action_id)
                reasons.append(f"{decision.action_id}:ACTION_CAP")
                planning_truncated = True
                continue
            action_meta = metadata[decision.action_id]
            prerequisites = tuple(action_meta["prerequisites"])
            if any(item not in selected_ids for item in prerequisites):
                rejected.append(decision.action_id)
                reasons.append(f"{decision.action_id}:PREREQUISITE_NOT_SELECTED")
                planning_truncated = True
                continue
            period = self._schedule_period(
                action_meta,
                dataset_group=dataset_group,
                periods=periods,
                period_minutes=period_minutes,
                prerequisite_periods=tuple(
                    action.scheduled_period for action in selected if action.action_id in prerequisites
                ),
            )
            if period is None:
                rejected.append(decision.action_id)
                reasons.append(f"{decision.action_id}:WORKLOAD_CAP")
                planning_truncated = True
                continue
            workload = int(action_meta["weekly_minutes"])
            period_minutes[period] += workload
            selected.append(
                SelectedAction(
                    action_id=decision.action_id,
                    priority=decision.priority,
                    scheduled_period=period,
                    weekly_minutes=workload,
                    reason_codes=decision.reason_codes,
                    supporting_evidence=decision.supporting_evidence,
                    success_criterion=action_meta["success_criterion"],
                    requires_human_contact=decision.requires_human_contact,
                    policy_version=decision.policy_version,
                )
            )
            selected_ids.add(decision.action_id)
        return ConstraintResult(
            selected_actions=tuple(selected),
            rejected_actions=tuple(dict.fromkeys(rejected)),
            constraint_reasons=tuple(reasons),
            truncated=planning_truncated,
        )

    def _topological_order(
        self,
        valid: Mapping[str, PolicyActionDecision],
        metadata: Mapping[str, Any],
        *,
        stage: str,
        dataset_group: str,
    ) -> tuple[PolicyActionDecision, ...]:
        pending = dict(valid)
        emitted: list[PolicyActionDecision] = []
        emitted_ids: set[str] = set()
        while pending:
            ready = [
                item
                for item in pending.values()
                if set(metadata[item.action_id]["prerequisites"]).issubset(emitted_ids)
            ]
            if not ready:
                raise ValueError("planning action prerequisites contain a cycle")
            ready.sort(key=lambda item: self.selector.key(item, stage=stage, dataset_group=dataset_group))
            item = ready[0]
            emitted.append(item)
            emitted_ids.add(item.action_id)
            pending.pop(item.action_id)
        return tuple(emitted)

    def _schedule_period(
        self,
        metadata: Mapping[str, Any],
        *,
        dataset_group: str,
        periods: tuple[str, ...],
        period_minutes: Mapping[str, int],
        prerequisite_periods: tuple[str, ...],
    ) -> str | None:
        preferred = self.config[dataset_group]["period_preferences"].get(metadata["category"], periods[0])
        ordered = tuple(sorted(periods, key=lambda item: (item != preferred, periods.index(item))))
        earliest = max((periods.index(item) for item in prerequisite_periods), default=0)
        workload = int(metadata["weekly_minutes"])
        limit = int(self.config["max_minutes_per_period"])
        for period in ordered:
            if periods.index(period) < earliest:
                continue
            if period_minutes[period] + workload <= limit:
                return period
        return None


__all__ = ["ConstraintResult", "HybridConstraintSolver"]
