"""Append-safe persistence for replayable recommend_hybrid plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .common.plan_contracts import LearningPlan


class PlanRepository(Protocol):
    def save(self, plan: LearningPlan) -> LearningPlan: ...
    def retrieve(self, plan_id: str) -> LearningPlan | None: ...
    def replay(self, plan_id: str) -> LearningPlan | None: ...


class InMemoryPlanRepository:
    def __init__(self) -> None:
        self._plans: dict[str, dict] = {}

    def save(self, plan: LearningPlan) -> LearningPlan:
        payload = plan.to_dict()
        current = self._plans.get(plan.plan_id)
        if current is not None and current != payload:
            raise ValueError("append-safe persistence refuses plan overwrite")
        self._plans.setdefault(plan.plan_id, payload)
        return plan

    def retrieve(self, plan_id: str) -> LearningPlan | None:
        payload = self._plans.get(plan_id)
        return LearningPlan.from_dict(payload) if payload is not None else None

    def replay(self, plan_id: str) -> LearningPlan | None:
        return self.retrieve(plan_id)


class JsonPlanRepository:
    """Versioned file store used by CLI; one immutable JSON document per plan ID."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def save(self, plan: LearningPlan) -> LearningPlan:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{plan.plan_id}.json"
        payload = json.dumps(plan.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise ValueError("append-safe persistence refuses plan overwrite")
            return plan
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        return plan

    def retrieve(self, plan_id: str) -> LearningPlan | None:
        path = self.directory / f"{plan_id}.json"
        if not path.exists():
            return None
        return LearningPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def replay(self, plan_id: str) -> LearningPlan | None:
        return self.retrieve(plan_id)


class PostgresPlanRepository:
    """Append-only adapter for the existing recommendation.plan/action JSONB schema."""

    def __init__(self, connection, *, risk_profile_id: str, policy_id: str) -> None:
        self.connection = connection
        self.risk_profile_id = risk_profile_id
        self.policy_id = policy_id

    def save(self, plan: LearningPlan) -> LearningPlan:
        from psycopg2.extras import Json

        payload = plan.to_dict()
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT payload FROM recommendation.plan WHERE plan_id=%s", (plan.plan_id,))
            existing = cursor.fetchone()
            if existing is not None:
                if existing[0] != payload:
                    raise ValueError("append-safe persistence refuses legacy plan overwrite")
                return plan
            cursor.execute(
                "SELECT COALESCE(MAX(revision_no),0)+1 FROM recommendation.plan WHERE risk_profile_id=%s",
                (self.risk_profile_id,),
            )
            revision = int(cursor.fetchone()[0])
            priority = plan.selected_actions[0].priority.value if plan.selected_actions else "NOT_APPLICABLE"
            cursor.execute(
                """INSERT INTO recommendation.plan(
                    plan_id,risk_profile_id,policy_id,revision_no,priority,goal,rationale,status,payload,checksum
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    encode(digest(convert_to(%s,'UTF8'),'sha256'),'hex'))""",
                (
                    plan.plan_id,
                    self.risk_profile_id,
                    self.policy_id,
                    revision,
                    priority,
                    "Evidence-based learning support",
                    plan.explanation.routing,
                    plan.automation_status.value,
                    Json(payload),
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                ),
            )
            for index, action in enumerate(plan.selected_actions, start=1):
                action_payload = action.to_dict()
                cursor.execute(
                    """INSERT INTO recommendation.action(
                        plan_id,action_code,week_no,priority,workload_minutes,status,action_text,payload,checksum
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,
                        encode(digest(convert_to(%s,'UTF8'),'sha256'),'hex'))""",
                    (
                        plan.plan_id,
                        action.action_id,
                        plan.plan_periods.index(action.scheduled_period),
                        index,
                        action.weekly_minutes,
                        "planned",
                        action.success_criterion,
                        Json(action_payload),
                        json.dumps(action_payload, sort_keys=True, separators=(",", ":")),
                    ),
                )
        return plan

    def retrieve(self, plan_id: str) -> LearningPlan | None:
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT payload FROM recommendation.plan WHERE plan_id=%s", (plan_id,))
            row = cursor.fetchone()
        return LearningPlan.from_dict(row[0]) if row else None

    def replay(self, plan_id: str) -> LearningPlan | None:
        return self.retrieve(plan_id)


__all__ = [
    "InMemoryPlanRepository",
    "JsonPlanRepository",
    "PlanRepository",
    "PostgresPlanRepository",
]
