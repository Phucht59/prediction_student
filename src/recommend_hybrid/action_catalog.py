"""Controlled bilingual action catalog loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .contracts import CandidateAction, Stage
from .exceptions import CatalogValidationError

OBSERVED_EVIDENCE_FIELDS = frozenset(
    {
        "activity_level",
        "inactivity_streak",
        "assessment_progress",
        "grade_trend",
        "course_progress",
        "recent_activity_trend",
        "total_activity",
        "recent_activity",
        "average_activity",
        "assessments_due",
        "assessments_completed",
    }
)


class ActionCatalog:
    def __init__(self, actions: tuple[CandidateAction, ...], version: str) -> None:
        self.actions = actions
        self.version = version
        self.validate()

    @classmethod
    def load(cls, path: Path) -> "ActionCatalog":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        version = str(payload["catalog_version"])
        actions = tuple(
            CandidateAction(
                action_id=row["action_id"],
                category=row["category"],
                title=row["title_en"],
                description=row["description_en"],
                weekly_minutes=int(row["weekly_minutes"]),
                applicable_stages=tuple(Stage(stage) for stage in row["applicable_stages"]),
                required_evidence=tuple(row.get("required_evidence", [])),
                prerequisites=tuple(row.get("prerequisites", [])),
                contraindications=tuple(row.get("contraindications", [])),
                requires_human_review=bool(row["requires_human_review"]),
                success_criterion=row["success_criterion"],
                active=bool(row["active"]),
                catalog_version=version,
            )
            for row in payload["actions"]
        )
        return cls(actions, version)

    def validate(self) -> None:
        ids = [action.action_id for action in self.actions]
        if len(ids) != len(set(ids)):
            raise CatalogValidationError("duplicate action ID")
        known = set(ids)
        for action in self.actions:
            undefined = set(action.prerequisites) - known
            if undefined:
                raise CatalogValidationError(f"undefined prerequisites for {action.action_id}")
            invalid_evidence = set(action.required_evidence) - OBSERVED_EVIDENCE_FIELDS
            if invalid_evidence:
                raise CatalogValidationError(f"undefined evidence for {action.action_id}")
        graph = {action.action_id: set(action.prerequisites) for action in self.actions}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise CatalogValidationError("cyclic action prerequisite")
            if node in visited:
                return
            visiting.add(node)
            for dependency in graph[node]:
                visit(dependency)
            visiting.remove(node)
            visited.add(node)

        for action_id in ids:
            visit(action_id)

    def by_id(self, action_id: str) -> CandidateAction:
        for action in self.actions:
            if action.action_id == action_id:
                return action
        raise KeyError(action_id)


__all__ = ["ActionCatalog", "OBSERVED_EVIDENCE_FIELDS"]
