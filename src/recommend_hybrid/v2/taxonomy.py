"""Recommendation V2 action taxonomy and coverage audit."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


class ActionRole(str, Enum):
    LEARNED_BEHAVIOUR = "LEARNED_BEHAVIOUR"
    GOVERNANCE_ROUTE = "GOVERNANCE_ROUTE"
    RESEARCH_CANDIDATE = "RESEARCH_CANDIDATE"


@dataclass(frozen=True)
class ActionDefinition:
    action_id: str
    role: ActionRole
    observable_measure: str | None
    modifiable: bool
    human_review_required: bool
    minimum_stage: str
    description: str


ACTION_DEFINITIONS: tuple[ActionDefinition, ...] = (
    ActionDefinition(
        "ASSESSMENT_COMPLETION",
        ActionRole.LEARNED_BEHAVIOUR,
        "assessment_completion_rate",
        True,
        False,
        "EARLY_20",
        "Complete open and valid due assessments.",
    ),
    ActionDefinition(
        "STUDY_REGULARITY",
        ActionRole.LEARNED_BEHAVIOUR,
        "study_regularity_score",
        True,
        False,
        "EARLY_20",
        "Establish regular purposeful study activity.",
    ),
    ActionDefinition(
        "VLE_ENGAGEMENT",
        ActionRole.LEARNED_BEHAVIOUR,
        "vle_active_day_rate",
        True,
        False,
        "EARLY_20",
        "Re-engage with the virtual learning environment.",
    ),
    ActionDefinition(
        "QUIZ_OR_RETRIEVAL_PRACTICE",
        ActionRole.LEARNED_BEHAVIOUR,
        "retrieval_practice_rate",
        True,
        False,
        "EARLY_20",
        "Use short retrieval or quiz practice on studied material.",
    ),
    ActionDefinition(
        "CONTENT_REVIEW",
        ActionRole.LEARNED_BEHAVIOUR,
        "content_review_coverage",
        True,
        False,
        "EARLY_20",
        "Review and consolidate previously studied content.",
    ),
    ActionDefinition(
        "PROGRESS_MONITORING",
        ActionRole.GOVERNANCE_ROUTE,
        None,
        False,
        False,
        "EARLY_20",
        "Do not issue a behavioural action; review at the next landmark.",
    ),
    ActionDefinition(
        "DIAGNOSTIC_CHECK",
        ActionRole.GOVERNANCE_ROUTE,
        None,
        False,
        False,
        "EARLY_20",
        "Gather evidence before selecting a behavioural action.",
    ),
    ActionDefinition(
        "INSTRUCTOR_CONTACT",
        ActionRole.GOVERNANCE_ROUTE,
        None,
        False,
        True,
        "EARLY_35",
        "Route a specific academic support request to the instructor.",
    ),
    ActionDefinition(
        "ADVISOR_ESCALATION",
        ActionRole.GOVERNANCE_ROUTE,
        None,
        False,
        True,
        "EARLY_20",
        "Escalate a critical or safety-sensitive case for human review.",
    ),
    ActionDefinition(
        "ASSESSMENT_TIMELINESS",
        ActionRole.RESEARCH_CANDIDATE,
        "late_submission_rate_to_date",
        True,
        False,
        "EARLY_35",
        "Improve on-time submission behaviour separately from completion.",
    ),
)

ACTION_BY_ID = {row.action_id: row for row in ACTION_DEFINITIONS}
LEARNED_ACTIONS = tuple(
    row.action_id for row in ACTION_DEFINITIONS if row.role is ActionRole.LEARNED_BEHAVIOUR
)
GOVERNANCE_ROUTES = tuple(
    row.action_id for row in ACTION_DEFINITIONS if row.role is ActionRole.GOVERNANCE_ROUTE
)
RESEARCH_CANDIDATES = tuple(
    row.action_id for row in ACTION_DEFINITIONS if row.role is ActionRole.RESEARCH_CANDIDATE
)


def validate_learned_action_order(values: Iterable[str]) -> tuple[str, ...]:
    supplied = tuple(str(value) for value in values)
    if supplied != LEARNED_ACTIONS:
        raise ValueError(
            f"learned action order must be exactly {LEARNED_ACTIONS}, got {supplied}"
        )
    return supplied


def audit_taxonomy(
    rows: pd.DataFrame,
    *,
    stages: Iterable[str],
    action_column: str = "action_id",
    label_column: str = "silver_label",
    group_columns: tuple[str, ...] = ("record_id", "stage"),
    minimum_positive: int = 30,
    maximum_pairwise_phi: float = 0.95,
) -> dict[str, object]:
    """Audit learned-action coverage, support and binary-label redundancy."""

    required = {action_column, label_column, *group_columns}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise KeyError(f"taxonomy audit is missing columns: {missing}")
    frame = rows.loc[rows[action_column].isin(LEARNED_ACTIONS)].copy()
    if frame.empty:
        raise ValueError("taxonomy audit contains no learned actions")
    frame[label_column] = pd.to_numeric(frame[label_column], errors="raise").astype(int)
    frame["positive"] = frame[label_column].gt(0).astype(np.int8)
    stage_set = tuple(str(stage) for stage in stages)

    action_rows: list[dict[str, object]] = []
    for action_id in LEARNED_ACTIONS:
        selected = frame.loc[frame[action_column].eq(action_id)]
        positive = int(selected["positive"].sum())
        represented_stages = sorted(selected["stage"].astype(str).unique().tolist())
        action_rows.append(
            {
                "action_id": action_id,
                "rows": int(len(selected)),
                "positive_rows": positive,
                "positive_rate": float(positive / len(selected)) if len(selected) else 0.0,
                "represented_stages": represented_stages,
                "all_stages_represented": set(stage_set).issubset(represented_stages),
                "minimum_positive_pass": positive >= minimum_positive,
            }
        )

    pivot = frame.pivot_table(
        index=list(group_columns),
        columns=action_column,
        values="positive",
        aggfunc="max",
        fill_value=0,
    ).reindex(columns=LEARNED_ACTIONS, fill_value=0)
    phi_rows: list[dict[str, object]] = []
    maximum_phi = 0.0
    for left_index, left in enumerate(LEARNED_ACTIONS):
        for right in LEARNED_ACTIONS[left_index + 1 :]:
            if pivot[left].nunique() < 2 or pivot[right].nunique() < 2:
                phi = 0.0
            else:
                phi = float(abs(np.corrcoef(pivot[left], pivot[right])[0, 1]))
            maximum_phi = max(maximum_phi, phi)
            phi_rows.append({"left": left, "right": right, "absolute_phi": phi})

    return {
        "status": "PASS"
        if all(row["minimum_positive_pass"] for row in action_rows)
        and maximum_phi <= maximum_pairwise_phi
        else "REVIEW_REQUIRED",
        "learned_action_count": len(LEARNED_ACTIONS),
        "governance_route_count": len(GOVERNANCE_ROUTES),
        "research_candidate_count": len(RESEARCH_CANDIDATES),
        "actions": action_rows,
        "pairwise_redundancy": phi_rows,
        "maximum_absolute_phi": maximum_phi,
        "maximum_allowed_phi": maximum_pairwise_phi,
        "governance_routes_ranked_by_action_head": False,
        "research_candidates_activated": False,
    }


def taxonomy_manifest() -> Mapping[str, object]:
    return {
        "learned_actions": list(LEARNED_ACTIONS),
        "governance_routes": list(GOVERNANCE_ROUTES),
        "research_candidates": list(RESEARCH_CANDIDATES),
        "definitions": [
            {
                "action_id": row.action_id,
                "role": row.role.value,
                "observable_measure": row.observable_measure,
                "modifiable": row.modifiable,
                "human_review_required": row.human_review_required,
                "minimum_stage": row.minimum_stage,
                "description": row.description,
            }
            for row in ACTION_DEFINITIONS
        ],
    }


__all__ = [
    "ACTION_BY_ID",
    "ACTION_DEFINITIONS",
    "ActionDefinition",
    "ActionRole",
    "GOVERNANCE_ROUTES",
    "LEARNED_ACTIONS",
    "RESEARCH_CANDIDATES",
    "audit_taxonomy",
    "taxonomy_manifest",
    "validate_learned_action_order",
]
