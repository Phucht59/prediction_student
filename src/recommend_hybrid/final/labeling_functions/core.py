from __future__ import annotations
from dataclasses import dataclass
from typing import Any

ABSTAIN = -1

@dataclass(frozen=True)
class LabelVote:
    value: int
    source_id: str
    family: str
    version: str = "v2"
    def __post_init__(self) -> None:
        if self.value not in (ABSTAIN, 0, 1, 2, 3):
            raise ValueError("weak labels must be ABSTAIN or ordinal values 0..3")

def literature_vote(_: dict[str, Any], __: str) -> LabelVote:
    return LabelVote(ABSTAIN, "literature_directional_v2", "literature")

def behavior_vote(row: dict[str, Any], action: str) -> LabelVote:
    if action == "RECOVER_ENGAGEMENT" and row.get("inactivity_streak") is not None:
        return LabelVote(3 if row["inactivity_streak"] >= 7 else 1, "oulad_inactivity_v2", "behavior")
    if action == "STUDY_REGULARITY" and row.get("regularity_score") is not None:
        return LabelVote(3 if row["regularity_score"] < 0.4 else 1, "oulad_regularity_v2", "behavior")
    return LabelVote(ABSTAIN, "oulad_behavior_v2", "behavior")

def feasibility_vote(row: dict[str, Any], action: str) -> LabelVote:
    eligible = bool(row.get("eligible", False))
    return LabelVote(1 if eligible else 0, "policy_feasibility_v2", "policy")
