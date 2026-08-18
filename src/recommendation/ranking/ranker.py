"""Deterministic feasibility-aware ranking for five action scores."""

from __future__ import annotations

from .scorer import ACTION_ORDER, ACTION_QUALITY

TIE_ORDER = {action: index for index, action in enumerate(ACTION_ORDER)}


def release_status(action_id: str, feasibility_status: str) -> str:
    if feasibility_status == "INFEASIBLE":
        return "NOT_RELEASED"
    if feasibility_status == "UNKNOWN":
        return "NEEDS_VERIFICATION"
    if action_id == "retrieval_practice" or ACTION_QUALITY.get(action_id) == "REVIEW":
        return "REVIEW_REQUIRED"
    return "RELEASED"


def rank_actions(rows: list[dict], *, top_k: int = 3) -> list[dict]:
    ordered = sorted(
        rows,
        key=lambda row: (-float(row["relevance_score"]), -float(row["raw_score"]), TIE_ORDER[row["action_id"]]),
    )
    for index, row in enumerate(ordered, start=1):
        row["rank"] = index
        row["release_status"] = release_status(row["action_id"], row["feasibility_status"])
        row["quality_warning"] = ACTION_QUALITY.get(row["action_id"])
    releasable = [row for row in ordered if row["feasibility_status"] != "INFEASIBLE"]
    for index, row in enumerate(releasable, start=1):
        row["releasable_rank"] = index
    top = releasable[:top_k]
    if not top:
        plan = "NO_ACTION"
    elif top[0]["release_status"] in {"REVIEW_REQUIRED", "NEEDS_VERIFICATION"}:
        plan = "REVIEW"
    else:
        plan = "RECOMMEND"
    for row in ordered:
        row["in_top_k"] = row.get("releasable_rank", 999) <= top_k and row["feasibility_status"] != "INFEASIBLE"
        row["plan_status"] = plan
    return ordered


def top_k_actions(ranked: list[dict], k: int = 3) -> list[str]:
    return [row["action_id"] for row in ranked if row.get("in_top_k")][:k]
