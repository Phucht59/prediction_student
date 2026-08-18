"""Model-faithful EBM explanations. No LLM prose."""

from __future__ import annotations

from .scorer import ACTION_QUALITY


def compact_explanation(action_row: dict) -> dict:
    return {
        "action_id": action_row["action_id"],
        "score": action_row["relevance_score"],
        "raw_score": action_row["raw_score"],
        "rank": action_row.get("rank"),
        "top_positive_reasons": action_row.get("top_positive_reasons", [])[:3],
        "top_negative_reasons": action_row.get("top_negative_reasons", [])[:2],
        "feasibility_status": action_row["feasibility_status"],
        "release_status": action_row.get("release_status"),
        "quality_warning": action_row.get("quality_warning") or ACTION_QUALITY.get(action_row["action_id"]),
        "model_version": action_row.get("model_version"),
    }
