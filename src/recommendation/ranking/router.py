"""Operational recommendation contract: score, attach feasibility, rank, explain."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .ranker import rank_actions, top_k_actions
from .scorer import load_ebm_bundle, score_case


def recommend_case(state_row, models: dict, *, model_version: str, top_k: int = 3) -> dict:
    scored = score_case(state_row, models, model_version=model_version)
    ranked = rank_actions(scored, top_k=top_k)
    return {
        "case_id": str(ranked[0]["case_id"]),
        "plan_status": ranked[0]["plan_status"],
        "top_k": top_k_actions(ranked, top_k),
        "actions": ranked,
    }


def recommend_frame(states: pd.DataFrame, manifest: dict, root: Path, *, top_k: int = 3) -> list[dict]:
    models = load_ebm_bundle(manifest, root)
    version = manifest.get("version")
    return [recommend_case(row, models, model_version=version, top_k=top_k) for _, row in states.sort_values("case_id").iterrows()]
