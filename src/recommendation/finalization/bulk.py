"""Deterministic bulk scoring of frozen EBMs over Student State rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..feasibility.rules_v2 import evaluate_feasibility_v2
from ..models.ebm import local_contributions, predict_raw
from ..models.features import ACTION_TO_KEY, APPROVED_FEATURES, encode_state_features
from ..ranking.ranker import rank_actions
from ..weak_supervision.matrix import FINAL_ACTIONS
from . import BUNDLE_VERSION, STATE_VERSION
from .authority import ACTION_DISPLAY, ACTION_STATUS


def _feasibility(row: pd.Series, action_id: str) -> str:
    status, _reason, _source = evaluate_feasibility_v2(row.to_dict(), ACTION_TO_KEY[action_id])
    return status


def score_states(states: pd.DataFrame, models: dict, *, chunk_size: int = 4000) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "FINAL" in set(states["stage"].astype(str)):
        raise ValueError("FINAL stage rows are not eligible")
    features = encode_state_features(states)
    X = features.to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("feature matrix contains NaN/inf")
    raw_by_action = {}
    explain_by_action = {}
    for action_id in FINAL_ACTIONS:
        model = models[action_id]["model"]
        raw = np.empty(len(states), dtype=float)
        intercepts = np.empty(len(states), dtype=float)
        positives = []
        negatives = []
        for start in range(0, len(states), chunk_size):
            end = min(start + chunk_size, len(states))
            chunk = X[start:end]
            raw[start:end] = predict_raw(model, chunk)
            terms, names, intercept = local_contributions(model, chunk)
            intercepts[start:end] = intercept
            for row_terms in terms:
                order = np.argsort(row_terms)
                negatives.append([
                    {"term": str(names[i]), "contribution": float(row_terms[i])}
                    for i in order[:2] if row_terms[i] < 0
                ])
                positives.append([
                    {"term": str(names[i]), "contribution": float(row_terms[i])}
                    for i in order[::-1][:3] if row_terms[i] > 0
                ])
        raw_by_action[action_id] = raw
        explain_by_action[action_id] = (intercepts, positives, negatives)
    score_rows = []
    plan_rows = []
    for index, state in enumerate(states.itertuples(index=False)):
        values = state._asdict() if hasattr(state, "_asdict") else dict(state._asdict()) if False else None
        series = states.iloc[index]
        ranked_input = []
        for action_id in FINAL_ACTIONS:
            raw = float(raw_by_action[action_id][index])
            ranked_input.append({
                "action_id": action_id,
                "raw_score": raw,
                "relevance_score": float(np.clip(raw, 0, 3)),
                "feasibility_status": _feasibility(series, action_id),
            })
        ranked = rank_actions(ranked_input, top_k=3)
        for item in ranked:
            intercepts, positives, negatives = explain_by_action[item["action_id"]]
            score_rows.append({
                "case_id": str(series["case_id"]),
                "enrollment_identity": str(series.get("enrollment_identity") or series.get("record_id")),
                "student_id": str(series.get("student_id")),
                "module": str(series.get("module")),
                "presentation": str(series.get("presentation")),
                "stage": str(series.get("stage")),
                "action_id": item["action_id"],
                "display_name": ACTION_DISPLAY[item["action_id"]],
                "raw_score": item["raw_score"],
                "relevance_score": item["relevance_score"],
                "rank": item["rank"],
                "feasibility_status": item["feasibility_status"],
                "release_status": item["release_status"],
                "quality_warning": ACTION_STATUS[item["action_id"]],
                "in_top_k": item["in_top_k"],
                "plan_status": item["plan_status"],
                "intercept": float(intercepts[index]),
                "top_positive_reasons": positives[index],
                "top_negative_reasons": negatives[index],
                "bundle_version": BUNDLE_VERSION,
                "state_version": STATE_VERSION,
            })
        plan_rows.append({
            "case_id": str(series["case_id"]),
            "enrollment_identity": str(series.get("enrollment_identity") or series.get("record_id")),
            "student_id": str(series.get("student_id")),
            "module": str(series.get("module")),
            "presentation": str(series.get("presentation")),
            "stage": str(series.get("stage")),
            "risk_probability": float(series.get("risk_probability")),
            "plan_status": ranked[0]["plan_status"],
            "top1": next((row["action_id"] for row in ranked if row.get("in_top_k")), None),
            "top_actions": [row["action_id"] for row in ranked if row.get("in_top_k")][:3],
            "bundle_version": BUNDLE_VERSION,
            "state_version": STATE_VERSION,
        })
        del values
    scores = pd.DataFrame(score_rows)
    plans = pd.DataFrame(plan_rows)
    if scores["case_id"].nunique() != len(states):
        raise ValueError("bulk inference lost cases")
    if len(scores) != len(states) * 5:
        raise ValueError("bulk inference must emit five action scores per case")
    if scores[["raw_score", "relevance_score"]].isna().any().any():
        raise ValueError("NaN scores are not allowed")
    return scores, plans
