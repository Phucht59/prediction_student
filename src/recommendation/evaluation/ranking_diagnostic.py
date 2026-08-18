"""Panel A OOF ranking diagnostic and shared case-level metric aggregation."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from ..ranking.ranker import rank_actions, top_k_actions
from .metrics import mean_optional, mrr, ndcg_at_k, pairwise_accuracy, precision_at_1, recall_at_k


def _case_reference(group: pd.DataFrame, value_col: str) -> dict[str, float]:
    return {str(row["action_id"]): float(row[value_col]) for _, row in group.iterrows()}


def evaluate_case_ranking(
    *,
    scores: dict[str, tuple[float, float]],
    feasibility: dict[str, str],
    reference: dict[str, float],
    top_k: int = 3,
    relevant_threshold: float = 2.0,
) -> dict:
    rows = []
    for action_id, (raw, clipped) in scores.items():
        rows.append({
            "action_id": action_id,
            "raw_score": raw,
            "relevance_score": clipped,
            "feasibility_status": feasibility.get(action_id, "UNKNOWN"),
        })
    ranked = rank_actions(rows, top_k=top_k)
    ranked_ids = top_k_actions(ranked, top_k)
    labeled_scores = {action: scores[action][1] for action in reference if action in scores}
    labeled_rank = [action for action in ranked_ids if action in reference]
    invalid = sum(1 for action in ranked_ids if feasibility.get(action) == "INFEASIBLE")
    return {
        "ranked": ranked_ids,
        "ndcg@3": ndcg_at_k(reference, labeled_rank, 3),
        "precision@1": precision_at_1(reference, labeled_rank, relevant_threshold=relevant_threshold),
        "recall@3": recall_at_k(reference, labeled_rank, 3, relevant_threshold=relevant_threshold),
        "mrr": mrr(reference, labeled_rank, relevant_threshold=relevant_threshold),
        "pairwise_accuracy": pairwise_accuracy(reference, labeled_scores),
        "invalid_action_rate": float(invalid / max(len(ranked_ids), 1)),
        "coverage": float(len(ranked_ids) > 0),
        "a5_in_top1": bool(ranked_ids and ranked_ids[0] == "retrieval_practice"),
        "a5_in_top3": "retrieval_practice" in ranked_ids,
        "plan_status": ranked[0]["plan_status"] if ranked else "NO_ACTION",
    }


def aggregate_case_metrics(rows: list[dict]) -> dict:
    keys = ["ndcg@3", "precision@1", "recall@3", "mrr", "pairwise_accuracy", "invalid_action_rate", "coverage"]
    summary = {key: mean_optional([row.get(key) for row in rows]) for key in keys}
    summary["n_cases"] = len(rows)
    summary["n_ndcg"] = sum(row.get("ndcg@3") is not None for row in rows)
    summary["a5_top1_rate"] = mean_optional([float(row.get("a5_in_top1", False)) for row in rows])
    summary["a5_top3_rate"] = mean_optional([float(row.get("a5_in_top3", False)) for row in rows])
    summary["review_plan_rate"] = mean_optional([float(row.get("plan_status") == "REVIEW") for row in rows])
    return summary


def panel_a_oof_ranking(oof: pd.DataFrame, feasibility: pd.DataFrame, *, model_col: str = "y_pred_oof") -> dict:
    feas = feasibility.copy()
    feas["case_id"] = feas["case_id"].astype(str)
    if feas["action_id"].isin(["A1", "A2", "A3", "A4", "A5"]).any():
        from ..feasibility.rules_v2 import LEGACY_TO_FINAL
        feas["action_id"] = feas["action_id"].map(lambda value: LEGACY_TO_FINAL.get(str(value), str(value)))
    feas_map = {(str(row.case_id), str(row.action_id)): str(row.feasibility_status) for row in feas.itertuples(index=False)}
    case_rows = []
    for case_id, group in oof.groupby("case_id"):
        reference = _case_reference(group, "y_expected")
        scores = {str(row["action_id"]): (float(row[model_col]), float(max(0.0, min(3.0, row[model_col])))) for _, row in group.iterrows()}
        feasibility_map = {action: feas_map.get((str(case_id), action), "UNKNOWN") for action in scores}
        metrics = evaluate_case_ranking(scores=scores, feasibility=feasibility_map, reference=reference)
        metrics["case_id"] = str(case_id)
        case_rows.append(metrics)
    return {"label": "PANEL_A_OOF_DIAGNOSTIC", "summary": aggregate_case_metrics(case_rows), "cases": case_rows}
