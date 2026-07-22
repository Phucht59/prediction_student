from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    hamming_loss,
    precision_recall_fscore_support,
)

from .engine import validate_recommendation
from .taxonomy import ACTION_TAXONOMY, POLICY_VERSION, WEEKLY_MINUTES_CAP


EXPERT_COLUMNS = [
    "reviewer_id",
    "case_id",
    "blinded_option",
    "recommended_actions",
    "relevant_actions",
    "missing_important_action",
    "unnecessary_action",
    "priority_rating",
    "correct_priority",
    "feasibility",
    "personalization",
    "clarity",
    "safety",
    "overall_usefulness",
    "advisor_escalation_correct",
    "decision",
    "comments",
]


def automatic_metrics(
    recommendations: list[dict[str, Any]],
    replayed: list[dict[str, Any]],
    latencies_ms: list[float],
) -> dict[str, Any]:
    if len(recommendations) != len(replayed) or len(recommendations) != len(latencies_ms):
        raise ValueError("Recommendation metric inputs are not aligned")
    for recommendation in recommendations:
        validate_recommendation(recommendation)
    total = max(1, len(recommendations))
    generated = [row for row in recommendations if row["abstention_status"] == "GENERATED"]
    action_ids = [
        [action["action_id"] for action in row["ranked_actions"]] for row in recommendations
    ]
    conflicts = 0
    duplicates = sum(len(actions) != len(set(actions)) for actions in action_ids)
    workload = sum(row["weekly_minutes"] > WEEKLY_MINUTES_CAP for row in recommendations)
    missing_reason = sum(
        not action["reason_codes"]
        for row in recommendations
        for action in row["ranked_actions"]
    )
    missing_lineage = sum(
        not all(
            row.get(name)
            for name in ("prediction_set_id", "deep_model_registry_id", "ml_model_registry_id")
        )
        for row in recommendations
    )
    post_cutoff = sum(bool(row["post_cutoff_features_used"]) for row in recommendations)
    replay_equal = [first == second for first, second in zip(recommendations, replayed)]
    jaccard = []
    rank_equal = []
    for first, second in zip(recommendations, replayed):
        left = [action["action_id"] for action in first["ranked_actions"]]
        right = [action["action_id"] for action in second["ranked_actions"]]
        union = set(left) | set(right)
        jaccard.append(len(set(left) & set(right)) / len(union) if union else 1.0)
        rank_equal.append(left == right)
    disagreement = sum(bool(row["disagreement"]["label_disagreement"]) for row in recommendations)
    action_counts = {action: 0 for action in ACTION_TAXONOMY}
    for actions in action_ids:
        for action in actions:
            action_counts[action] += 1
    risk_counts = {
        name: sum(row["risk_level"] == name for row in recommendations)
        for name in ("low", "medium", "high")
    }
    confidence_counts = {
        name: sum(row["confidence_level"] == name for row in recommendations)
        for name in ("low", "medium", "high")
    }
    values = np.asarray(latencies_ms, dtype=float)
    return {
        "status": "PASS"
        if not any((conflicts, duplicates, workload, missing_reason, missing_lineage, post_cutoff))
        and all(replay_equal)
        else "FAIL",
        "technical_correctness": {
            "conflict_rate": conflicts / total,
            "duplicate_action_rate": duplicates / total,
            "workload_violation_rate": workload / total,
            "missing_reason_code_rate": missing_reason / total,
            "missing_model_lineage_rate": missing_lineage / total,
            "post_cutoff_leakage_rate": post_cutoff / total,
            "deterministic_replay_rate": float(np.mean(replay_equal)),
            "advisor_review_contract_coverage": float(
                np.mean(
                    [
                        (not row["disagreement"]["label_disagreement"])
                        or row["requires_advisor_review"]
                        or row["agreement_score"] >= 0.75
                        for row in recommendations
                    ]
                )
            ),
            "revision_history_coverage": float(
                np.mean([bool(row.get("revision_history")) for row in recommendations])
            ),
        },
        "operational": {
            "recommendation_coverage": len(generated) / total,
            "abstention_rate": 1 - len(generated) / total,
            "advisor_escalation_rate": float(
                np.mean([row["requires_advisor_review"] for row in recommendations])
            ),
            "deep_ml_disagreement_rate": disagreement / total,
            "normal_plan_rate": float(
                np.mean(
                    [
                        row["abstention_status"] == "GENERATED"
                        and not row["requires_advisor_review"]
                        for row in recommendations
                    ]
                )
            ),
            "average_actions": float(np.mean([len(actions) for actions in action_ids])),
            "average_weekly_minutes": float(
                np.mean([row["weekly_minutes"] for row in recommendations])
            ),
            "generation_latency_ms_mean": float(values.mean()),
            "generation_latency_ms_p95": float(np.percentile(values, 95)),
            "cases_by_risk_level": risk_counts,
            "cases_by_confidence_level": confidence_counts,
            "cases_by_action_category": action_counts,
        },
        "stability": {
            "exact_output_replay_rate": float(np.mean(replay_equal)),
            "action_set_jaccard": float(np.mean(jaccard)),
            "rank_stability": float(np.mean(rank_equal)),
            "plan_version_consistency": float(
                np.mean(
                    [
                        first["policy_version"] == second["policy_version"] == POLICY_VERSION
                        for first, second in zip(recommendations, replayed)
                    ]
                )
            ),
        },
    }


def write_review_template(casebook: pd.DataFrame, path: Path, reviewer_id: str) -> None:
    rows = []
    for row in casebook.itertuples(index=False):
        rows.append(
            {
                "reviewer_id": reviewer_id,
                "case_id": row.case_id,
                "blinded_option": row.blinded_option,
                "recommended_actions": row.recommended_actions,
                **{column: "" for column in EXPERT_COLUMNS[4:]},
            }
        )
    pd.DataFrame(rows, columns=EXPERT_COLUMNS).to_csv(path, index=False)


def _parse_actions(value: object) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {item.strip() for item in str(value).split("|") if item.strip()}


def expert_evaluation(
    reviewer_1: Path,
    reviewer_2: Path,
    *,
    blinding_key: dict[str, str],
) -> dict[str, Any]:
    if not reviewer_1.is_file() or not reviewer_2.is_file():
        return {"status": "PENDING_EXPERT_LABELS", "synthetic_ratings": False}
    frames = [pd.read_csv(path) for path in (reviewer_1, reviewer_2)]
    required_rating = ["relevant_actions", "decision", "advisor_escalation_correct"]
    if any(frame[required_rating].isna().any().any() for frame in frames):
        return {"status": "PENDING_EXPERT_LABELS", "synthetic_ratings": False}
    combined = pd.concat(frames, ignore_index=True)
    labels = list(ACTION_TAXONOMY)
    predicted_matrix = []
    target_matrix = []
    top1_hits = []
    top3_hits = []
    for row in combined.itertuples(index=False):
        predicted = list(_parse_actions(row.recommended_actions))
        relevant = _parse_actions(row.relevant_actions)
        predicted_matrix.append([int(label in predicted) for label in labels])
        target_matrix.append([int(label in relevant) for label in labels])
        top1_hits.append(bool(predicted[:1] and relevant & set(predicted[:1])))
        top3_hits.append(bool(relevant & set(predicted[:3])))
    predicted_array = np.asarray(predicted_matrix)
    target_array = np.asarray(target_matrix)
    micro = precision_recall_fscore_support(
        target_array, predicted_array, average="micro", zero_division=0
    )
    macro = precision_recall_fscore_support(
        target_array, predicted_array, average="macro", zero_division=0
    )
    score_columns = [
        "feasibility",
        "personalization",
        "clarity",
        "safety",
        "overall_usefulness",
    ]
    score_summary = {}
    for column in score_columns:
        values = pd.to_numeric(combined[column], errors="raise").to_numpy(dtype=float)
        score_summary[column] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
            "at_least_4_rate": float((values >= 4).mean()),
        }
    decisions = combined.decision.astype(str)
    decision_values = ["approve", "minor_modification", "major_modification", "reject"]
    decision_metrics = {name + "_rate": float((decisions == name).mean()) for name in decision_values}
    decision_metrics["approve_or_minor_rate"] = float(
        decisions.isin(["approve", "minor_modification"]).mean()
    )
    reviewer_decisions = frames[0].decision.astype(str).to_numpy()
    reviewer_2_decisions = frames[1].decision.astype(str).to_numpy()
    return {
        "status": "COMPLETE_REAL_EXPERT_LABELS",
        "synthetic_ratings": False,
        "multi_label": {
            "hamming_accuracy": float(1 - hamming_loss(target_array, predicted_array)),
            "exact_match_accuracy": float(
                np.mean(np.all(target_array == predicted_array, axis=1))
            ),
            "micro_precision": float(micro[0]),
            "micro_recall": float(micro[1]),
            "micro_f1": float(micro[2]),
            "macro_precision": float(macro[0]),
            "macro_recall": float(macro[1]),
            "macro_f1": float(macro[2]),
            "top1_recall": float(np.mean(top1_hits)),
            "top3_recall": float(np.mean(top3_hits)),
        },
        "expert_scores": score_summary,
        "decision": decision_metrics,
        "reviewer_agreement": {
            "decision_cohen_kappa": float(
                cohen_kappa_score(reviewer_decisions, reviewer_2_decisions)
            ),
            "decision_agreement": float(
                accuracy_score(reviewer_decisions, reviewer_2_decisions)
            ),
        },
        "blinding_key_entries": len(blinding_key),
    }


__all__ = [
    "EXPERT_COLUMNS",
    "automatic_metrics",
    "expert_evaluation",
    "write_review_template",
]
