"""Diagnose recommendability gating and conditional action ranking separately.

This script does not train a model and does not alter the registered silver
labels. It decomposes the completed hybrid-only OOF result into:

1. Stage A recommendability: whether any positive future action exists.
2. Stage B conditional ranking: whether the selected action is positive when a
   positive action exists in the group.
3. End-to-end selective recommendation performance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / "configs/recommend_hybrid/two_stage_v3_protocol.yaml"
CANDIDATE_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/hybrid_only_final/dataset/candidate_rows.parquet"
)
OOF_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/hybrid_only_final/evaluation/OOF_PREDICTIONS.parquet"
)
V21_PATH = (
    ROOT
    / "artifacts/recommend_hybrid/outcome_grounded_v2_1/final_oof/NESTED_OOF_RESULTS.json"
)
OUT = ROOT / "artifacts/recommend_hybrid/two_stage_v3"
REPORT = ROOT / "reports/recommend_hybrid/TWO_STAGE_V3_DIAGNOSTIC.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(score)
    y = y_true[mask].astype(int)
    s = score[mask].astype(float)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy(dtype=float)
    positive_rank_sum = float(ranks[y == 1].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(score)
    y = y_true[mask].astype(int)
    s = score[mask].astype(float)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    ranked = y[order]
    cumulative = np.cumsum(ranked)
    precision = cumulative / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positives)


def _best_precision_at_recall(
    y_true: np.ndarray,
    score: np.ndarray,
    minimum_recall: float,
) -> dict[str, float | int | None]:
    mask = np.isfinite(score)
    y = y_true[mask].astype(int)
    s = score[mask].astype(float)
    positives = int(y.sum())
    if positives == 0:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "threshold": None,
            "issued": 0,
        }
    order = np.argsort(-s, kind="stable")
    y = y[order]
    s = s[order]
    cumulative_true = np.cumsum(y)
    issued = np.arange(1, len(y) + 1)
    precision = cumulative_true / issued
    recall = cumulative_true / positives
    threshold_boundary = np.r_[s[1:] != s[:-1], True]
    eligible = threshold_boundary & (recall >= minimum_recall)
    if not eligible.any():
        return {
            "precision": 0.0,
            "recall": float(recall[-1]),
            "threshold": float(s[-1]),
            "issued": int(len(y)),
        }
    indices = np.flatnonzero(eligible)
    best = indices[np.argmax(precision[indices])]
    return {
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "threshold": float(s[best]),
        "issued": int(issued[best]),
    }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _stage_breakdown(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage, group in frame.groupby("stage", sort=True):
        issued = group["issued"].astype(bool)
        positive = group["group_has_positive"].astype(bool)
        correct = issued & group["silver_positive"].astype(bool)
        issued_positive = issued & positive
        rows.append(
            {
                "stage": str(stage),
                "groups": int(len(group)),
                "positive_groups": int(positive.sum()),
                "issued_groups": int(issued.sum()),
                "issued_positive_groups": int(issued_positive.sum()),
                "false_issue_groups": int((issued & ~positive).sum()),
                "correct_issued_actions": int(correct.sum()),
                "recommendability_precision": _safe_ratio(
                    int(issued_positive.sum()), int(issued.sum())
                ),
                "recommendability_recall": _safe_ratio(
                    int(issued_positive.sum()), int(positive.sum())
                ),
                "conditional_action_precision_issued_positive": _safe_ratio(
                    int(correct.sum()), int(issued_positive.sum())
                ),
                "conditional_action_precision_all_positive": float(
                    group.loc[positive, "silver_positive"].mean()
                )
                if positive.any()
                else 0.0,
                "end_to_end_precision": _safe_ratio(
                    int(correct.sum()), int(issued.sum())
                ),
            }
        )
    return rows


def _action_breakdown(frame: pd.DataFrame) -> list[dict[str, Any]]:
    issued = frame[frame["issued"] == 1].copy()
    rows: list[dict[str, Any]] = []
    for action, group in issued.groupby("action_family", sort=True):
        positive_group = group["group_has_positive"].astype(bool)
        rows.append(
            {
                "action_family": str(action),
                "issued": int(len(group)),
                "correct": int(group["silver_positive"].sum()),
                "precision": float(group["silver_positive"].mean()),
                "issued_positive_groups": int(positive_group.sum()),
                "conditional_precision_given_positive_group": float(
                    group.loc[positive_group, "silver_positive"].mean()
                )
                if positive_group.any()
                else 0.0,
            }
        )
    return rows


def _historical_v21() -> dict[str, Any]:
    if not V21_PATH.exists():
        return {"available": False}
    payload = json.loads(V21_PATH.read_text(encoding="utf-8"))
    folds = payload.get("fold_metrics", {})
    precision = {
        str(fold): float(metrics["model_score"]["precision_at_1"])
        for fold, metrics in folds.items()
        if "model_score" in metrics
    }
    return {
        "available": bool(precision),
        "model_precision_at_1_by_fold": precision,
        "mean_model_precision_at_1": float(np.mean(list(precision.values())))
        if precision
        else None,
        "claim_boundary": payload.get("claim_boundary"),
    }


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    candidates = pd.read_parquet(CANDIDATE_PATH)
    oof = pd.read_parquet(OOF_PATH)

    required_oof = {
        "group_id",
        "base_record_id",
        "stage",
        "action_family",
        "issued",
        "silver_positive",
        "group_has_positive",
        "hybrid_score",
        "top_margin",
        "risk_reduction",
        "risk_uncertainty",
        "evidence_strength",
    }
    missing = required_oof - set(oof.columns)
    if missing:
        raise RuntimeError(f"OOF diagnostic columns missing: {sorted(missing)}")

    derived_positive = (
        candidates.groupby("group_id", sort=False)["silver_positive"].max().astype(int)
    )
    joined = oof.merge(
        derived_positive.rename("derived_group_has_positive"),
        left_on="group_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    if joined["derived_group_has_positive"].isna().any():
        raise RuntimeError("OOF groups are missing from candidate rows")
    mismatch = int(
        (
            joined["group_has_positive"].astype(int)
            != joined["derived_group_has_positive"].astype(int)
        ).sum()
    )
    if mismatch:
        raise RuntimeError(f"group target mismatch count: {mismatch}")

    issued = joined["issued"].astype(bool)
    positive = joined["group_has_positive"].astype(bool)
    correct = issued & joined["silver_positive"].astype(bool)
    issued_positive = issued & positive
    issued_count = int(issued.sum())
    positive_count = int(positive.sum())
    issued_positive_count = int(issued_positive.sum())
    correct_count = int(correct.sum())
    false_issue_count = int((issued & ~positive).sum())

    signals = {
        "hybrid_score": joined["hybrid_score"].to_numpy(dtype=float),
        "top_margin": joined["top_margin"].to_numpy(dtype=float),
        "risk_reduction": joined["risk_reduction"].to_numpy(dtype=float),
        "certainty": -joined["risk_uncertainty"].to_numpy(dtype=float),
        "evidence_strength": joined["evidence_strength"].to_numpy(dtype=float),
    }
    y_group = positive.to_numpy(dtype=int)
    signal_results = {
        name: {
            "roc_auc": _roc_auc(y_group, values),
            "average_precision": _average_precision(y_group, values),
            "best_at_recall_0_50": _best_precision_at_recall(
                y_group, values, 0.50
            ),
        }
        for name, values in signals.items()
    }

    overall = {
        "groups": int(len(joined)),
        "learners": int(joined["base_record_id"].nunique()),
        "positive_groups": positive_count,
        "positive_group_prevalence": _safe_ratio(
            positive_count, int(len(joined))
        ),
        "issued_groups": issued_count,
        "issued_positive_groups": issued_positive_count,
        "false_issue_groups": false_issue_count,
        "correct_issued_actions": correct_count,
        "recommendability_precision": _safe_ratio(
            issued_positive_count, issued_count
        ),
        "recommendability_recall": _safe_ratio(
            issued_positive_count, positive_count
        ),
        "conditional_action_precision_issued_positive": _safe_ratio(
            correct_count, issued_positive_count
        ),
        "conditional_action_precision_all_positive": float(
            joined.loc[positive, "silver_positive"].mean()
        )
        if positive.any()
        else 0.0,
        "end_to_end_precision_at_1": _safe_ratio(correct_count, issued_count),
        "perfect_ranker_same_gate_precision_ceiling": _safe_ratio(
            issued_positive_count, issued_count
        ),
        "decomposition_product": _safe_ratio(
            issued_positive_count, issued_count
        )
        * _safe_ratio(correct_count, issued_positive_count),
    }

    result = {
        "schema_version": "two_stage_v3_diagnostic_v1",
        "status": "COMPLETE",
        "claim_boundary": protocol["claim_boundary"],
        "labels_changed": False,
        "models_trained": False,
        "overall": overall,
        "scalar_recommendability_signals": signal_results,
        "per_stage": _stage_breakdown(joined),
        "per_action": _action_breakdown(joined),
        "historical_v2_1": _historical_v21(),
        "interpretation": {
            "dominant_failure": (
                "RECOMMENDABILITY_GATE"
                if false_issue_count > issued_positive_count - correct_count
                else "CONDITIONAL_ACTION_RANKING"
            ),
            "registered_end_to_end_80_percent_supported": bool(
                overall["end_to_end_precision_at_1"] >= 0.80
            ),
            "conditional_80_percent_supported": bool(
                overall["conditional_action_precision_issued_positive"] >= 0.80
            ),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    _write_json(OUT / "DIAGNOSTIC.json", result)

    lines = [
        "# Two-stage V3 diagnostic",
        "",
        "## Overall decomposition",
        "",
        f"- Groups: {overall['groups']}",
        f"- Positive groups: {overall['positive_groups']}",
        f"- Positive prevalence: {overall['positive_group_prevalence']:.4f}",
        f"- Issued groups: {overall['issued_groups']}",
        f"- Issued positive groups: {overall['issued_positive_groups']}",
        f"- False issues: {overall['false_issue_groups']}",
        f"- Correct issued actions: {overall['correct_issued_actions']}",
        f"- Stage A precision: {overall['recommendability_precision']:.4f}",
        f"- Stage A recall: {overall['recommendability_recall']:.4f}",
        f"- Stage B conditional Precision@1: {overall['conditional_action_precision_issued_positive']:.4f}",
        f"- End-to-end Precision@1: {overall['end_to_end_precision_at_1']:.4f}",
        f"- Perfect-ranker ceiling with the same gate: {overall['perfect_ranker_same_gate_precision_ceiling']:.4f}",
        "",
        "## Scalar recommendability signals",
        "",
    ]
    for name, metrics in signal_results.items():
        best = metrics["best_at_recall_0_50"]
        lines.append(
            f"- {name}: ROC-AUC={metrics['roc_auc']:.4f}, "
            f"AP={metrics['average_precision']:.4f}, "
            f"best precision at recall>=0.50={best['precision']:.4f} "
            f"(recall={best['recall']:.4f})"
        )
    lines.extend(["", "## Per stage", ""])
    for row in result["per_stage"]:
        lines.append(
            f"- {row['stage']}: gate precision={row['recommendability_precision']:.4f}, "
            f"gate recall={row['recommendability_recall']:.4f}, "
            f"conditional P@1={row['conditional_action_precision_issued_positive']:.4f}, "
            f"end-to-end P@1={row['end_to_end_precision']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            f"- Dominant failure: `{result['interpretation']['dominant_failure']}`",
            f"- End-to-end 80% supported: `{result['interpretation']['registered_end_to_end_80_percent_supported']}`",
            f"- Conditional 80% supported: `{result['interpretation']['conditional_80_percent_supported']}`",
            "",
            "Conditional metrics must not be reported as unconditional end-to-end recommendation accuracy.",
            "",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
