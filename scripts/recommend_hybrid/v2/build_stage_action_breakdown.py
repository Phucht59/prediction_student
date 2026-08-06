"""Build held-out Recommendation V2 stage/action ranking breakdowns."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.recommend_hybrid.v2.evaluate_full_population import (  # noqa: E402
    DEFAULT_LABELS,
    DEFAULT_LANDMARK,
    DEFAULT_OOF,
    DEFAULT_SIMULATION,
    _prepare_arrays,
)
from src.recommend_hybrid.v2.eligibility import (  # noqa: E402
    EligibilityDecision,
    EligibilityPolicy,
    apply_eligibility_policy,
)
from src.recommend_hybrid.v2.evaluation import ranking_metrics  # noqa: E402
from src.recommend_hybrid.v2.ranking import (  # noqa: E402
    MinMaxNormalizer,
    RankingWeights,
    ranking_baselines,
    utility_scores,
)
from src.recommend_hybrid.v2.taxonomy import LEARNED_ACTIONS  # noqa: E402

DEFAULT_EVIDENCE = ROOT / "artifacts/recommend_hybrid/v2/FULL_POPULATION_EVIDENCE.json"
DEFAULT_OUTPUT = ROOT / "artifacts/recommend_hybrid/v2/STAGE_ACTION_BREAKDOWN.json"
DEFAULT_REPORT = ROOT / "reports/recommend_hybrid/v2/STAGE_ACTION_BREAKDOWN.md"
STAGES = ("EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75")


def _action_breakdown(
    scores: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> list[dict[str, object]]:
    top = np.argmax(scores, axis=1)
    rows: list[dict[str, object]] = []
    for index, action in enumerate(LEARNED_ACTIONS):
        available = mask[:, index]
        positive = available & (target[:, index] > 0)
        selected = available & (top == index)
        correct = selected & positive
        rows.append(
            {
                "action_id": action,
                "available_groups": int(available.sum()),
                "positive_groups": int(positive.sum()),
                "top1_selected": int(selected.sum()),
                "correct_top1": int(correct.sum()),
                "precision_when_selected": float(correct.sum() / max(selected.sum(), 1)),
                "recall_of_positive_groups": float(correct.sum() / max(positive.sum(), 1)),
                "mean_utility_score_when_available": float(scores[available, index].mean())
                if available.any()
                else 0.0,
            }
        )
    return rows


def run(
    *,
    landmark_path: Path,
    oof_path: Path,
    labels_path: Path,
    simulation_path: Path,
    evidence_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, object]:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    frame, arrays, simulation_status = _prepare_arrays(
        landmark_path,
        oof_path,
        labels_path,
        simulation_path,
    )
    validation = frame["protocol_split"].eq("validation").to_numpy()
    test = frame["protocol_split"].eq("test").to_numpy()
    policy = EligibilityPolicy(**evidence["eligibility"]["selected_policy"])
    weights = RankingWeights(**evidence["ranking"]["selected_weights"])
    decisions = apply_eligibility_policy(
        risk_probability=arrays["risk"][test],
        need_score=arrays["maximum_need"][test],
        predictive_entropy=arrays["entropy"][test],
        seed_disagreement=np.zeros(int(test.sum())),
        policy=policy,
    )
    issued = decisions == EligibilityDecision.BEHAVIOURAL_ACTION.value
    test_indices = np.flatnonzero(test)[issued]
    normalizer = MinMaxNormalizer.fit(
        arrays["raw_risk_reduction"][validation],
        arrays["action_mask"][validation],
    )
    reduction = normalizer.transform(arrays["raw_risk_reduction"])
    scores = utility_scores(
        action_probability=arrays["action_probability"][test_indices],
        need_severity=arrays["need"][test_indices],
        simulated_risk_reduction=reduction[test_indices],
        evidence_confidence=arrays["evidence_confidence"][test_indices],
        workload=arrays["workload"][test_indices],
        uncertainty=arrays["uncertainty"][test_indices],
        mask=arrays["action_mask"][test_indices],
        weights=weights,
    )
    target = arrays["action_target"][test_indices]
    mask = arrays["action_mask"][test_indices]
    issued_frame = frame.iloc[test_indices].reset_index(drop=True)

    validation_indices = np.flatnonzero(validation)
    prevalence = (
        arrays["action_target"][validation_indices].sum(axis=0)
        / np.maximum(arrays["action_mask"][validation_indices].sum(axis=0), 1)
    )
    baseline_scores = ranking_baselines(
        action_probability=arrays["action_probability"][test_indices],
        need_severity=arrays["need"][test_indices],
        simulated_risk_reduction=reduction[test_indices],
        evidence_confidence=arrays["evidence_confidence"][test_indices],
        workload=arrays["workload"][test_indices],
        mask=mask,
        prevalence=prevalence,
    )
    overall_baselines = {
        name: ranking_metrics(value, target, mask)
        for name, value in baseline_scores.items()
    }
    best_baseline_name, best_baseline = max(
        overall_baselines.items(),
        key=lambda item: (
            item[1]["precision_at_1"],
            item[1]["ndcg_at_3"],
            item[1]["mrr"],
        ),
    )
    overall = ranking_metrics(scores, target, mask)
    stage_rows: dict[str, object] = {}
    for stage in STAGES:
        selected = issued_frame["stage"].eq(stage).to_numpy()
        if not selected.any():
            stage_rows[stage] = {"issued_groups": 0, "ranking": None, "actions": []}
            continue
        stage_rows[stage] = {
            "issued_groups": int(selected.sum()),
            "ranking": ranking_metrics(scores[selected], target[selected], mask[selected]),
            "actions": _action_breakdown(scores[selected], target[selected], mask[selected]),
        }
    payload = {
        "status": "COMPLETE",
        "simulation_component_status": simulation_status,
        "overall": overall,
        "actions": _action_breakdown(scores, target, mask),
        "stages": stage_rows,
        "baselines": overall_baselines,
        "best_baseline": {
            "name": best_baseline_name,
            "metrics": best_baseline,
        },
        "improvement_over_best_baseline": {
            "precision_at_1": float(overall["precision_at_1"] - best_baseline["precision_at_1"]),
            "ndcg_at_3": float(overall["ndcg_at_3"] - best_baseline["ndcg_at_3"]),
            "mrr": float(overall["mrr"] - best_baseline["mrr"]),
            "pairwise_accuracy": float(
                overall["pairwise_accuracy"] - best_baseline["pairwise_accuracy"]
            ),
        },
        "test_used_for_weight_selection": False,
        "claim_boundary": "SILVER_LABEL_RANKING_CONSISTENCY_NOT_REAL_WORLD_ACCURACY",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Recommendation V2 Stage and Action Breakdown",
        "",
        f"Best baseline: **{best_baseline_name}**.",
        "",
        "| Stage | Issued | P@1 | Recall@1 | Recall@3 | NDCG@3 | MRR | Pairwise |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        row = stage_rows[stage]
        metric = row["ranking"]
        if metric is None:
            lines.append(f"| {stage} | 0 | N/A | N/A | N/A | N/A | N/A | N/A |")
        else:
            lines.append(
                "| {stage} | {issued} | {p1:.4f} | {r1:.4f} | {r3:.4f} | {ndcg:.4f} | {mrr:.4f} | {pairwise:.4f} |".format(
                    stage=stage,
                    issued=row["issued_groups"],
                    p1=metric["precision_at_1"],
                    r1=metric["recall_at_1"],
                    r3=metric["recall_at_3"],
                    ndcg=metric["ndcg_at_3"],
                    mrr=metric["mrr"],
                    pairwise=metric["pairwise_accuracy"],
                )
            )
    lines.extend(
        [
            "",
            "| Action | Available | Positive | Selected top-1 | Precision selected | Recall positive |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["actions"]:
        lines.append(
            "| {action_id} | {available_groups} | {positive_groups} | {top1_selected} | {precision_when_selected:.4f} | {recall_of_positive_groups:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "These metrics measure agreement with scientific silver labels on issued held-out groups. They are not expert agreement or deployed effectiveness.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark", type=Path, default=DEFAULT_LANDMARK)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--simulation", type=Path, default=DEFAULT_SIMULATION)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run(
        landmark_path=args.landmark,
        oof_path=args.oof,
        labels_path=args.labels,
        simulation_path=args.simulation,
        evidence_path=args.evidence,
        output_path=args.output,
        report_path=args.report,
    )
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
