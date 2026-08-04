"""Post-hoc feasibility audit for the failed Two-Stage V4 execution.

This script does not train a model and must never authorize release. It uses
held-out OOF labels only to estimate optimistic score/threshold ceilings and to
separate ranking capacity from recommendability capacity.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/two_stage_v4"
OOF_PATH = OUT / "final_oof/OOF_PREDICTIONS.parquet"
RELEASE_PATH = OUT / "TWO_STAGE_V4_RELEASE.json"
PROTOCOL_PATH = ROOT / "configs/recommend_hybrid/two_stage_v4_protocol.yaml"
AUDIT_PATH = OUT / "V4_FEASIBILITY_AUDIT.json"
REPORT_PATH = ROOT / "reports/recommend_hybrid/TWO_STAGE_V4_FEASIBILITY_AUDIT.md"
STAGES = ("EARLY_20", "EARLY_35", "MIDDLE_50")
ACTION_COUNT = 5
EPSILON = 1.0e-12


@dataclass(frozen=True)
class FrontierPoint:
    threshold: float
    issued: int
    issued_positive: int
    correct: int
    precision: float
    coverage: float
    stage_a_precision: float
    conditional_precision: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "threshold": self.threshold,
            "issued": self.issued,
            "issued_positive": self.issued_positive,
            "correct": self.correct,
            "precision": self.precision,
            "coverage": self.coverage,
            "stage_a_precision": self.stage_a_precision,
            "conditional_precision": self.conditional_precision,
        }


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _frontier(
    scores: np.ndarray,
    positive: np.ndarray,
    correct: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
    minimum_coverage: float = 0.50,
) -> tuple[list[FrontierPoint], FrontierPoint | None]:
    score = np.asarray(scores, dtype=np.float64)
    target = np.asarray(positive, dtype=bool)
    hit = np.asarray(correct, dtype=bool)
    allowed = np.ones(len(score), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    indexes = np.where(allowed)[0]
    total_positive = int(target.sum())
    if not len(indexes) or not total_positive:
        return [], None
    order = indexes[np.argsort(-score[indexes], kind="stable")]
    ordered_score = score[order]
    cumulative_positive = np.cumsum(target[order], dtype=np.int64)
    cumulative_correct = np.cumsum(hit[order], dtype=np.int64)
    boundaries = np.r_[np.where(np.diff(ordered_score) != 0.0)[0], len(order) - 1]
    points: list[FrontierPoint] = []
    best: FrontierPoint | None = None
    for boundary in boundaries:
        issued = int(boundary + 1)
        issued_positive = int(cumulative_positive[boundary])
        correct_count = int(cumulative_correct[boundary])
        point = FrontierPoint(
            threshold=float(ordered_score[boundary]),
            issued=issued,
            issued_positive=issued_positive,
            correct=correct_count,
            precision=_ratio(correct_count, issued),
            coverage=_ratio(issued_positive, total_positive),
            stage_a_precision=_ratio(issued_positive, issued),
            conditional_precision=_ratio(correct_count, issued_positive),
        )
        points.append(point)
        if point.coverage + EPSILON < minimum_coverage:
            continue
        if best is None or (
            point.precision,
            point.coverage,
            point.conditional_precision,
            -point.issued,
        ) > (
            best.precision,
            best.coverage,
            best.conditional_precision,
            -best.issued,
        ):
            best = point
    return points, best


def _stage_points(
    scores: np.ndarray,
    stage_mask: np.ndarray,
    eligible: np.ndarray,
    positive: np.ndarray,
    correct: np.ndarray,
    *,
    quantile_count: int,
    minimum_stage_coverage: float,
) -> list[dict[str, float | int]]:
    mask = np.asarray(stage_mask, dtype=bool)
    allowed = mask & np.asarray(eligible, dtype=bool)
    stage_positive = int((mask & positive).sum())
    values = np.asarray(scores, dtype=np.float64)[allowed]
    if not len(values) or not stage_positive:
        return []
    thresholds = np.unique(
        np.concatenate(
            [
                [float(values.max()) + EPSILON],
                np.quantile(values, np.linspace(0.0, 1.0, quantile_count)),
                [float(values.min()) - EPSILON],
            ]
        )
    )
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        issued_mask = allowed & (scores >= threshold)
        issued = int(issued_mask.sum())
        issued_positive = int((issued_mask & positive).sum())
        if _ratio(issued_positive, stage_positive) + EPSILON < minimum_stage_coverage:
            continue
        rows.append(
            {
                "threshold": float(threshold),
                "issued": issued,
                "issued_positive": issued_positive,
                "correct": int((issued_mask & correct).sum()),
                "stage_positive": stage_positive,
            }
        )
    return rows


def _registered_grid_oracle(
    frame: pd.DataFrame,
    *,
    positive: np.ndarray,
    ranking_correct: np.ndarray,
    perfect_ranking: bool,
    protocol: dict,
    quantile_count: int = 61,
) -> dict[str, object]:
    direct = frame["direct_gate_probability"].to_numpy(dtype=np.float64)
    action_any = frame["action_any_probability"].to_numpy(dtype=np.float64)
    top_probability = frame["top_action_probability"].to_numpy(dtype=np.float64)
    margin = frame["top_action_margin"].to_numpy(dtype=np.float64)
    stages = frame["stage"].astype(str).to_numpy(dtype=object)
    correct = positive if perfect_ranking else ranking_correct
    total_positive = int(positive.sum())
    minimum_global = float(
        protocol["selection"]["required_global_positive_group_coverage"]
    )
    stage_floors = protocol["selection"]["required_stage_coverage_for_calibration"]
    best: dict[str, object] | None = None
    evaluated = 0
    feasible = 0

    for blend in protocol["selection"]["direct_action_blend_weights"]:
        alpha = float(blend)
        score = np.exp(
            alpha * np.log(np.clip(direct, EPSILON, 1.0))
            + (1.0 - alpha) * np.log(np.clip(action_any, EPSILON, 1.0))
        )
        for action_probability in protocol["selection"][
            "action_probability_threshold_grid"
        ]:
            for margin_threshold in protocol["selection"][
                "action_margin_threshold_grid"
            ]:
                eligible = (top_probability >= float(action_probability)) & (
                    margin >= float(margin_threshold)
                )
                stage_rows: list[list[dict[str, float | int]]] = []
                valid = True
                for stage in STAGES:
                    rows = _stage_points(
                        score,
                        stages == stage,
                        eligible,
                        positive,
                        correct,
                        quantile_count=quantile_count,
                        minimum_stage_coverage=float(stage_floors[stage]),
                    )
                    if not rows:
                        valid = False
                        break
                    stage_rows.append(rows)
                if not valid:
                    continue
                first, second, third = stage_rows
                first_issued = np.asarray([row["issued"] for row in first], dtype=np.int64)
                first_positive = np.asarray(
                    [row["issued_positive"] for row in first], dtype=np.int64
                )
                first_correct = np.asarray([row["correct"] for row in first], dtype=np.int64)
                second_issued = np.asarray([row["issued"] for row in second], dtype=np.int64)
                second_positive = np.asarray(
                    [row["issued_positive"] for row in second], dtype=np.int64
                )
                second_correct = np.asarray([row["correct"] for row in second], dtype=np.int64)
                pair_issued = (first_issued[:, None] + second_issued[None, :]).reshape(-1)
                pair_positive = (
                    first_positive[:, None] + second_positive[None, :]
                ).reshape(-1)
                pair_correct = (first_correct[:, None] + second_correct[None, :]).reshape(-1)
                for third_index, third_row in enumerate(third):
                    issued = pair_issued + int(third_row["issued"])
                    issued_positive = pair_positive + int(third_row["issued_positive"])
                    correct_count = pair_correct + int(third_row["correct"])
                    coverage = issued_positive / max(total_positive, 1)
                    allowed = (coverage + EPSILON >= minimum_global) & (issued > 0)
                    evaluated += int(len(issued))
                    if not allowed.any():
                        continue
                    feasible += int(allowed.sum())
                    precision = np.divide(
                        correct_count,
                        issued,
                        out=np.zeros_like(correct_count, dtype=np.float64),
                        where=issued > 0,
                    )
                    candidate_indexes = np.where(allowed)[0]
                    local = candidate_indexes[
                        np.lexsort(
                            (
                                -coverage[candidate_indexes],
                                -precision[candidate_indexes],
                            )
                        )[0]
                    ]
                    first_index = int(local // len(second))
                    second_index = int(local % len(second))
                    candidate = {
                        "precision": float(precision[local]),
                        "coverage": float(coverage[local]),
                        "stage_a_precision": _ratio(
                            int(issued_positive[local]), int(issued[local])
                        ),
                        "conditional_precision": _ratio(
                            int(correct_count[local]), int(issued_positive[local])
                        ),
                        "issued": int(issued[local]),
                        "issued_positive": int(issued_positive[local]),
                        "correct": int(correct_count[local]),
                        "direct_action_blend": alpha,
                        "minimum_action_probability": float(action_probability),
                        "minimum_action_margin": float(margin_threshold),
                        "stage_thresholds": [
                            float(first[first_index]["threshold"]),
                            float(second[second_index]["threshold"]),
                            float(third[third_index]["threshold"]),
                        ],
                    }
                    if best is None or (
                        candidate["precision"],
                        candidate["coverage"],
                        candidate["conditional_precision"],
                    ) > (
                        best["precision"],
                        best["coverage"],
                        best["conditional_precision"],
                    ):
                        best = candidate
    return {
        "status": "COMPLETE" if best is not None else "NO_FEASIBLE_CONFIGURATION",
        "post_hoc_test_label_oracle": True,
        "release_eligible": False,
        "perfect_ranking": perfect_ranking,
        "evaluated_combinations": evaluated,
        "feasible_combinations": feasible,
        "best": best,
    }


def _learner_instability(frame: pd.DataFrame) -> dict[str, float | int]:
    learner = frame.groupby("base_record_id", sort=True)["group_has_positive"].agg(
        ["min", "max", "count", "mean"]
    )
    mixed = learner["min"] != learner["max"]
    return {
        "learners": int(len(learner)),
        "learners_with_multiple_groups": int((learner["count"] > 1).sum()),
        "learners_with_mixed_group_targets": int(mixed.sum()),
        "mixed_target_rate": float(mixed.mean()) if len(learner) else 0.0,
    }


def _ranking_correct(frame: pd.DataFrame) -> np.ndarray:
    top = frame["top_action_index"].to_numpy(dtype=np.int64)
    targets = frame[
        [f"action_target_{index}" for index in range(ACTION_COUNT)]
    ].to_numpy(dtype=np.int8)
    return targets[np.arange(len(frame)), top] > 0


def _write_report(payload: dict[str, object]) -> None:
    current = payload["current_v4"]
    score_rows = payload["global_score_frontiers"]
    oracle = payload["registered_grid_oracle"]
    perfect = payload["perfect_ranking_gate_oracle"]
    lines = [
        "# Two-Stage V4 feasibility audit",
        "",
        "## Current held-out result",
        "",
        f"- End-to-end Precision@1: {current['end_to_end_precision_at_1']:.4f}",
        f"- Positive-group coverage: {current['positive_group_coverage']:.4f}",
        f"- Stage A precision: {current['stage_a_precision']:.4f}",
        f"- Conditional action Precision@1: {current['conditional_precision_at_1']:.4f}",
        f"- Stage A precision required for 80% end-to-end at current conditional precision: {payload['required_stage_a_precision']:.4f}",
        "",
        "## Exact global score frontiers",
        "",
    ]
    for name, row in score_rows.items():
        best = row.get("best_at_coverage_floor")
        if best is None:
            lines.append(f"- {name}: no point reaches the coverage floor")
        else:
            lines.append(
                f"- {name}: best P@1={best['precision']:.4f}, coverage={best['coverage']:.4f}, Stage A precision={best['stage_a_precision']:.4f}"
            )
    lines.extend(
        [
            "",
            "## Optimistic registered-grid oracle",
            "",
            "This section uses held-out labels to choose thresholds and is diagnostic only. It cannot authorize release.",
            "",
            f"- Current-ranking oracle: {json.dumps(oracle.get('best'), ensure_ascii=False)}",
            f"- Perfect-ranking gate oracle: {json.dumps(perfect.get('best'), ensure_ascii=False)}",
            "",
            "## Target stability",
            "",
            f"- Learners with mixed positive/non-positive stage targets: {payload['learner_target_instability']['learners_with_mixed_group_targets']}",
            f"- Mixed-target rate: {payload['learner_target_instability']['mixed_target_rate']:.4f}",
            "",
            "## Interpretation",
            "",
            "If the post-hoc registered-grid oracle remains below 0.80 at coverage 0.50, no further threshold or calibration tuning on the current V4 scores can satisfy the original gate. A new representation/target boundary would be required. Conditional action-ranking evidence remains separate from end-to-end recommendability evidence.",
            "",
            "Claim boundary: `OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT`",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    frame = pd.read_parquet(OOF_PATH).sort_values("group_id", kind="stable")
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    required = {
        "group_id",
        "base_record_id",
        "stage",
        "group_has_positive",
        "direct_gate_probability",
        "action_any_probability",
        "joint_gate_probability",
        "top_action_index",
        "top_action_probability",
        "top_action_margin",
        *[f"action_target_{index}" for index in range(ACTION_COUNT)],
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"V4 OOF predictions missing columns: {sorted(missing)}")
    if len(frame) != 29043 or frame["group_id"].nunique() != 29043:
        raise RuntimeError("V4 group authority changed")

    positive = frame["group_has_positive"].to_numpy(dtype=bool)
    ranking_correct = _ranking_correct(frame)
    minimum_coverage = float(
        protocol["selection"]["required_global_positive_group_coverage"]
    )
    score_variants = {
        "direct_gate_probability": frame["direct_gate_probability"].to_numpy(
            dtype=np.float64
        ),
        "action_any_probability": frame["action_any_probability"].to_numpy(
            dtype=np.float64
        ),
        "joint_gate_probability": frame["joint_gate_probability"].to_numpy(
            dtype=np.float64
        ),
        "joint_x_top_action_probability": (
            frame["joint_gate_probability"].to_numpy(dtype=np.float64)
            * frame["top_action_probability"].to_numpy(dtype=np.float64)
        ),
        "direct_x_top_action_probability": (
            frame["direct_gate_probability"].to_numpy(dtype=np.float64)
            * frame["top_action_probability"].to_numpy(dtype=np.float64)
        ),
    }
    frontiers: dict[str, object] = {}
    per_stage: dict[str, object] = {}
    for name, scores in score_variants.items():
        points, best = _frontier(
            scores,
            positive,
            ranking_correct,
            minimum_coverage=minimum_coverage,
        )
        frontiers[name] = {
            "point_count": len(points),
            "best_at_coverage_floor": best.to_dict() if best else None,
        }
        stage_payload: dict[str, object] = {}
        for stage in STAGES:
            mask = frame["stage"].astype(str).to_numpy(dtype=object) == stage
            stage_points, stage_best = _frontier(
                scores[mask],
                positive[mask],
                ranking_correct[mask],
                minimum_coverage=0.50,
            )
            stage_payload[stage] = {
                "point_count": len(stage_points),
                "best_at_stage_coverage_0_50": (
                    stage_best.to_dict() if stage_best else None
                ),
            }
        per_stage[name] = stage_payload

    current = release["overall"]
    conditional = float(current["stage_b_conditional_precision_at_1"])
    payload: dict[str, object] = {
        "schema_version": "two_stage_v4_feasibility_audit_v1",
        "status": "COMPLETE",
        "diagnostic_only": True,
        "release_eligible": False,
        "labels_changed": False,
        "models_trained": False,
        "groups": int(len(frame)),
        "learners": int(frame["base_record_id"].nunique()),
        "positive_groups": int(positive.sum()),
        "current_v4": {
            "end_to_end_precision_at_1": float(
                current["end_to_end_precision_at_1"]
            ),
            "positive_group_coverage": float(current["positive_group_coverage"]),
            "stage_a_precision": float(current["stage_a_precision"]),
            "conditional_precision_at_1": conditional,
        },
        "required_stage_a_precision": min(1.0, 0.80 / max(conditional, EPSILON)),
        "global_score_frontiers": frontiers,
        "per_stage_score_frontiers": per_stage,
        "registered_grid_oracle": _registered_grid_oracle(
            frame,
            positive=positive,
            ranking_correct=ranking_correct,
            perfect_ranking=False,
            protocol=protocol,
        ),
        "perfect_ranking_gate_oracle": _registered_grid_oracle(
            frame,
            positive=positive,
            ranking_correct=ranking_correct,
            perfect_ranking=True,
            protocol=protocol,
        ),
        "learner_target_instability": _learner_instability(frame),
        "claim_boundary": protocol["claim_boundary"],
    }
    best = payload["registered_grid_oracle"].get("best")
    payload["current_score_family_can_reach_original_gate"] = bool(
        best is not None
        and float(best["precision"]) >= 0.80
        and float(best["coverage"]) >= minimum_coverage
    )
    AUDIT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
