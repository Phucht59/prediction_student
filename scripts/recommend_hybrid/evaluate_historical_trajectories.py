"""Check observational behavior trajectories after generated recommendations.

This script runs only after counterfactual recommendations have been generated.
It may inspect later behavior and final outcomes for evaluation, but its outputs
are never consumed by the action generator, utility ranker, or constraint solver.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.oulad import BASE_CHANNELS, _build_bundle
from src.recommend_hybrid.counterfactual.historical_validation import (
    HISTORICAL_CLAIM_BOUNDARY,
    HistoricalTrajectoryRow,
    aggregate_historical_metrics,
)

OUT = ROOT / "artifacts/recommend_hybrid/counterfactual"
REPORT = ROOT / "reports/recommend_hybrid/HISTORICAL_TRAJECTORY_VALIDATION.md"
STAGE_PATH = {
    "EARLY_20": ("E1_EARLY_20PCT", "E2_EARLY_35PCT"),
    "EARLY_35": ("E2_EARLY_35PCT", "M1_MIDDLE_50PCT"),
    "MIDDLE_50": ("M1_MIDDLE_50PCT", "L1_LATE_75PCT"),
    "LATE_75": ("L1_LATE_75PCT", None),
}
NEXT_OOF_STAGE = {
    "EARLY_20": "E2_EARLY_35PCT",
    "EARLY_35": "M1_MIDDLE_50PCT",
    "MIDDLE_50": "L1_LATE_75PCT",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _next_risk_map() -> dict[tuple[str, str], float]:
    path = ROOT / "artifacts/canonical_v3/predictions/oulad_oof_predictions.parquet"
    frame = pd.read_parquet(
        path,
        columns=["base_record_id", "stage", "model", "probability"],
    )
    frame = frame.loc[
        frame["model"].eq("hybrid")
        & frame["stage"].isin(set(NEXT_OOF_STAGE.values()))
    ]
    return {
        (str(row.base_record_id), str(row.stage)): float(row.probability)
        for row in frame.itertuples(index=False)
    }


def _row_index(data: Any) -> dict[str, int]:
    return {
        str(record_id): int(index)
        for index, record_id in enumerate(data.frame["base_record_id"])
    }


def _weekly_rate(
    sequence: np.ndarray,
    *,
    channel: int,
    start: int,
    stop: int,
) -> float:
    weeks = stop - start
    if weeks <= 0:
        return 0.0
    return float(sequence[start:stop, channel].sum() / weeks)


def _behavior_alignment(
    action_id: str,
    current_sequence: np.ndarray,
    current_length: int,
    next_sequence: np.ndarray,
    next_length: int,
) -> bool | None:
    if next_length <= current_length:
        return None
    index = {name: position for position, name in enumerate(BASE_CHANNELS)}
    interval_start = current_length
    interval_stop = next_length

    if action_id == "VLE_ENGAGEMENT":
        current = _weekly_rate(
            current_sequence,
            channel=index["total_clicks"],
            start=0,
            stop=current_length,
        )
        future = _weekly_rate(
            next_sequence,
            channel=index["total_clicks"],
            start=interval_start,
            stop=interval_stop,
        )
        active_days = next_sequence[
            interval_start:interval_stop,
            index["active_days"],
        ].sum()
        return bool(future > current and active_days > 0)

    if action_id == "STUDY_SCHEDULE":
        current = _weekly_rate(
            current_sequence,
            channel=index["active_days"],
            start=0,
            stop=current_length,
        )
        future = _weekly_rate(
            next_sequence,
            channel=index["active_days"],
            start=interval_start,
            stop=interval_stop,
        )
        current_inactivity = float(
            current_sequence[
                current_length - 1,
                index["days_since_last_vle_activity"],
            ]
        )
        next_inactivity = float(
            next_sequence[
                next_length - 1,
                index["days_since_last_vle_activity"],
            ]
        )
        return bool(future >= current and next_inactivity < current_inactivity)

    if action_id == "ASSESSMENT_COMPLETION":
        submissions = next_sequence[
            interval_start:interval_stop,
            index["submitted_assessment_count"],
        ].sum()
        return bool(submissions > 0)

    if action_id in {"RETRIEVAL_PRACTICE", "TARGETED_PRACTICE"}:
        current = _weekly_rate(
            current_sequence,
            channel=index["quiz_clicks"],
            start=0,
            stop=current_length,
        )
        future = _weekly_rate(
            next_sequence,
            channel=index["quiz_clicks"],
            start=interval_start,
            stop=interval_stop,
        )
        return bool(future > current)

    if action_id == "LEARNING_CONSOLIDATION":
        current = _weekly_rate(
            current_sequence,
            channel=index["content_clicks"],
            start=0,
            stop=current_length,
        )
        future = _weekly_rate(
            next_sequence,
            channel=index["content_clicks"],
            start=interval_start,
            stop=interval_stop,
        )
        return bool(future >= current and future > 0)

    return None


def evaluate() -> dict[str, Any]:
    evaluation_path = OUT / "evaluation_rows.csv"
    if not evaluation_path.is_file():
        raise FileNotFoundError(
            "run evaluate_counterfactual_recommender.py before trajectory validation"
        )
    recommendations = pd.read_csv(evaluation_path)
    recommendations = recommendations.loc[
        recommendations["top_action_id"].notna()
    ].copy()
    bundle = _build_bundle()
    stage_indices = {
        stage: _row_index(data) for stage, data in bundle.stages.items()
    }
    next_risk = _next_risk_map()
    rows: list[HistoricalTrajectoryRow] = []

    for recommendation in recommendations.itertuples(index=False):
        stage = str(recommendation.stage)
        if stage not in STAGE_PATH:
            continue
        current_stage, next_stage = STAGE_PATH[stage]
        current_data = bundle.stages[current_stage]
        current_index = stage_indices[current_stage].get(
            str(recommendation.student_key)
        )
        if current_index is None:
            continue
        current_frame = current_data.frame.iloc[current_index]
        favorable = int(current_frame["target"]) == 0
        alignment: bool | None = None
        next_probability: float | None = None
        if next_stage is not None:
            next_index = stage_indices[next_stage].get(
                str(recommendation.student_key)
            )
            if next_index is not None:
                next_data = bundle.stages[next_stage]
                alignment = _behavior_alignment(
                    str(recommendation.top_action_id),
                    current_data.sequence[current_index],
                    int(current_data.lengths[current_index]),
                    next_data.sequence[next_index],
                    int(next_data.lengths[next_index]),
                )
            next_probability = next_risk.get(
                (
                    str(recommendation.student_key),
                    str(NEXT_OOF_STAGE[stage]),
                )
            )
        rows.append(
            HistoricalTrajectoryRow(
                student_key=str(recommendation.student_key),
                course_key=str(recommendation.course_key),
                stage=stage,
                action_id=str(recommendation.top_action_id),
                behavior_aligned=alignment,
                next_stage_risk=next_probability,
                favorable_final_outcome=favorable,
            )
        )

    overall = aggregate_historical_metrics(rows)
    by_action = {
        action_id: aggregate_historical_metrics(
            row for row in rows if row.action_id == action_id
        )
        for action_id in sorted({row.action_id for row in rows})
    }
    by_stage = {
        stage: aggregate_historical_metrics(
            row for row in rows if row.stage == stage
        )
        for stage in sorted({row.stage for row in rows})
    }
    payload = {
        "schema_version": "counterfactual_historical_trajectory_v1",
        "generated_at": _utc_now(),
        "claim_boundary": HISTORICAL_CLAIM_BOUNDARY,
        "overall": overall,
        "by_action": by_action,
        "by_stage": by_stage,
        "scientific_guards": {
            "historical_outcomes_used_for_action_ranking": False,
            "historical_outcomes_used_for_action_selection": False,
            "behavior_alignment_is_proxy_not_compliance_measure": True,
            "causal_effect_claimed": False,
        },
        "status": "PASS" if rows else "FAIL",
    }
    _write_json(OUT / "historical_trajectory.json", payload)
    _write_csv(
        OUT / "historical_trajectory_rows.csv",
        [row.to_dict() for row in rows],
    )
    _write_report(payload)
    return payload


def _write_report(payload: dict[str, Any]) -> None:
    overall = payload["overall"]
    lines = [
        "# Historical trajectory validation",
        "",
        f"- Status: `{payload['status']}`",
        f"- Claim boundary: `{payload['claim_boundary']}`",
        f"- Records: `{overall['record_count']}`",
        f"- Behavior-evaluable rate: "
        f"`{overall['behavior_evaluable_rate']:.4f}`",
        f"- Behavior-alignment rate: "
        f"`{overall['behavior_alignment_rate']:.4f}`",
        f"- Observed next-stage risk difference: "
        f"`{overall['observed_next_stage_risk_difference']}`",
        f"- Observed favorable-outcome difference: "
        f"`{overall['observed_favorable_outcome_difference']}`",
        "",
        "## Interpretation boundary",
        "",
        "These values describe observational associations. Students who later "
        "changed behavior may differ from other students for many unmeasured "
        "reasons. The results are not treatment-effect or causal evidence, and "
        "they are not fed back into recommendation ranking.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    payload = evaluate()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
