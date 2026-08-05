"""Memory-safe entry point for the authoritative OULAD landmark builder."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.recommend_hybrid.causal import build_oulad_landmark_rows as base
from src.recommend_hybrid.causal.oulad_activity import (
    collect_weekly_activity_sqlite,
)
from src.recommend_hybrid.causal.study_regularity import (
    study_regularity_components,
    study_regularity_score,
)

_ORIGINAL_VLE_MEASURES = base._vle_measures
_ORIGINAL_EXPAND_ACTIONS = base._expand_actions


def _vle_measures_with_gap(
    window: pd.DataFrame,
    weekly: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in window.itertuples(index=False):
        baseline_days = int(item.cutoff_day - item.baseline_start_day)
        followup_days = int(item.followup_end_day - item.cutoff_day)
        baseline_weeks = max(2, int(np.ceil(baseline_days / 7.0)))
        followup_weeks = max(2, int(np.ceil(followup_days / 7.0)))
        payload: dict[str, Any] = {"record_id": item.record_id}
        for period, days, weeks in (
            ("baseline", baseline_days, baseline_weeks),
            ("followup", followup_days, followup_weeks),
        ):
            total = base._weekly_vector(
                weekly, item.record_id, period, weeks, "total_clicks"
            )
            active = base._weekly_vector(
                weekly, item.record_id, period, weeks, "active_days"
            )
            quiz = base._weekly_vector(
                weekly, item.record_id, period, weeks, "quiz_clicks"
            )
            content = base._weekly_vector(
                weekly, item.record_id, period, weeks, "content_clicks"
            )
            components = study_regularity_components(total[None, :])
            payload[f"{period}__study_regularity_score"] = float(
                study_regularity_score(total[None, :])[0]
            )
            payload[f"{period}__maximum_inactive_gap"] = int(
                components["maximum_inactive_gap"][0]
            )
            payload[f"{period}__vle_active_day_rate"] = float(
                np.clip(active.sum() / max(1, days), 0.0, 1.0)
            )
            payload[f"{period}__retrieval_practice_rate"] = float(
                np.mean(quiz[:weeks] > 0.0)
            )
            payload[f"{period}__content_review_coverage"] = float(
                np.mean(content[:weeks] > 0.0)
            )
        rows.append(payload)
    return pd.DataFrame(rows)


def _expand_actions_with_gap(base_frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    expanded = _ORIGINAL_EXPAND_ACTIONS(base_frame, stage)
    selected = expanded["action_id"].eq("STUDY_REGULARITY")
    invalid_gap = expanded["followup__maximum_inactive_gap"].astype(int) > 2
    expanded.loc[selected & invalid_gap, "followup_measure"] = 0.0
    return expanded


def build(
    output_path: Path = base.OUTPUT,
    manifest_path: Path = base.MANIFEST,
    *,
    chunksize: int = 750_000,
    batch_size: int = 512,
    force_bundle: bool = False,
):
    original_activity = base._collect_weekly_activity
    original_vle = base._vle_measures
    original_expand = base._expand_actions
    try:
        base._collect_weekly_activity = collect_weekly_activity_sqlite
        base._vle_measures = _vle_measures_with_gap
        base._expand_actions = _expand_actions_with_gap
        return base.build(
            output_path,
            manifest_path,
            chunksize=chunksize,
            batch_size=batch_size,
            force_bundle=force_bundle,
        )
    finally:
        base._collect_weekly_activity = original_activity
        base._vle_measures = original_vle
        base._expand_actions = original_expand


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=base.OUTPUT)
    parser.add_argument("--manifest", type=Path, default=base.MANIFEST)
    parser.add_argument("--chunksize", type=int, default=750_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--force-bundle", action="store_true")
    args = parser.parse_args()
    frame = build(
        args.output,
        args.manifest,
        chunksize=args.chunksize,
        batch_size=args.batch_size,
        force_bundle=args.force_bundle,
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "rows": len(frame),
                "output": str(args.output),
                "activity_aggregation": "SQLITE_DISK_BACKED",
                "study_regularity_maximum_inactive_gap_weeks": 2,
            }
        )
    )


if __name__ == "__main__":
    main()
