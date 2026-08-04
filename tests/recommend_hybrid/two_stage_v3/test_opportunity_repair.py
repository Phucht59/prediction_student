from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/recommend_hybrid/two_stage_v3/repair_opportunity_count.py"
)
SPEC = importlib.util.spec_from_file_location("two_stage_v3_opportunity_repair_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "activity_type": ["quiz", "resource", "forumng", "page"],
            "week_from": [2.0, 2.0, 5.0, 4.0],
            "week_to": [3.0, 3.0, 6.0, np.nan],
        }
    )


def test_assessment_opportunity_uses_open_closed_cutoff_window() -> None:
    count = module._action_opportunity(
        action_family="ASSESSMENT_COMPLETION",
        cutoff_day=14.0,
        target_day=35.0,
        assessment_dates=np.array([14.0, 20.0, 35.0, 40.0]),
        vle_schedule=pd.DataFrame(),
    )
    assert count == 2


def test_vle_opportunity_matches_original_overlap_semantics() -> None:
    frame = schedule()
    assert module._action_opportunity(
        action_family="VLE_ENGAGEMENT",
        cutoff_day=14.0,
        target_day=28.0,
        assessment_dates=np.empty(0),
        vle_schedule=frame,
    ) == 3
    assert module._action_opportunity(
        action_family="QUIZ_OR_RETRIEVAL_PRACTICE",
        cutoff_day=14.0,
        target_day=28.0,
        assessment_dates=np.empty(0),
        vle_schedule=frame,
    ) == 1
    assert module._action_opportunity(
        action_family="CONTENT_REVIEW",
        cutoff_day=14.0,
        target_day=28.0,
        assessment_dates=np.empty(0),
        vle_schedule=frame,
    ) == 2


def test_study_regularity_uses_available_days() -> None:
    assert module._action_opportunity(
        action_family="STUDY_REGULARITY",
        cutoff_day=28.0,
        target_day=49.0,
        assessment_dates=np.empty(0),
        vle_schedule=pd.DataFrame(),
    ) == 21
