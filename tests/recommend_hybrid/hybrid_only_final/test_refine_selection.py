from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts/recommend_hybrid/hybrid_only_final"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

refine = importlib.import_module("refine_selection")


def row(config_id: str, precision: float, coverage: float, meets: bool) -> dict:
    return {
        "config_id": config_id,
        "risk_weight": 0.8,
        "evidence_weight": 0.2,
        "need_weight": 0.1,
        "certainty_weight": 0.1,
        "workload_weight": 0.05,
        "minimum_risk_reduction": 0.01,
        "maximum_uncertainty": 0.2,
        "minimum_evidence": 0.4,
        "minimum_top_margin": 0.02,
        "minimum_top_score": 0.15,
        "mean_precision_at_1": precision,
        "mean_actionable_coverage": coverage,
        "mean_selective_accuracy": 0.5,
        "mean_macro_action_precision": precision,
        "worst_stage_precision": precision - 0.05,
        "minimum_action_diversity": 3,
        "maximum_top_action_concentration": 0.5,
        "meets_target": meets,
    }


def test_target_met_prefers_higher_coverage() -> None:
    selected = refine._select(
        pd.DataFrame(
            [
                row("high_precision", 0.90, 0.55, True),
                row("high_coverage", 0.82, 0.70, True),
            ]
        )
    )
    assert selected["config_id"] == "high_coverage"
    assert selected["selection_reason"] == "TARGET_MET_MAXIMIZE_COVERAGE"


def test_target_not_met_prefers_precision_with_coverage_floor() -> None:
    selected = refine._select(
        pd.DataFrame(
            [
                row("coverage_only", 0.60, 0.90, False),
                row("near_target", 0.79, 0.50, False),
                row("no_coverage", 0.95, 0.20, False),
            ]
        )
    )
    assert selected["config_id"] == "near_target"
    assert (
        selected["selection_reason"]
        == "TARGET_NOT_MET_MAXIMIZE_PRECISION_WITH_COVERAGE"
    )
