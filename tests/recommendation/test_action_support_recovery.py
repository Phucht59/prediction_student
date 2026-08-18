from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.recommendation.recover_supported_labels import _validate_supported_item
from src.recommendation.labeling.parser import LabelParseError


ROOT = Path(__file__).resolve().parents[2]
LEGACY_SUPPORTED = {"A1", "A2", "A3", "A5"}


def test_action_support_contract_freezes_final_five_actions_and_history():
    config = yaml.safe_load((ROOT / "configs/recommendation/action_support.yaml").read_text(encoding="utf-8"))
    assert set(config["supported_actions"]) == {"A1", "A2", "A3", "A4", "A5"}
    assert config["active_action_ids"]["A4"] == "progress_monitoring"
    assert config["actions"]["A4"]["status"] == "SUPPORTED"
    assert config["retired_actions"]["academic_help_seeking"]["status"] == "REJECTED_CANDIDATE"
    assert config["validation_history"]["progress_monitoring_gemma"]["status"] == "REJECTED_NOT_ROBUST"
    assert config["retired_actions"]["content_review"]["status"] == "RETIRED"
    assert config["actions"]["A5"]["review_status"] == "REVIEW"


@pytest.mark.parametrize("filename", ["gemma_supported_labels.parquet", "gemini_supported_labels.parquet"])
def test_supported_normalized_table_has_500_cases_x_4_actions(filename):
    frame = pd.read_parquet(ROOT / "artifacts/recommendation/labeling/normalized" / filename)
    assert len(frame) == 2000
    assert frame["case_id"].nunique() == 500
    assert set(frame["action_id"]) == LEGACY_SUPPORTED
    assert not frame.duplicated(["case_id", "action_id", "lf_name"]).any()
    assert set(frame["label"].astype(str)).issubset({"0", "1", "2", "3", "ABSTAIN"})


def test_supported_tables_exclude_panel_b_and_a4():
    panel_b = set(pd.read_parquet(ROOT / "artifacts/recommendation/panels/panel_b.parquet")["case_id"].astype(str))
    for filename in ("gemma_supported_labels.parquet", "gemini_supported_labels.parquet"):
        frame = pd.read_parquet(ROOT / "artifacts/recommendation/labeling/normalized" / filename)
        assert not set(frame["case_id"].astype(str)) & panel_b
        assert "A4" not in set(frame["action_id"])


def test_a4_unknown_infeasible_is_not_reused_for_supported_normalization():
    with pytest.raises(LabelParseError):
        _validate_supported_item({"label": "ABSTAIN", "reason": "INFEASIBLE"}, "A4", "UNKNOWN")
    assert _validate_supported_item({"label": "ABSTAIN", "reason": "INSUFFICIENT_INFORMATION"}, "A5", "UNKNOWN") == (
        "ABSTAIN", "INSUFFICIENT_INFORMATION"
    )


def test_comparison_report_keeps_a5_review_and_a4_supportability_decision():
    comparison = (ROOT / "reports/recommendation/WEAK_LABEL_QUALITY.md").read_text(encoding="utf-8")
    supportability = (ROOT / "reports/recommendation/ACTION_SUPPORTABILITY.md").read_text(encoding="utf-8")
    assert "A5`: REVIEW" in comparison
    assert "UNSUPPORTED_BY_CURRENT_STATE" in supportability
    assert "Current Student State lacks observable content-level evidence." in supportability
    assert "2000" in supportability
