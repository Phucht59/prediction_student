from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.recommendation.weak_supervision.matrix import validate_source_manifest


ROOT = Path(__file__).resolve().parents[2]
PANEL_A = ROOT / "artifacts/recommendation/panels/panel_a.parquet"
PANEL_B = ROOT / "artifacts/recommendation/panels/panel_b.parquet"
NORMALIZED = ROOT / "artifacts/recommendation/labeling/normalized"

FINAL_ACTIONS = {
    "assessment_recovery",
    "re_engagement",
    "study_planning",
    "progress_monitoring",
    "retrieval_practice",
}
EFFECTIVE = {
    "assessment_recovery": {"LF_GEMINI35", "LF_GEMMA4", "LF_BEHAVIOR"},
    "re_engagement": {"LF_GEMINI35", "LF_GEMMA4", "LF_BEHAVIOR"},
    "study_planning": {"LF_GEMINI35", "LF_GEMMA4", "LF_BEHAVIOR"},
    "progress_monitoring": {"LF_GEMINI35", "LF_GEMINI31"},
    "retrieval_practice": {"LF_GEMINI35", "LF_GEMMA4", "LF_BEHAVIOR"},
}
LLM_EFFECTIVE = {
    "assessment_recovery": {"LF_GEMINI35", "LF_GEMMA4"},
    "re_engagement": {"LF_GEMINI35", "LF_GEMMA4"},
    "study_planning": {"LF_GEMINI35", "LF_GEMMA4"},
    "progress_monitoring": {"LF_GEMINI35", "LF_GEMINI31"},
    "retrieval_practice": {"LF_GEMINI35", "LF_GEMMA4"},
}


def _panel_ids(path: Path) -> set[str]:
    return set(pd.read_parquet(path)["case_id"].astype(str))


def test_final_action_and_source_contracts_are_exact():
    actions = yaml.safe_load((ROOT / "configs/recommendation/actions.yaml").read_text(encoding="utf-8"))
    assert {item["id"] for item in actions["active_actions"]} == FINAL_ACTIONS
    assert next(item for item in actions["active_actions"] if item["id"] == "retrieval_practice")["status"] == "ACTIVE_REVIEW"
    assert next(item for item in actions["retired_actions"] if item["id"] == "content_review")["status"] == "RETIRED"
    assert next(item for item in actions["rejected_candidates"] if item["id"] == "academic_help_seeking")["status"] == "REJECTED_CANDIDATE"

    registry = yaml.safe_load((ROOT / "configs/recommendation/label_sources.yaml").read_text(encoding="utf-8"))
    assert {action: set(item["effective_sources"]) for action, item in registry["actions"].items()} == EFFECTIVE
    phase7 = yaml.safe_load((ROOT / "configs/recommendation/phase7_input.yaml").read_text(encoding="utf-8"))
    assert {action: set(sources) for action, sources in phase7["actions"].items()} == EFFECTIVE
    assert phase7["variable_lf_count"] is True


def test_canonical_phase6_llm_table_has_5000_effective_rows_and_no_panel_b():
    frame = pd.read_parquet(NORMALIZED / "phase6_llm_labels.parquet")
    assert len(frame) == 5000
    assert set(frame["action_id"]) == FINAL_ACTIONS
    assert not frame.duplicated(["case_id", "action_id", "lf_name"]).any()
    assert not set(frame["case_id"].astype(str)) & _panel_ids(PANEL_B)
    assert set(frame["case_id"].astype(str)) == _panel_ids(PANEL_A)
    assert set(frame["label"].astype(str)).issubset({"0", "1", "2", "3", "ABSTAIN"})
    assert {action: set(group["lf_name"]) for action, group in frame.groupby("action_id")} == LLM_EFFECTIVE
    assert all(len(group) == 500 for _, group in frame.groupby(["action_id", "lf_name"]))


def test_a4_gemini31_normalized_table_is_canonical_and_complete():
    frame = pd.read_parquet(NORMALIZED / "progress_monitoring_gemini31_labels.parquet")
    assert len(frame) == 500
    assert frame["action_id"].eq("progress_monitoring").all()
    assert frame["lf_name"].eq("LF_GEMINI31").all()
    assert frame["model"].eq("gemini-3.1-flash-lite").all()
    assert not frame.duplicated(["case_id", "action_id", "lf_name"]).any()
    assert not set(frame["case_id"].astype(str)) & _panel_ids(PANEL_B)


def test_behavioral_table_has_2500_rows_and_intentional_a4_abstain():
    frame = pd.read_parquet(NORMALIZED / "behavioral_labels.parquet")
    assert len(frame) == 2500
    assert set(frame["action_id"]) == FINAL_ACTIONS
    assert not frame.duplicated(["case_id", "action_id", "lf_name"]).any()
    assert not set(frame["case_id"].astype(str)) & _panel_ids(PANEL_B)
    a4 = frame[frame["action_id"] == "progress_monitoring"]
    assert len(a4) == 500
    assert a4["label"].eq("ABSTAIN").all()
    assert a4["reason_code"].eq("UNSUPPORTED_BEHAVIOR_SIGNAL").all()


def test_source_manifest_marks_only_effective_sources_for_phase7():
    manifest = json.loads((ROOT / "artifacts/recommendation/labeling/phase6_source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["panel_b_overlap_count"] == 0
    assert manifest["effective_llm_rows"] == 5000
    assert manifest["behavioral_rows"] == 2500
    assert {action: set(sources) for action, sources in manifest["effective_sources_by_action"].items()} == EFFECTIVE
    effective = {(entry["action_id"], entry["lf_name"]) for entry in manifest["sources"] if entry["used_in_phase7"]}
    assert effective == {(action, source) for action, sources in EFFECTIVE.items() for source in sources}
    excluded = {(entry["action_id"], entry["lf_name"]): entry for entry in manifest["sources"] + manifest["excluded_audit_sources"] if not entry["used_in_phase7"]}
    assert excluded[("progress_monitoring", "LF_BEHAVIOR")]["status"] == "EXCLUDED_ZERO_COVERAGE"
    assert excluded[("progress_monitoring", "LF_GEMMA4")]["status"] == "REJECTED_DEGENERATE"
    assert excluded[("content_review", "HISTORICAL")]["status"] == "RETIRED"
    assert excluded[("progress_monitoring", "LF_ACADEMIC_HELP_SEEKING")]["status"] == "REJECTED_CANDIDATE"
    validated = validate_source_manifest(ROOT / "artifacts/recommendation/labeling/phase6_source_manifest.json", PANEL_A, PANEL_B)
    assert validated["effective_llm_rows"] == 5000


def test_phase6_reports_record_required_gate_and_quality_findings():
    diagnostics = (ROOT / "reports/recommendation/PHASE6_LF_DIAGNOSTICS.md").read_text(encoding="utf-8")
    validation = (ROOT / "reports/recommendation/PHASE6_VALIDATION.md").read_text(encoding="utf-8")
    assert "A5" in diagnostics and "REVIEW_HIGH_CONFLICT" in diagnostics
    assert "0.044" in diagnostics and "0.148" in diagnostics and "-0.098" in diagnostics
    assert "A4" in diagnostics and "0.557" in diagnostics and "0.714" in diagnostics
    assert "Panel-B overlap" in validation
    assert "API calls: `0`" in validation
    assert "Snorkel execution: `0`" in validation
    assert "EBM training: `0`" in validation
    assert "final_result" in validation
