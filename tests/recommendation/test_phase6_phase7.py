from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.recommendation.build_progress_monitoring_gemma_jobs import build_jobs
from scripts.recommendation.evaluate_lf_diagnostics import build as build_diagnostics
from src.recommendation.labeling.behavioral import FINAL_ACTIONS, behavioral_label, derive_thresholds
from src.recommendation.labeling.progress_monitoring import parse_progress_function_call, progress_function_declaration
from src.recommendation.weak_supervision.matrix import FINAL_ACTIONS as MATRIX_ACTIONS, SOURCES_BY_ACTION, A4GemmaGateError, build_matrices, load_sources


ROOT = Path(__file__).resolve().parents[2]
PANEL_A = ROOT / "artifacts/recommendation/panels/panel_a.parquet"
PANEL_B = ROOT / "artifacts/recommendation/panels/panel_b.parquet"


def _synthetic_matrices():
    case_ids = [f"c{i:03d}" for i in range(500)]
    matrices = {}
    for action_index, action_id in enumerate(MATRIX_ACTIONS):
        matrix = pd.DataFrame({"case_id": case_ids})
        for source_index, source in enumerate(SOURCES_BY_ACTION[action_id]):
            matrix[source] = [-1 if source == "LF_BEHAVIOR" and i % 5 == 0 else (i + action_index + source_index) % 4 for i in range(500)]
        matrices[action_id] = matrix
    return matrices


def test_final_action_contract_and_retirement():
    config = yaml.safe_load((ROOT / "configs/recommendation/actions.yaml").read_text(encoding="utf-8"))
    assert [item["id"] for item in config["active_actions"]] == ["assessment_recovery", "re_engagement", "study_planning", "progress_monitoring", "retrieval_practice"]
    assert next(item for item in config["active_actions"] if item["id"] == "progress_monitoring")["status"] == "ACTIVE"
    assert next(item for item in config["active_actions"] if item["id"] == "retrieval_practice")["status"] == "ACTIVE_REVIEW"
    assert next(item for item in config["retired_actions"] if item["id"] == "content_review")["status"] == "RETIRED"
    assert next(item for item in config["rejected_candidates"] if item["id"] == "academic_help_seeking")["status"] == "REJECTED_CANDIDATE"


def test_progress_gemma_jobs_are_single_case_and_final_prompt():
    jobs = [__import__("json").loads(line) for line in (ROOT / "artifacts/recommendation/labeling/jobs/progress_monitoring_gemma_single_jobs.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(jobs) == 500
    assert all(len(job["case_ids"]) == 1 and len(job["payload"]) == 1 for job in jobs)
    assert all(job["model"] == "gemma-4-31b-it" for job in jobs)
    assert all(job["prompt_version"] == "recommendation_progress_monitoring_gemma_v1" for job in jobs)
    assert all("C01" in job["prompt"] for job in jobs)
    assert all("B2" not in job["prompt"] and "academic help" not in job["prompt"].lower() for job in jobs)
    assert all(set(job["payload"][0]) == {"case_id", "stage", "risk_probability", "risk_band", "recent_activity", "activity_trend", "active_days_ratio", "assessment_completion", "missing_assessments", "course_progress"} for job in jobs)


def test_progress_gemma_function_call_is_single_a4_label():
    response = {"candidates": [{"content": {"parts": [{"functionCall": {
        "name": "submit_progress_monitoring_label",
        "args": {"cases": [{"case_ref": "C01", "label": "2"}]},
    }}]}}]}
    parsed = parse_progress_function_call(__import__("json").dumps(response), ["real-case"])
    assert parsed["real-case"]["labels"] == {"A4": {"label": 2}}
    schema = progress_function_declaration()
    item = schema["parameters"]["properties"]["cases"]["items"]
    assert set(item["required"]).issubset(item["properties"])


def test_behavioral_output_shape_and_panel_b_exclusion():
    frame = pd.read_parquet(ROOT / "artifacts/recommendation/labeling/normalized/behavioral_labels.parquet")
    assert len(frame) == 2500
    assert frame["case_id"].nunique() == 500
    assert set(frame["action_id"]) == set(FINAL_ACTIONS)
    assert not set(frame["case_id"]) & set(pd.read_parquet(PANEL_B)["case_id"].astype(str))
    assert frame["lf_name"].str.startswith("LF_BEHAVIOR_").all()
    assert frame["reason_code"].notna().all()


def test_behavior_thresholds_are_reproducible_and_a4_gap_is_not_invented():
    panel = pd.read_parquet(PANEL_A)
    assert derive_thresholds(panel) == derive_thresholds(panel)
    thresholds = derive_thresholds(panel)
    state = panel.iloc[0].to_dict()
    assert behavioral_label(state, "progress_monitoring", "UNKNOWN", thresholds)["reason_code"] == "UNSUPPORTED_BEHAVIOR_SIGNAL"


def test_behavior_action_specific_rules_and_a5_unknown():
    panel = pd.read_parquet(PANEL_A)
    thresholds = derive_thresholds(panel)
    state = panel.iloc[0].to_dict()
    assert behavioral_label(state, "assessment_recovery", "INFEASIBLE", thresholds)["label"] == "ABSTAIN"
    assert behavioral_label(state, "retrieval_practice", "UNKNOWN", thresholds)["label"] == "ABSTAIN"
    quiz_state = state.copy()
    quiz_state["quiz_activity"] = 25.0
    assert behavioral_label(quiz_state, "retrieval_practice", "UNKNOWN", thresholds)["label"] != "ABSTAIN"
    state.update({"inactive_streak": 20, "active_days_ratio": 0.0, "recent_activity": 0.0, "activity_trend": -1.0, "vle_available": True})
    a2 = behavioral_label(state, "re_engagement", "FEASIBLE", thresholds)
    a3 = behavioral_label(state, "study_planning", "FEASIBLE", thresholds)
    assert a2["label"] != "ABSTAIN"
    assert a3["label"] == "ABSTAIN"


def test_matrix_gate_stops_without_gemini31_a4(tmp_path):
    with pytest.raises(A4GemmaGateError, match="WAITING_FOR_A4_GEMINI31_LABELS"):
        load_sources(
            ROOT / "artifacts/recommendation/labeling/normalized/behavioral_labels.parquet",
            ROOT / "artifacts/recommendation/labeling/normalized/gemini_supported_labels.parquet",
            ROOT / "artifacts/recommendation/labeling/normalized/a4_replacement_gemini_labels.parquet",
            tmp_path / "missing_gemini31.parquet",
            ROOT / "artifacts/recommendation/labeling/normalized/gemma_supported_labels.parquet",
            PANEL_A,
            PANEL_B,
        )


def test_variable_lf_matrix_contract_without_execution(tmp_path):
    matrices = _synthetic_matrices()
    out = tmp_path / "matrices"
    panel_path = tmp_path / "panel_a.parquet"
    pd.DataFrame({"case_id": [f"c{i:03d}" for i in range(500)]}).to_parquet(panel_path, index=False)
    def source_frame(source):
        return pd.concat([
            frame[["case_id", source]].rename(columns={source: "label"}).assign(action_id=action_id, lf_name=source)
            for action_id, frame in matrices.items() if source in frame
        ])
    source_names = sorted({source for sources in SOURCES_BY_ACTION.values() for source in sources})
    built = build_matrices({
        source: source_frame(source) for source in source_names
    }, panel_path, out)
    assert set(built) == set(MATRIX_ACTIONS)
    assert all(frame.shape == (500, 4) for action, frame in built.items() if action != "progress_monitoring")
    assert built["progress_monitoring"].shape == (500, 3)
    assert -1 in built["retrieval_practice"]["LF_BEHAVIOR"].values
    assert list(built["progress_monitoring"].columns) == ["case_id", "LF_GEMINI35", "LF_GEMINI31"]
