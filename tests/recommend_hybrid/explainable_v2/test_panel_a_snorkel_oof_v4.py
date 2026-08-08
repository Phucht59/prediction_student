from pathlib import Path

from scripts.recommend_hybrid.explainable_v2 import fit_weak_label_models as runner


def test_panel_a_snorkel_protocol_is_locked():
    assert runner.CARDINALITY == 4
    assert runner.EXPECTED_PANEL_A_CASES == 300
    assert runner.EXPECTED_FROZEN_RECORDS == 1117
    assert runner.EXPECTED_ACTION_ROWS == 1500
    assert runner.EXPECTED_FROZEN_SHA256 == (
        "4a4871426880bdcd1257dc15c29a36c23de34481f07be68d8e5095dc20efefb9"
    )


def test_real_external_review_is_single_source():
    llm = [s for s in runner.SOURCES if s.family == "LLM_EXPERT"]
    assert len(llm) == 1
    assert llm[0].name == "REAL_EXTERNAL_GEMINI_REVIEW_V4"


def test_no_panel_b_artifact_path_in_runner_source():
    text = Path(runner.__file__).read_text(encoding="utf-8")
    assert "panel_b_cases" not in text
    assert "panel_b_request_batches" not in text
    assert "full_model_ndcg" not in text
    assert "leave_one_family_out" not in text


def test_blinded_case_id_is_deterministic():
    a = runner._blinded_case_id("query-1", "secret")
    b = runner._blinded_case_id("query-1", "secret")
    c = runner._blinded_case_id("query-2", "secret")
    assert a == b
    assert a != c
    assert a.startswith("case_")
    assert len(a) == 29
