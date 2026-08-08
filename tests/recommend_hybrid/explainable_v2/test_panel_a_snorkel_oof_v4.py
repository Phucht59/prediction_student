from pathlib import Path

import numpy as np

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


def test_minimum_two_source_families_controls_training_retention():
    retained, status = runner._retention_metadata(np.asarray([1, 2, 3]))

    assert runner.MINIMUM_INDEPENDENT_SOURCE_FAMILIES == 2
    assert retained.tolist() == [False, True, True]
    assert status.tolist() == [
        "INSUFFICIENT_SOURCE_SUPPORT",
        "OOF_PANEL_A_SILVER_LABEL",
        "OOF_PANEL_A_SILVER_LABEL",
    ]


def test_unsupported_row_keeps_probabilities_but_is_not_trainable():
    retained, status = runner._retention_metadata(np.asarray([1]))

    probabilities = np.asarray([[0.01, 0.02, 0.10, 0.87]])
    assert np.isfinite(probabilities).all()
    assert probabilities.sum(axis=1).tolist() == [1.0]
    assert retained.tolist() == [False]
    assert status.tolist() == ["INSUFFICIENT_SOURCE_SUPPORT"]


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


def test_case_lineage_hash_is_order_invariant():
    import pandas as pd

    mapping = pd.DataFrame(
        {
            "query_id": ["q2", "q1"],
            "blinded_case_id": ["case_b", "case_a"],
        }
    )
    assert runner._case_lineage_sha256(mapping) == runner._case_lineage_sha256(
        mapping.iloc[::-1]
    )


def test_frozen_support_violation_is_preserved_but_not_trainable():
    import pandas as pd

    labels = pd.read_parquet(
        runner.OUTPUT_DIR / "probabilistic_relevance_labels.parquet"
    )
    unsupported = labels.loc[~labels["retained_for_training"].astype(bool)]

    assert len(labels) == 1500
    assert int(labels["retained_for_training"].sum()) == 1499
    assert unsupported[
        ["query_id", "case_id", "action_id", "label_status"]
    ].to_dict("records") == [
        {
            "query_id": "414696::GGG::2014B::EARLY_20",
            "case_id": "case_3bed45903f5da99df23a2022",
            "action_id": "RECOVER_ENGAGEMENT",
            "label_status": "INSUFFICIENT_SOURCE_SUPPORT",
        }
    ]
    probability_columns = [
        f"probability_relevance_{class_id}" for class_id in range(4)
    ]
    assert np.isfinite(unsupported[probability_columns].to_numpy()).all()
