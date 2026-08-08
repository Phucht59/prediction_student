from scripts.recommend_hybrid.explainable_v2 import (
    dispatch_gemini_panel_a_batch01_v3 as core,
)
from scripts.recommend_hybrid.explainable_v2.dispatch_gemini_panel_a_batch_v4 import (
    configure_batch,
)


def test_configure_panel_a_batch02():
    configure_batch(2)
    assert core.BATCH_ID == "panel_a_batch_02"
    assert core.SOURCE_BATCH_PATH.name == "batch_02.jsonl"
    assert core.BATCH_DIR.name == "panel_a_batch_02"
    assert core.IMPORT_RAW_PATH.name == "panel_a_batch_02_gemini.jsonl"


def test_configure_panel_a_batch06():
    configure_batch(6)
    assert core.BATCH_ID == "panel_a_batch_06"
    assert core.SOURCE_BATCH_PATH.name == "batch_06.jsonl"
    assert core.BATCH_DIR.name == "panel_a_batch_06"
    assert core.IMPORT_RAW_PATH.name == "panel_a_batch_06_gemini.jsonl"


def test_configure_rejects_out_of_range():
    try:
        configure_batch(7)
    except ValueError as exc:
        assert "between 1 and 6" in str(exc)
    else:
        raise AssertionError("batch 7 should be rejected")
