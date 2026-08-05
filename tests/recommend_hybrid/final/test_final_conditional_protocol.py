from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_protocol_is_conditional_and_runtime_disabled():
    text = (ROOT / "configs/recommend_hybrid/final/conditional_action_protocol.yaml").read_text()
    assert "module_boundary: conditional_hybrid_action_ranker" in text
    assert "end_to_end_recommendability_in_scope: false" in text
    assert "external_ml_ranker_allowed: false" in text
    assert "runtime_authorized: false" in text
