from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_final_namespace_uses_integrated_head_only():
    text = "\n".join(
        path.read_text(errors="ignore").lower()
        for root in (ROOT / "src/recommend_hybrid/final", ROOT / "configs/recommend_hybrid/final")
        for path in root.rglob("*")
        if path.is_file()
    )
    assert "external_ml_ranker_allowed: false" in text
    assert "conditional_hybrid_action_ranker" in text
