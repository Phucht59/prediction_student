"""V3 hygiene: no Panel B, no Gemini runtime, no H1 checkpoint load."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V3 = ROOT / "src" / "recommend_hybrid" / "v3"


def test_v3_metrics_does_not_import_legacy_final():
    text = (V3 / "metrics.py").read_text(encoding="utf-8")
    assert "recommend_hybrid.final" not in text


def test_v3_source_has_no_panel_b_or_gemini_runtime():
    forbidden = (
        "panel_b_real_external_reviews",
        "PANEL_B_FINAL_HELDOUT",
        "google.generativeai",
        "genai.Client",
        "H1_TABULAR_RESIDUAL_EXPERT",
        "oulad_h1_shared",
        "StudentRepresentation",
    )
    for path in V3.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} contains {token}"


def test_package_import_is_v3_not_legacy_final():
    import src.recommend_hybrid as rec

    assert rec.RecommendationPipeline is rec.RecommendationV3Pipeline
    try:
        import src.recommend_hybrid.final as legacy
    except ImportError as exc:
        assert "v3" in str(exc).lower() or "Panel B" in str(exc) or "legacy" in str(exc).lower()
    else:
        raise AssertionError("legacy final package should not import")


def test_v3_does_not_import_h1_checkpoint_manifest():
    text = (V3 / "prediction_adapter.py").read_text(encoding="utf-8")
    assert "RECOMMEND_HYBRID_CHECKPOINT_MANIFEST" not in text
    assert "from src.prediction.contracts import PredictionResult" in text
