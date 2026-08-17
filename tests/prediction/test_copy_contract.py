from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[2]


def test_final_hybrid_identity_and_artifacts():
    cfg = json.loads((ROOT / "artifacts/hybrid/phase8/final_development/hybrid_config.json").read_text(encoding="utf-8"))
    assert cfg["model_id"] == "hybrid"
    assert cfg["display_name"] == "Hybrid"
    assert (ROOT / "artifacts/prediction/final/results.csv").exists()
    assert (ROOT / "artifacts/prediction/final/summary.csv").exists()
    assert (ROOT / "artifacts/prediction/final/predictions/predictions.csv").exists()


def test_phase8_model_imports_without_training():
    from src.hybrid.phase8.model import Phase8HybridConfig, Phase8UnifiedHybrid

    model = Phase8UnifiedHybrid(Phase8HybridConfig(static_dim=3, temporal_dim=4, aggregate_dim=2))
    assert model.model_id == "hybrid"
    assert model.display_name == "Unified Hybrid"
