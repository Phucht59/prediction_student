import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_frozen_prediction_authority_is_registered():
    text = (ROOT / "configs/recommend_hybrid/final/prediction_authority.yaml").read_text()
    assert "parameter_count: 160492" in text
    manifest = json.loads(
        (ROOT / "artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json").read_text()
    )
    assert manifest["parameter_count"] == 160492
    assert manifest["status"] == "RELEASE_FROZEN"
