import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_migrated_artifacts_are_byte_identical():
    manifest = json.loads((ROOT / "artifacts/recommend_hybrid/final/FINAL_ARTIFACT_MANIFEST.json").read_text())
    assert manifest["scientific_values_changed"] is False
    for item in manifest["artifacts"]:
        path = ROOT / item["final_path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == item["source_sha256"] == item["final_sha256"]
        assert item["content_changed"] is False
