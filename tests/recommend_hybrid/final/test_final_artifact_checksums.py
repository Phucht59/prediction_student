import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = ROOT / "artifacts/recommend_hybrid/final/FINAL_ARTIFACT_MANIFEST.json"


def test_final_artifact_checksums_match_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["scientific_values_changed"] is False
    for item in manifest["artifacts"]:
        path = ROOT / item["final_path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == item["final_sha256"]
        assert item["scientific_values_changed"] is False
        if item["content_changed"] is False:
            assert item["source_sha256"] == item["final_sha256"]
        else:
            assert item["source_sha256"] != item["final_sha256"]


def test_final_evidence_preserves_registered_scientific_values():
    evidence = json.loads(
        (
            ROOT
            / "artifacts/recommend_hybrid/final/CONDITIONAL_ACTION_FINAL_EVIDENCE.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["overall"]["precision_at_1"] == 0.9374462596732588
    assert evidence["overall"]["ndcg_at_3"] == 0.9723128213125097
    assert evidence["overall"]["mrr"] == 0.9668690885640585
    assert evidence["release"]["status"] == (
        "CONDITIONAL_ACTION_RANKING_OFFLINE_VALIDATED"
    )
    assert evidence["release"]["runtime_authorized"] is False
    assert evidence["end_to_end_context"]["status"] == (
        "TWO_STAGE_V4_EVIDENCE_BELOW_GATE"
    )
