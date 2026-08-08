import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RECOMMEND_SRC = ROOT / "src/recommend_hybrid"
FINAL_MANIFEST = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/frozen/final_release_v1"
    / "FINAL_RELEASE_MANIFEST.json"
)
MODEL_DIR = (
    ROOT
    / "artifacts/recommend_hybrid/explainable_v2/frozen/ranker_panel_a_v2"
    / "final_models"
)


def test_release_tree_contains_no_superseded_recommender_namespaces() -> None:
    legacy = {
        "causal",
        "common",
        "counterfactual",
        "final",
        "oulad",
        "scientific_model",
        "two_stage_v3",
        "uci",
        "weak_supervision",
    }
    assert not {path.name for path in RECOMMEND_SRC.iterdir()} & legacy


def test_exactly_five_official_ebm_artifacts_match_release_hashes() -> None:
    release = json.loads(FINAL_MANIFEST.read_text(encoding="utf-8"))
    expected = release["five_model_sha256"]
    model_paths = sorted(MODEL_DIR.glob("*.joblib"))

    assert len(expected) == 5
    assert len(model_paths) == 5
    assert {path.stem for path in model_paths} == set(expected)
    assert {
        path.stem: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in model_paths
    } == expected


def test_public_source_does_not_import_local_test_lab() -> None:
    offenders = []
    for path in RECOMMEND_SRC.rglob("*.py"):
        if "test_lab" in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []
