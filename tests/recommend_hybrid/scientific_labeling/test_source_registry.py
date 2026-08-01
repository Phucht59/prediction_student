from pathlib import Path

from src.recommend_hybrid.weak_supervision.registry import (
    load_action_mappings,
    load_sources,
    stable_json,
)
from src.recommend_hybrid.weak_supervision.validation import validate_registries

ROOT = Path(__file__).resolve().parents[3]


def test_registry_metadata_and_action_coverage() -> None:
    sources = load_sources(
        ROOT / "artifacts/recommend_hybrid/scientific_labeling/source_registry.yaml"
    )
    actions = load_action_mappings(
        ROOT / "artifacts/recommend_hybrid/scientific_labeling/action_evidence_map.yaml"
    )
    validate_registries(sources, actions)
    assert sources
    assert all(source.verification_status == "VERIFIED" for source in sources)
    assert all(
        action.evidence_source_ids or action.status == "INSUFFICIENT_EVIDENCE"
        for action in actions
    )


def test_serialization_is_deterministic() -> None:
    sources = load_sources(
        ROOT / "artifacts/recommend_hybrid/scientific_labeling/source_registry.yaml"
    )
    payload = [source.to_dict() for source in sources]
    assert stable_json(payload) == stable_json(payload)
