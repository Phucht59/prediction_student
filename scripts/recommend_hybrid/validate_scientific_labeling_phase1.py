"""Validate the locked Phase 1 scientific-labeling foundation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.weak_supervision.labels import (  # noqa: E402
    LF_ABSTAIN,
    RELEVANCE_VALUES,
    TARGET_VALUES,
)
from src.recommend_hybrid.weak_supervision.registry import (  # noqa: E402
    load_action_mappings,
    load_sources,
    stable_json,
)
from src.recommend_hybrid.weak_supervision.validation import (  # noqa: E402
    PREDICTION_AUTHORITIES,
    validate_registries,
)

PASS_TOKEN = "RECOMMEND_SCIENTIFIC_LABELING_PHASE1_PASS"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate(root: Path = ROOT) -> dict:
    config = _yaml(root / "configs/recommend_hybrid/scientific_labeling.yaml")
    source_path = root / config["source_registry_path"]
    action_path = root / config["action_evidence_map_path"]
    sources = load_sources(source_path)
    actions = load_action_mappings(action_path)
    validate_registries(sources, actions)

    if set(config["labels"].values()) != TARGET_VALUES:
        raise ValueError("target labels must be exactly 0/1/2")
    if config["lf_abstain"] != LF_ABSTAIN or LF_ABSTAIN in TARGET_VALUES:
        raise ValueError("LF abstain must be -1 and separate from target labels")
    if set(config["ranking_relevance_grades"].values()) != RELEVANCE_VALUES:
        raise ValueError("ranking relevance grades must be exactly 0/1/2")
    if config["thresholds"]["status"] != "PROVISIONAL_NOT_TUNED":
        raise ValueError("untuned thresholds must remain provisional")
    if set(config["required_prediction_authorities"]) != {
        dataset.value for dataset in PREDICTION_AUTHORITIES
    }:
        raise ValueError("prediction-authority dataset coverage is incomplete")

    expected_actions = set()
    for policy_name in ("policy_uci_mat.yaml", "policy_uci_por.yaml", "policy_oulad.yaml"):
        expected_actions.update(
            _yaml(root / "configs/recommend_hybrid" / policy_name)["allowed_actions"]
        )
    if {action.action_id for action in actions} != expected_actions:
        raise ValueError("action map must cover the canonical policy action union")

    verified = sum(source.verification_status == "VERIFIED" for source in sources)
    unverified = len(sources) - verified
    result = {
        "action_count": len(actions),
        "evidence_mapped": sum(action.status == "EVIDENCE_MAPPED" for action in actions),
        "gate": PASS_TOKEN,
        "insufficient_evidence": sum(
            action.status == "INSUFFICIENT_EVIDENCE" for action in actions
        ),
        "source_count": len(sources),
        "unverified_sources": unverified,
        "verified_sources": verified,
    }
    gate_path = root / "reports/recommend_hybrid/scientific_labeling/PHASE1_GATE.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if stable_json(result) != stable_json(json.loads(gate_path.read_text(encoding="utf-8"))):
        raise ValueError("gate serialization is not deterministic")
    return result


def main() -> int:
    validate()
    print(PASS_TOKEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
