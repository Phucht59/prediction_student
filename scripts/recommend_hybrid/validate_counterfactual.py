"""Validate final evidence claims and the non-causal module boundary."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    evidence = json.loads((ROOT / "artifacts/recommend_hybrid/final/CONDITIONAL_ACTION_FINAL_EVIDENCE.json").read_text())
    assert evidence["status"] == "COMPLETE"
    assert evidence["groups"] if "groups" in evidence else evidence["overall"]["groups"] == 9304
    assert evidence["release"]["runtime_authorized"] is False
    assert evidence["claim_boundary"] == "OFFLINE_CONDITIONAL_ACTION_RANKING_NOT_END_TO_END_OR_CAUSAL_EFFECT"
    assert evidence["overall"]["precision_at_1"] == 0.9374462596732588
    print("COUNTERFACTUAL_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
