import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_claim_boundary_and_status_are_preserved():
    evidence = json.loads((ROOT / "artifacts/recommend_hybrid/final/CONDITIONAL_ACTION_FINAL_EVIDENCE.json").read_text())
    assert evidence["claim_boundary"] == "OFFLINE_CONDITIONAL_ACTION_RANKING_NOT_END_TO_END_OR_CAUSAL_EFFECT"
    assert evidence["release"]["status"] == "CONDITIONAL_ACTION_RANKING_OFFLINE_VALIDATED"
    assert evidence["release"]["runtime_authorized"] is False
    assert evidence["end_to_end_context"]["end_to_end_precision_at_1"] == 0.6588623743058682
