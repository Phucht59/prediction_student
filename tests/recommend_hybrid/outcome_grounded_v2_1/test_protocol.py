import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'artifacts/recommend_hybrid/outcome_grounded_v2_1'
def test_protocol_locked_and_claim_boundary():
 p=json.loads((OUT/'PROTOCOL.json').read_text()); assert p['config']['claim_boundary']=='OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT'; assert p['config']['status']=='PREREGISTERED_LOCKED_BEFORE_V2_1_DATASET'
def test_old_v2_namespace_is_separate():
 assert (ROOT/'artifacts/recommend_hybrid/outcome_grounded').exists(); assert OUT != ROOT/'artifacts/recommend_hybrid/outcome_grounded'
def test_action_registry_has_five_families():
 p=json.loads((OUT/'ACTION_FAMILY_REGISTRY.json').read_text()); assert len(p['families'])==5
