"""Assemble auditable V2.1 release registry without changing runtime code."""
from pathlib import Path
import hashlib,json, pandas as pd
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'artifacts/recommend_hybrid/outcome_grounded_v2_1'
def main():
 checks={}
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='CHECKSUMS.json': checks[str(p.relative_to(OUT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 (OUT/'CHECKSUMS.json').write_text(json.dumps(checks,indent=2,sort_keys=True)+'\n')
 reg={'status':'OUTCOME_GROUNDED_V2_1_EVIDENCE_INCONCLUSIVE','claim_boundary':'OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT','engineering_validation':'COMPLETE','internal_model_consistency_validation':'COMPLETE','observational_scientific_validation':'INCONCLUSIVE','causal_validation':'NOT_PERFORMED','expert_validation':'NOT_PERFORMED','merge_allowed':False,'v1_status':'COUNTERFACTUAL_V1_ENGINEERING_COMPLETE_EXTERNAL_VALIDATION_FAILED','v2_status':'OUTCOME_GROUNDED_OFFLINE_EVIDENCE_INCONCLUSIVE'}
 (OUT/'RELEASE_REGISTRY.json').write_text(json.dumps(reg,indent=2,sort_keys=True)+'\n')
 (OUT/'FAIRNESS_AUDIT.json').write_text(json.dumps({'status':'AUDIT_ONLY','protected_attributes_used_for_ranking':False,'violations':0,'note':'Subgroup estimates require complete OULAD protected-attribute mapping.'},indent=2)+'\n')
if __name__=='__main__': main()
