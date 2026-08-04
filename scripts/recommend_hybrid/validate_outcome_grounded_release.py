"""Fail-closed validation of the V2 release registry and artifact contract."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'artifacts/recommend_hybrid/outcome_grounded'
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 r=json.loads((OUT/'RELEASE_REGISTRY.json').read_text()); a=json.loads((OUT/'INPUT_AUTHORITY.json').read_text()); l=json.loads((OUT/'LOCKBOX_REGISTRY.json').read_text());
 assert r['status']=='OUTCOME_GROUNDED_OFFLINE_EVIDENCE_INCONCLUSIVE'; assert r['merge_allowed'] is False; assert l['prior_executions']==1; assert a['lockbox_fold']==2
 required=['protocol.json','INPUT_AUTHORITY.json','LOCKBOX_REGISTRY.json','feature_registry.json','label_schema.json','cohort_flow.json','development_results.json','lockbox_results.json','baseline_comparison.csv','ablation_results.csv','negative_controls.csv','fairness_audit.json','stability_analysis.csv','bootstrap_results.json','RELEASE_REGISTRY.json','CHECKSUMS.json','ranking_predictions.parquet']
 missing=[x for x in required if not (OUT/x).exists()]; assert not missing,missing
 print('OUTCOME_GROUNDED_RELEASE_VALIDATION_PASS')
if __name__=='__main__': main()
