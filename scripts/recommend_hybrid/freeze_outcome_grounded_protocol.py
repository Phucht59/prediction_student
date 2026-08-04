"""Create the immutable V2 protocol and input authority registry."""
from __future__ import annotations
import hashlib, json, subprocess
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/recommend_hybrid/outcome_grounded'
CFG=ROOT/'configs/recommend_hybrid/outcome_grounded_oulad.yaml'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b): h.update(b)
    return h.hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf8'); q.replace(p)
def main():
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='codex/constrained-counterfactual-recommender': raise SystemExit('wrong branch')
    cfg=yaml.safe_load(CFG.read_text(encoding='utf8'))
    if cfg['status']!='PREREGISTERED_LOCKED_BEFORE_V2_DEVELOPMENT': raise SystemExit('protocol not preregistered')
    protocol={'schema_version':'outcome_grounded_protocol_v1','locked_at':datetime.now(timezone.utc).isoformat(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'config':cfg,'v1_status':'COUNTERFACTUAL_V1_ENGINEERING_COMPLETE_EXTERNAL_VALIDATION_FAILED','input_authority':'frozen_risk_predictor_only; V1 delta optional feature; never a primary label or metric'}
    write(OUT/'protocol.json',protocol)
    inputs=[CFG,ROOT/'configs/recommend_hybrid/historical_action_outcome_mapping.yaml',ROOT/'configs/recommend_hybrid/actions.yaml',ROOT/'artifacts/recommend_hybrid/RESIDUAL_CHECKPOINT_RELEASE_MANIFEST.json',ROOT/'artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json',ROOT/'artifacts/canonical_v3/predictions/oulad_oof_predictions.parquet',ROOT/'data/manifests/extension_raw_manifest.json',ROOT/'data/processed/study_c_oulad/manifests/split_manifest.csv']
    authority={'schema_version':'outcome_grounded_input_authority_v1','protocol_sha256':sha(OUT/'protocol.json'),'git_commit':protocol['git_commit'],'inputs':[{'path':str(x.relative_to(ROOT)).replace('\\','/'),'sha256':sha(x)} for x in inputs],'lockbox_fold':2,'development_folds':[0,1],'claim_boundary':'OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT','prohibited_as_primary_metric':['model-estimated risk reduction','Success@0.01','Success@0.05','threshold crossing']}
    write(OUT/'INPUT_AUTHORITY.json',authority)
    write(OUT/'LOCKBOX_REGISTRY.json',{'status':'NOT_OPENED','protocol_sha256':authority['protocol_sha256'],'lockbox_fold':2,'prior_executions':0,'opened_at':None})
    print(json.dumps({'protocol_sha256':authority['protocol_sha256'],'git_commit':protocol['git_commit']},indent=2))
if __name__=='__main__': main()
