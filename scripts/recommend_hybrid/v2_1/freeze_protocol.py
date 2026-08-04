from __future__ import annotations
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'artifacts/recommend_hybrid/outcome_grounded_v2_1'; CFG=ROOT/'configs/recommend_hybrid/outcome_grounded_v2_1.yaml'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def write(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf8')
def main():
 cfg=yaml.safe_load(CFG.read_text(encoding='utf8')); assert cfg['status']=='PREREGISTERED_LOCKED_BEFORE_V2_1_DATASET'
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(); protocol={'schema_version':'outcome_grounded_v2_1_protocol_v1','locked_at':datetime.now(timezone.utc).isoformat(),'git_commit':commit,'v1_status':'COUNTERFACTUAL_V1_ENGINEERING_COMPLETE_EXTERNAL_VALIDATION_FAILED','v2_status':'OUTCOME_GROUNDED_OFFLINE_EVIDENCE_INCONCLUSIVE','config':cfg}
 write(OUT/'PROTOCOL.json',protocol)
 inputs=[CFG,ROOT/'configs/recommend_hybrid/historical_action_outcome_mapping.yaml',ROOT/'configs/recommend_hybrid/actions.yaml',ROOT/'artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json',ROOT/'artifacts/canonical_v3/predictions/oulad_oof_predictions.parquet',ROOT/'data/manifests/extension_raw_manifest.json',ROOT/'data/processed/study_c_oulad/manifests/split_manifest.csv']
 write(OUT/'INPUT_AUTHORITY.json',{'schema_version':'outcome_grounded_v2_1_input_authority_v1','protocol_sha256':sha(OUT/'PROTOCOL.json'),'git_commit':commit,'inputs':[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha(p)} for p in inputs],'lockbox_reopened':False,'evaluation_design':'NESTED_GROUPED_CV_ALL_OUTER_FOLDS','claim_boundary':cfg['claim_boundary']})
 print(json.dumps({'protocol_sha256':sha(OUT/'PROTOCOL.json'),'git_commit':commit},indent=2))
if __name__=='__main__': main()
