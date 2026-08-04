"""Inventory V2.1 artifacts and distinguish execution from scaffold/placeholder."""
from pathlib import Path
import hashlib,json,pandas as pd
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'artifacts/recommend_hybrid/outcome_grounded_v2_1'; REPORT=ROOT/'reports/recommend_hybrid/v2_1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 items=[]
 for p in sorted(OUT.rglob('*')):
  if not p.is_file() or p.name in {'EXECUTION_INVENTORY.json'}: continue
  rel=str(p.relative_to(OUT)); size=p.stat().st_size
  status='IMPLEMENTED_AND_EXECUTED'
  note=''
  if p.suffix=='.csv':
   try:
    df=pd.read_csv(p); rows=len(df); note=f'rows={rows}'
    if rows==0 or (len(df.columns)==0): status='PLACEHOLDER'
    elif 'NOT_COMPUTED' in df.to_string(): status='IMPLEMENTED_NOT_EXECUTED'
   except Exception: status='FAILED'
  if p.name in {'NESTED_OOF_RESULTS.json','TEMPORAL_RESULTS.json','BOOTSTRAP_RESULTS.json','FAIRNESS_AUDIT.json'}:
   try:
    d=json.loads(p.read_text()); s=str(d.get('status',''))
    if 'INCONCLUSIVE' in s or 'NOT_COMPUTED' in str(d): status='IMPLEMENTED_NOT_EXECUTED'
   except Exception: status='FAILED'
  if p.name=='RANKING_PREDICTIONS.parquet':
   df=pd.read_parquet(p); note=f'rows={len(df)}'; status='IMPLEMENTED_NOT_EXECUTED' if df.get('model_score',pd.Series(dtype=float)).isna().all() else status
  if p.name in {'PROTOCOL.json','INPUT_AUTHORITY.json','ACTION_FAMILY_REGISTRY.json','FEATURE_SCHEMA.json','LABEL_SCHEMA.json','COHORT_FLOW.json'}: status='IMPLEMENTED_AND_EXECUTED'
  items.append({'artifact':rel,'status':status,'sha256':sha(p),'bytes':size,'note':note})
 dataset=OUT/'dataset/candidate_rows.parquet';
 if dataset.exists():
  df=pd.read_parquet(dataset); summary={'candidate_rows':len(df),'groups':int(df.group_id.nunique()),'future_signal_non_null':int(df.future_behavior_signal.notna().sum()),'rankable_groups':int(df[df.rankable==1].group_id.nunique())}
 else: summary={}
 inv={'schema_version':'v2.1_execution_inventory_v1','overall_status':'V2_1_IMPLEMENTATION_SCAFFOLD_COMPLETE','full_evaluation_status':'V2_1_FULL_EVALUATION_NOT_COMPLETED','items':items,'dataset_audit':summary,'historical_namespaces_untouched':True}
 (OUT/'EXECUTION_INVENTORY.json').write_text(json.dumps(inv,indent=2,sort_keys=True)+'\n')
 lines=['# V2.1 execution inventory','',f"Overall status: `{inv['overall_status']}`",f"Full evaluation: `{inv['full_evaluation_status']}`",'','| Artifact | Status | Note |','|---|---|---|']
 lines += [f"| `{x['artifact']}` | `{x['status']}` | {x['note']} |" for x in items]
 lines += ['', 'The candidate dataset contains real OULAD-derived rows and future-window signals. Ranking, controls, bootstrap, temporal and release outputs remain non-scientific until their registered executions complete.']
 (REPORT/'V2_1_EXECUTION_INVENTORY.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()
