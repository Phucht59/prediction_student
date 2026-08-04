"""Memory-bounded deterministic cache builder for V2.1 evaluation."""
from pathlib import Path
import hashlib,json,os,time,pandas as pd
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'artifacts/recommend_hybrid/outcome_grounded_v2_1'; DATA=OUT/'dataset'; CACHE=OUT/'cache'; CACHE.mkdir(exist_ok=True)
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def atom(df,p):
 tmp=p.with_suffix(p.suffix+'.tmp'); df.to_parquet(tmp,index=False); os.replace(tmp,p)
def main():
 src=DATA/'candidate_rows.parquet'; d=pd.read_parquet(src,columns=['group_id','base_record_id','stage','outer_fold','course','presentation','action_family','action_available','action_needed','opportunity_count','deficit_score','evidence_strength','prerequisite_status','workload_minutes','risk_probability','risk_uncertainty','active_days','inactive_streak','activity_trend','assessment_progress','vle_intensity','counterfactual_v1_delta','future_behavior_signal','future_proximal_signal','proximal_outcome_available','rankable'])
 reg={'schema_hash':h(OUT/'FEATURE_SCHEMA.json'),'source_checksum':h(src),'files':{},'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 for f in [0,1,2]:
  for kind,x in [('train',d[d.outer_fold!=f]),('test',d[d.outer_fold==f])]:
   p=CACHE/f'fold_{f}_{kind}.parquet'; atom(x,p); reg['files'][p.name]={'sha256':h(p),'rows':len(x)}
 atom(d[d.presentation.isin(['2013J','2014B'])],CACHE/'temporal_train.parquet'); atom(d[d.presentation=='2014J'],CACHE/'temporal_test.parquet')
 for n in ['temporal_train.parquet','temporal_test.parquet']: reg['files'][n]={'sha256':h(CACHE/n),'rows':len(pd.read_parquet(CACHE/n))}
 (CACHE/'cache_registry.json').write_text(json.dumps(reg,indent=2,sort_keys=True)+'\n')
 prog={'DATASET':{'status':'COMPLETE'},'LABELS':{'status':'COMPLETE'},'ELIGIBILITY':{'status':'COMPLETE'},'FOLD_0':{'status':'PENDING'},'FOLD_1':{'status':'PENDING'},'FOLD_2':{'status':'PENDING'},'TEMPORAL':{'status':'PENDING'},'BASELINES':{'status':'PENDING'},'NEGATIVE_CONTROLS':{'status':'PENDING'},'ABLATIONS':{'status':'PENDING'},'BOOTSTRAP':{'status':'PENDING'},'FAIRNESS':{'status':'PENDING'},'RUNTIME':{'status':'NOT_AUTHORIZED'},'RELEASE':{'status':'PENDING'}}
 (OUT/'PROGRESS.json').write_text(json.dumps({'schema_version':'v2.1_progress_v1','stages':prog},indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
