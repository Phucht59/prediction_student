"""Build full OULAD learner-stage/action groups from cutoff-safe states.

Future trajectories are used only for target columns.  No ranking model is
fit here; this phase is deterministic and produces the data contract consumed
by development and lockbox evaluation.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'artifacts/recommend_hybrid/outcome_grounded'; sys.path.insert(0,str(ROOT))
from src.pipelines.oulad import BASE_CHANNELS, _build_bundle

TRANS=[('E1_EARLY_20PCT','E2_EARLY_35PCT','EARLY_20'),('E2_EARLY_35PCT','M1_MIDDLE_FROZEN','EARLY_35'),('M1_MIDDLE_FROZEN','L1_LATE_75PCT','MIDDLE_50')]
ACT=['VLE_ENGAGEMENT','STUDY_SCHEDULE','ASSESSMENT_COMPLETION','RETRIEVAL_PRACTICE','TARGETED_PRACTICE','LEARNING_CONSOLIDATION']
CF_STAGE={'EARLY_20':'EARLY_20','EARLY_35':'EARLY_35','MIDDLE_50':'MIDDLE_50'}
IDX={x:i for i,x in enumerate(BASE_CHANNELS)}

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def dump(name,x):
 OUT.mkdir(parents=True,exist_ok=True); (OUT/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf8')

def rate(seq,s,e,ch): return float(seq[s:e,IDX[ch]].sum()/max(e-s,1))
def action_proxy(a,cur,clen,nxt,nlen):
 if nlen<=clen: return None
 s,e=clen,nlen
 if a=='VLE_ENGAGEMENT': return int(rate(nxt,s,e,'total_clicks')>rate(cur,0,clen,'total_clicks') and nxt[s:e,IDX['active_days']].sum()>0)
 if a=='STUDY_SCHEDULE': return int(rate(nxt,s,e,'active_days')>=rate(cur,0,clen,'active_days') and nxt[e-1,IDX['days_since_last_vle_activity']]<cur[clen-1,IDX['days_since_last_vle_activity']])
 if a=='ASSESSMENT_COMPLETION': return int(nxt[s:e,IDX['submitted_assessment_count']].sum()>0)
 if a in ('RETRIEVAL_PRACTICE','TARGETED_PRACTICE'): return int(rate(nxt,s,e,'quiz_clicks')>rate(cur,0,clen,'quiz_clicks'))
 if a=='LEARNING_CONSOLIDATION': return int(rate(nxt,s,e,'content_clicks')>=rate(cur,0,clen,'content_clicks') and rate(nxt,s,e,'content_clicks')>0)
 return None
def proximal(a,cur,clen,nxt,nlen,ad):
 if ad is None or not ad: return 0
 s,e=clen,nlen
 if a=='VLE_ENGAGEMENT': return int(rate(nxt,s,e,'active_days')>=rate(cur,0,clen,'active_days'))
 if a=='STUDY_SCHEDULE': return int(nxt[e-1,IDX['days_since_last_vle_activity']]<cur[clen-1,IDX['days_since_last_vle_activity']])
 if a=='ASSESSMENT_COMPLETION': return int(nxt[s:e,IDX['available_score_count']].sum()>0)
 return int(rate(nxt,s,e,'total_clicks')>0)

def main():
 OUT.joinpath('dataset').mkdir(parents=True,exist_ok=True)
 bundle=_build_bundle(); pred=pd.read_parquet(ROOT/'artifacts/canonical_v3/predictions/oulad_oof_predictions.parquet'); pred=pred[(pred.model=='hybrid') & (pred.stage.isin(['E1_EARLY_20PCT','E2_EARLY_35PCT','M1_MIDDLE_50PCT']))].copy(); pred['stage_key']=pred.stage.replace({'M1_MIDDLE_50PCT':'M1_MIDDLE_FROZEN'})
 risk=pred.groupby(['base_record_id','stage_key']).probability.mean().to_dict(); seed=pd.read_parquet(ROOT/'artifacts/canonical_v3/predictions/oulad_seed_predictions.parquet'); seed=seed[(seed.model=='hybrid') & (seed.stage.isin(['E1_EARLY_20PCT','E2_EARLY_35PCT','M1_MIDDLE_50PCT']))]; unc=seed.groupby(['base_record_id','stage']).probability.std().fillna(0).to_dict()
 cf=pd.read_parquet(ROOT/'artifacts/recommend_hybrid/counterfactual/full_cohort/action_scores.parquet',columns=['student_key','stage','action_id','risk_reduction','workload_minutes']); cf=cf[cf.action_id.isin(ACT)]; cf_map={(str(r.student_key),str(r.stage),str(r.action_id)):float(r.risk_reduction) for r in cf.itertuples()}
 rows=[]; groups=[]; indexes={s:{str(x):i for i,x in enumerate(d.frame.base_record_id)} for s,d in bundle.stages.items()}
 for src,tgt,stage in TRANS:
  sd,td=bundle.stages[src],bundle.stages[tgt]
  for i,r in sd.frame.iterrows():
   key=str(r.base_record_id); ni=indexes[tgt].get(key)
   if ni is None: continue
   clen=int(sd.lengths[i]); nlen=int(td.lengths[ni]); cur=sd.sequence[i]; nxt=td.sequence[ni]; rid=f'{key}|{stage}'; groups.append({'group_id':rid,'base_record_id':key,'stage':stage,'outer_fold':int(r.outer_fold),'course':str(r.code_module),'presentation':str(r.code_presentation),'target_stage':tgt,'final_favorable':int(r.target)==0,'future_window_available':True})
   riskv=float(risk.get((key,src),np.nan)); u=float(unc.get((key,src.replace('M1_MIDDLE_FROZEN','M1_MIDDLE_50PCT')),0.0));
   vals={'vle_intensity':rate(cur,0,clen,'total_clicks'),'active_days':float(cur[:clen,IDX['active_days']].sum()),'inactive_streak':float(cur[clen-1,IDX['days_since_last_vle_activity']]),'activity_trend':float(cur[clen-1,IDX['total_clicks']]-cur[0,IDX['total_clicks']]),'assessment_completion':float(cur[clen-1,IDX['cumulative_weighted_score']]),'assessment_availability':float(cur[:clen,IDX['available_score_count']].sum()),'studied_credits':float(r.studied_credits),'previous_attempts':float(r.num_of_prev_attempts),'cutoff_day':float(r.cutoff_day),'risk_probability':riskv,'risk_uncertainty':u}
   for a in ACT:
    ad=action_proxy(a,cur,clen,nxt,nlen); pr=proximal(a,cur,clen,nxt,nlen,ad); grade=0 if ad is None or not ad else (2 if pr else 1); rr=cf_map.get((key,CF_STAGE[stage],a),np.nan); workload={'VLE_ENGAGEMENT':90,'STUDY_SCHEDULE':30,'ASSESSMENT_COMPLETION':150,'RETRIEVAL_PRACTICE':75,'TARGETED_PRACTICE':120,'LEARNING_CONSOLIDATION':90}[a]
    rows.append({'group_id':rid,'base_record_id':key,'stage':stage,'outer_fold':int(r.outer_fold),'course':str(r.code_module),'presentation':str(r.code_presentation),'action_id':a,'action_family':a.split('_')[0],'workload_minutes':workload,'evidence_strength':1.0,'action_availability':1,'counterfactual_v1_delta':rr,'adherence':ad,'proximal_positive':pr,'relevance_grade':grade,'final_favorable':int(r.target)==0,**vals})
 groups_df=pd.DataFrame(groups); cand=pd.DataFrame(rows)
 # Cross-fitted distal residual: fold 0 uses fold 1 and vice versa; lockbox uses development only.
 features=['risk_probability','risk_uncertainty','vle_intensity','active_days','inactive_streak','activity_trend','assessment_completion','assessment_availability','studied_credits','previous_attempts','cutoff_day']
 cand['expected_favorable_prob']=np.nan
 for fold in [0,1,2]:
  train=cand[cand.outer_fold.isin([1] if fold==0 else [0] if fold==1 else [0,1])].drop_duplicates('group_id'); test_idx=cand.index[cand.outer_fold==fold]
  if len(train) and train.final_favorable.nunique()>1:
   X=train[features].fillna(0); sc=StandardScaler().fit(X); m=LogisticRegression(max_iter=1000,random_state=20260804).fit(sc.transform(X),train.final_favorable); cand.loc[test_idx,'expected_favorable_prob']=m.predict_proba(sc.transform(cand.loc[test_idx,features].fillna(0)))[:,1]
 cand['residual_positive']=cand.final_favorable.astype(float)>cand.expected_favorable_prob.fillna(cand.final_favorable.mean())
 cand.loc[(cand.relevance_grade>=1)&(cand.proximal_positive==1)&cand.residual_positive,'relevance_grade']=3
 cand['label_available']=cand.adherence.notna().astype(int)
 groups_df.to_parquet(OUT/'dataset/learner_stage_groups.parquet',index=False); cand.to_parquet(OUT/'dataset/candidate_rows.parquet',index=False)
 pd.DataFrame([{'group_id':g.group_id,'source_stage':g.stage,'future_behavior_available':1,'final_favorable':g.final_favorable} for g in groups_df.itertuples()]).to_parquet(OUT/'dataset/future_outcomes.parquet',index=False)
 pd.DataFrame([{'group_id':r.group_id,'action_id':r.action_id,'adherence':r.adherence,'proximal_positive':r.proximal_positive} for r in cand.itertuples()]).to_parquet(OUT/'dataset/future_behavior_targets.parquet',index=False)
 registry=[{'feature_name':f,'source_table':'OULAD studentInfo/studentVle/studentAssessment or frozen OOF prediction','source_date_rule':'at_or_before_source_cutoff','mutable_or_static':'pre_cutoff','allowed_for_ranking':True,'reason':'registered cutoff-safe feature'} for f in features]+[{'feature_name':x,'source_table':'action catalog/counterfactual V1','source_date_rule':'frozen pre-cutoff candidate metadata','mutable_or_static':'action','allowed_for_ranking':True,'reason':'action feature'} for x in ['action_id','action_family','workload_minutes','evidence_strength','action_availability','counterfactual_v1_delta']]
 dump('dataset/feature_registry.json',{'schema_version':'feature_registry_v1','features':registry,'prohibited':['final_result','future_assessment_score','future_submission','future_vle_clicks','future_active_days','future_inactive_streak','future_adherence','label','target','post_cutoff_aggregate','protected_attributes']})
 dump('dataset/cohort_flow.json',{'raw_learner_course_records':len(bundle.base),'eligible_transition_groups':len(groups_df),'candidate_rows':len(cand),'excluded_missing_future_window':int(sum(1 for s,t,_ in TRANS for k in indexes[s] if k not in indexes[t])),'groups_by_fold':groups_df.groupby('outer_fold').size().to_dict(),'groups_by_stage':groups_df.groupby('stage').size().to_dict(),'zero_relevance_groups':int((cand.groupby('group_id').relevance_grade.max()==0).sum()),'missing_action_proxy':int(cand.adherence.isna().sum())})
 dump('dataset/label_schema.json',{'schema_version':'graded_relevance_v1','grades':{'0':'no_action_specific_adherence','1':'adherence_without_proximal_positive','2':'adherence_and_proximal_positive','3':'adherence_proximal_and_positive_residual'},'thresholds':'fixed semantic comparisons; no test-fold fitting','final_result_not_used_as_sole_action_label':True})
 files=[p for p in (OUT/'dataset').iterdir() if p.is_file()]; dump('dataset/CHECKSUMS.json',{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files})
 print(json.dumps({'groups':len(groups_df),'candidates':len(cand),'zero_relevance_groups':int((cand.groupby('group_id').relevance_grade.max()==0).sum())},indent=2))
if __name__=='__main__': main()
