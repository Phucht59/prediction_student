"""Build opportunity-aware, action-family-specific V2.1 targets."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'artifacts/recommend_hybrid/outcome_grounded_v2_1'; sys.path.insert(0,str(ROOT))
from src.pipelines.oulad import BASE_CHANNELS,_build_bundle
TRANS=[('E1_EARLY_20PCT','E2_EARLY_35PCT','EARLY_20'),('E2_EARLY_35PCT','M1_MIDDLE_FROZEN','EARLY_35'),('M1_MIDDLE_FROZEN','L1_LATE_75PCT','MIDDLE_50')]
FAMILIES=['ASSESSMENT_COMPLETION','STUDY_REGULARITY','VLE_ENGAGEMENT','QUIZ_OR_RETRIEVAL_PRACTICE','CONTENT_REVIEW']; IDX={x:i for i,x in enumerate(BASE_CHANNELS)}
FAMILY_CF={'ASSESSMENT_COMPLETION':'ASSESSMENT_COMPLETION','STUDY_REGULARITY':'STUDY_SCHEDULE','VLE_ENGAGEMENT':'VLE_ENGAGEMENT','QUIZ_OR_RETRIEVAL_PRACTICE':'RETRIEVAL_PRACTICE','CONTENT_REVIEW':'LEARNING_CONSOLIDATION'}
VLE_TYPES={'QUIZ_OR_RETRIEVAL_PRACTICE':{'quiz','externalquiz','questionnaire'},'CONTENT_REVIEW':{'resource','oucontent','page','subpage','url','folder','glossary'}}
WORK={'ASSESSMENT_COMPLETION':150,'STUDY_REGULARITY':30,'VLE_ENGAGEMENT':90,'QUIZ_OR_RETRIEVAL_PRACTICE':75,'CONTENT_REVIEW':90}
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def write(name,x):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf8')
def rate(x,s,e,ch): return float(x[s:e,IDX[ch]].sum()/max(e-s,1))
def main():
 bundle=_build_bundle(); OUT.joinpath('dataset').mkdir(parents=True,exist_ok=True)
 # Course schedule is public before cutoff; it defines opportunity, never future behavior.
 ass=pd.read_csv(ROOT/'data/raw/assessments.csv',usecols=['code_module','code_presentation','date','weight']); ass=ass[ass.weight>0]
 ass_map={(m,p):g[['date','weight']].to_numpy() for (m,p),g in ass.groupby(['code_module','code_presentation'])}
 vle=pd.read_csv(ROOT/'data/raw/vle.csv',usecols=['code_module','code_presentation','activity_type','week_from','week_to']);
 schedule={(m,p):g for (m,p),g in vle.groupby(['code_module','code_presentation'])}
 pred=pd.read_parquet(ROOT/'artifacts/canonical_v3/predictions/oulad_oof_predictions.parquet'); pred=pred[(pred.model=='hybrid') & pred.stage.isin(['E1_EARLY_20PCT','E2_EARLY_35PCT','M1_MIDDLE_50PCT'])]; pred=pred.assign(stage_key=pred.stage.replace({'M1_MIDDLE_50PCT':'M1_MIDDLE_FROZEN'})); risk=pred.groupby(['base_record_id','stage_key']).probability.mean().to_dict()
 seed=pd.read_parquet(ROOT/'artifacts/canonical_v3/predictions/oulad_seed_predictions.parquet'); seed=seed[(seed.model=='hybrid') & seed.stage.isin(['E1_EARLY_20PCT','E2_EARLY_35PCT','M1_MIDDLE_50PCT'])]; unc=seed.groupby(['base_record_id','stage']).probability.std().fillna(0).to_dict()
 cf=pd.read_parquet(ROOT/'artifacts/recommend_hybrid/counterfactual/full_cohort/action_scores.parquet',columns=['student_key','stage','action_id','risk_reduction']); cf_map={(str(x.student_key),str(x.stage),str(x.action_id)):float(x.risk_reduction) for x in cf.itertuples()}
 indexes={s:{str(x):i for i,x in enumerate(d.frame.base_record_id)} for s,d in bundle.stages.items()}; groups=[]; rows=[]
 for src,tgt,stage in TRANS:
  sd,td=bundle.stages[src],bundle.stages[tgt]
  for i,r in sd.frame.iterrows():
   key=str(r.base_record_id); ni=indexes[tgt].get(key)
   if ni is None: continue
   cur=sd.sequence[i]; nxt=td.sequence[ni]; clen,nlen=int(sd.lengths[i]),int(td.lengths[ni]); cday=float(r.cutoff_day); tday=float(td.frame.iloc[ni].cutoff_day); window=max(1,tday-cday); weeks=max(1,nlen-clen); module,pres=str(r.code_module),str(r.code_presentation); g=f'{key}|{stage}'
   sched=schedule.get((module,pres),pd.DataFrame()); week_lo=cday/7; week_hi=tday/7
   def opp(types=None):
    if len(sched)==0:return 0
    q=sched[(sched.week_to.fillna(sched.week_from)>=week_lo)&(sched.week_from.fillna(sched.week_to)<=week_hi)]
    if types is not None:q=q[q.activity_type.isin(types)]
    return int(q.shape[0])
   ao=ass_map.get((module,pres),np.empty((0,2))); assessment_opp=int(((ao[:,0]>cday)&(ao[:,0]<=tday)).sum()) if len(ao) else 0
   active_ratio=rate(cur,0,clen,'active_days')/7; current_click=rate(cur,0,clen,'total_clicks'); current_quiz=rate(cur,0,clen,'quiz_clicks'); current_content=rate(cur,0,clen,'content_clicks'); inactive=float(cur[clen-1,IDX['days_since_last_vle_activity']]);
   opps={'ASSESSMENT_COMPLETION':assessment_opp,'STUDY_REGULARITY':int(window),'VLE_ENGAGEMENT':opp(),'QUIZ_OR_RETRIEVAL_PRACTICE':opp(VLE_TYPES['QUIZ_OR_RETRIEVAL_PRACTICE']),'CONTENT_REVIEW':opp(VLE_TYPES['CONTENT_REVIEW'])}
   recent_delta=float(cur[clen-1,IDX['total_clicks']]-cur[max(0,clen-2),IDX['total_clicks']]); needed={'ASSESSMENT_COMPLETION':int(assessment_opp>0 and cur[clen-1,IDX['cumulative_weighted_score']]<0.8),'STUDY_REGULARITY':int(active_ratio<0.5 or inactive>=2),'VLE_ENGAGEMENT':int(current_click<10 or recent_delta<0),'QUIZ_OR_RETRIEVAL_PRACTICE':int(current_quiz<1),'CONTENT_REVIEW':int(current_content<5)}
   eligible=[f for f in FAMILIES if opps[f]>0 and needed[f]]; rankable=len(eligible)>=2
   groups.append({'group_id':g,'base_record_id':key,'stage':stage,'source_stage':src,'target_stage':tgt,'outer_fold':int(r.outer_fold),'course':module,'presentation':pres,'rankable':rankable,'eligible_action_count':len(eligible),'final_favorable':int(r.target)==0,'cutoff_day':cday,'target_cutoff_day':tday})
   for f in eligible:
    s,e=clen,nlen; oppn=opps[f]
    if f=='ASSESSMENT_COMPLETION': behavior=float(nxt[s:e,IDX['submitted_assessment_count']].sum()/max(assessment_opp,1)); proximal=float(nxt[s:e,IDX['available_score_count']].sum()/max(assessment_opp,1)); prox_avail=int(assessment_opp>0)
    elif f=='STUDY_REGULARITY': behavior=float(nxt[s:e,IDX['active_days']].sum()/max(window,1)); proximal=np.nan; prox_avail=0
    elif f=='VLE_ENGAGEMENT': behavior=float(nxt[s:e,IDX['total_clicks']].sum()/max(oppn,1)); proximal=np.nan; prox_avail=0
    elif f=='QUIZ_OR_RETRIEVAL_PRACTICE': behavior=float(nxt[s:e,IDX['quiz_clicks']].sum()/max(oppn,1)); proximal=np.nan; prox_avail=0
    else: behavior=float(nxt[s:e,IDX['content_clicks']].sum()/max(oppn,1)); proximal=np.nan; prox_avail=0
    evidence=float(min(1.0,oppn/10.0))*(0.5+0.5*needed[f]); deficit=float({'ASSESSMENT_COMPLETION':max(0.0,1-cur[clen-1,IDX['cumulative_weighted_score']]),'STUDY_REGULARITY':max(0,0.5-active_ratio)+max(0,inactive-1)/10,'VLE_ENGAGEMENT':max(0,10-current_click)/10,'QUIZ_OR_RETRIEVAL_PRACTICE':max(0,1-current_quiz),'CONTENT_REVIEW':max(0,5-current_content)/5}[f]); cfstage={'EARLY_20':'EARLY_20','EARLY_35':'EARLY_35','MIDDLE_50':'MIDDLE_50'}[stage]; rows.append({'group_id':g,'base_record_id':key,'stage':stage,'outer_fold':int(r.outer_fold),'course':module,'presentation':pres,'action_family':f,'action_available':1,'action_needed':needed[f],'opportunity_count':oppn,'deficit_score':deficit,'evidence_strength':evidence,'prerequisite_status':1,'workload_minutes':WORK[f],'risk_probability':float(risk.get((key,src),np.nan)),'risk_uncertainty':float(unc.get((key,src.replace('M1_MIDDLE_FROZEN','M1_MIDDLE_50PCT')),0.0)),'active_days':float(cur[:clen,IDX['active_days']].sum()),'inactive_streak':inactive,'activity_trend':recent_delta,'assessment_progress':float(cur[clen-1,IDX['cumulative_weighted_score']]),'vle_intensity':current_click,'counterfactual_v1_delta':cf_map.get((key,cfstage,FAMILY_CF[f]),np.nan),'future_behavior_signal':behavior,'future_proximal_signal':proximal,'proximal_outcome_available':prox_avail,'final_favorable':int(r.target)==0})
 gdf=pd.DataFrame(groups); cdf=pd.DataFrame(rows); cdf=cdf.merge(gdf[['group_id','rankable']],on='group_id',how='left'); cdf=cdf[cdf.rankable].copy();
 gdf.to_parquet(OUT/'dataset/learner_stage_groups.parquet',index=False); cdf.to_parquet(OUT/'dataset/candidate_rows.parquet',index=False); pd.DataFrame(rows).to_parquet(OUT/'dataset/all_candidate_rows_before_rankability.parquet',index=False)
 write('COHORT_FLOW.json',{'raw_learner_course_records':len(bundle.base),'transition_groups':len(gdf),'rankable_groups':int(gdf.rankable.sum()),'not_rankable_policy_fallback':int((~gdf.rankable).sum()),'candidate_rows_rankable':len(cdf),'candidate_rows_before_rankability':len(rows),'eligible_actions_by_family':cdf.action_family.value_counts().to_dict(),'groups_by_fold':gdf.groupby('outer_fold').size().to_dict(),'groups_by_stage':gdf.groupby('stage').size().to_dict(),'course_schedule_used':True,'action_availability_hardcoded':False})
 write('ACTION_FAMILY_REGISTRY.json',{'schema_version':'scientific_action_family_v2_1','families':FAMILIES,'duplicate_proxy_resolution':{'RETRIEVAL_PRACTICE':'merged_into_QUIZ_OR_RETRIEVAL_PRACTICE','TARGETED_PRACTICE':'merged_into_QUIZ_OR_RETRIEVAL_PRACTICE'},'policy_only_actions':['INSTRUCTOR_CONTACT','ADVISOR_ESCALATION','DIAGNOSTIC_CHECK','PROGRESS_MONITORING']})
 features=[c for c in cdf.columns if c not in ['group_id','base_record_id','outer_fold','future_behavior_signal','future_proximal_signal','proximal_outcome_available','final_favorable','rankable','action_available','action_needed']]
 write('FEATURE_SCHEMA.json',{'features':features,'prohibited':['final_favorable','future_behavior_signal','future_proximal_signal','proximal_outcome_available','label','target','post_cutoff_aggregate','protected_attributes'],'opportunity_normalized':True,'precutoff_only':True})
 write('LABEL_SCHEMA.json',{'continuous_primary':'0.6*standardized_behavior_residual + 0.4*standardized_proximal_residual_when_available','no_proximal':'standardized_behavior_residual','grade_thresholds':'outer-training median/q75/q90','final_outcome':'secondary_only'})
 print(json.dumps({'transition_groups':len(gdf),'rankable_groups':int(gdf.rankable.sum()),'candidate_rows':len(cdf),'fallback_groups':int((~gdf.rankable).sum())},indent=2))
if __name__=='__main__': main()
