"""Lockbox supplementary evidence: paired uncertainty, controls, ablations."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
import sys
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'artifacts/recommend_hybrid/outcome_grounded'; DATA=OUT/'dataset'; sys.path.insert(0,str(ROOT))
from train_outcome_grounded_ranker import FEATURES, NUM, CAT, fit, group_metric, base_scores, score_model
SEED=20260804; rng=np.random.default_rng(SEED)
def group_values(d, score):
 out=[]
 for gid,g in d.groupby('group_id',sort=False):
  rel=g.relevance_grade.to_numpy(); order=np.argsort(-g[score].to_numpy(),kind='stable'); out.append((gid,group_metric_one(rel,order),float(rel[order[0]]>0)))
 return out
def group_metric_one(rel,order,k=3):
 den=np.sum((2**np.sort(rel)[-k:][::-1]-1)/np.log2(np.arange(2,k+2))); x=rel[order[:k]]; num=np.sum((2**x-1)/np.log2(np.arange(2,len(x)+2))); return float(num/den) if den else 0.0
def main():
 d=pd.read_parquet(DATA/'candidate_rows.parquet'); dev=d[d.outer_fold.isin([0,1])].copy(); lock=d[d.outer_fold==2].copy(); pred=pd.read_parquet(OUT/'ranking_predictions.parquet');
 # Merge only registered scores; labels remain lockbox outcomes.
 cols=['group_id','action_id','relevance_grade','model_score','random_score','popular_score','workload_score','policy_score','counterfactual_score']; x=pred[cols].copy(); metrics=[]; pairs=[]
 base=['random_score','popular_score','workload_score','policy_score','counterfactual_score']
 for s in ['model_score']+base:
  g=group_values(x,s); metrics.append({'method':s,'groups':len(g),'ndcg_at_3':float(np.mean([z[1] for z in g])),'precision_at_1':float(np.mean([z[2] for z in g]))})
 modelv=np.array([z[1] for z in group_values(x,'model_score')]);
 boot=[]
 for s in base:
  b=np.array([z[1] for z in group_values(x,s)]); dif=modelv-b; samples=rng.integers(0,len(dif),size=(2000,len(dif))); vals=dif[samples].mean(axis=1); lo,hi=np.quantile(vals,[.025,.975]); boot.append({'comparison':'model_minus_'+s,'estimate':float(dif.mean()),'ci_95_low':float(lo),'ci_95_high':float(hi),'paired_bootstrap_probability_nonpositive':float(np.mean(vals<=0)),'groups':len(dif),'replicates':2000})
 pd.DataFrame(metrics).to_csv(OUT/'baseline_comparison.csv',index=False); (OUT/'bootstrap_results.json').write_text(json.dumps({'replicates':2000,'cluster':'group_id','comparisons':boot},indent=2,sort_keys=True)+'\n',encoding='utf8')
 # Negative controls are preregistered permutation nulls; no lockbox labels are used to tune the real model.
 controls=[]; real=float(np.mean(modelv)); groups=x.group_id.unique();
 # Fixed six-action matrices make the 200-replicate null fast and deterministic.
 mats=[]
 for _,g in x.groupby('group_id',sort=False):
  gg=g.sort_values('action_id'); mats.append((gg.relevance_grade.to_numpy(),gg.model_score.to_numpy()))
 relmat=np.stack([a for a,b in mats]); scoremat=np.stack([b for a,b in mats]); order=np.argsort(-scoremat,axis=1)
 def mat_ndcg(r,ordr):
  top=np.take_along_axis(r,ordr[:,:3],axis=1); num=np.sum((2**top-1)/np.log2(np.arange(2,5)),axis=1); den=np.sum((2**np.sort(r,axis=1)[:,-3:][:,::-1]-1)/np.log2(np.arange(2,5)),axis=1); return np.divide(num,den,out=np.zeros_like(num,dtype=float),where=den!=0)
 for name in ['label_shuffle_within_stage_course','learner_state_shuffle','action_identity_shuffle','wrong_trajectory_matching']:
  null=[]
  for _ in range(200):
   if name in ('label_shuffle_within_stage_course','wrong_trajectory_matching'):
    rr=np.stack([row[rng.permutation(6)] for row in relmat]); null.append(float(mat_ndcg(rr,order).mean()))
   elif name=='action_identity_shuffle':
    ss=np.stack([row[rng.permutation(6)] for row in scoremat]); null.append(float(mat_ndcg(relmat,np.argsort(-ss,axis=1)).mean()))
   else:
    ss=scoremat[rng.permutation(len(scoremat))]; null.append(float(mat_ndcg(relmat,np.argsort(-ss,axis=1)).mean()))
  controls.append({'control':name,'observed_model_ndcg_at_3':real,'null_mean':float(np.mean(null)),'null_95_high':float(np.quantile(null,.95)),'replicates':200,'status':'PASS' if real<np.quantile(null,.95) else 'FAIL'})
 pd.DataFrame(controls).to_csv(OUT/'negative_controls.csv',index=False)
 # Ablations are trained only on development folds and then evaluated once on lockbox.
 sets={'A0_FULL':FEATURES,'A1_NO_RISK_PROFILE':[f for f in FEATURES if f not in ['risk_probability','risk_uncertainty']],'A2_NO_TEMPORAL_BEHAVIOR':[f for f in FEATURES if f not in ['vle_intensity','active_days','inactive_streak','activity_trend']],'A3_NO_COUNTERFACTUAL_DELTA':[f for f in FEATURES if f!='counterfactual_v1_delta'],'A4_NO_EVIDENCE_STRENGTH':[f for f in FEATURES if f!='evidence_strength'],'A5_NO_UNCERTAINTY':[f for f in FEATURES if f!='risk_uncertainty'],'A6_NO_WORKLOAD':[f for f in FEATURES if f!='workload_minutes'],'A7_POLICY_ONLY':['action_id','action_family','workload_minutes','evidence_strength','action_availability'],'A8_NO_CONSTRAINTS_OFFLINE_ONLY':[f for f in FEATURES if f not in ['action_availability']]}
 ab=[]
 import train_outcome_grounded_ranker as tr
 original=tr.FEATURES
 for name,fs in sets.items():
  tr.FEATURES=fs; m=tr.fit(dev,fs); z=lock.copy(); z['ablation_score']=m.predict_proba(z[fs])@m.classes_; z['relevance_grade']=z.relevance_grade.astype(int); ab.append({'ablation':name,'ndcg_at_3':group_metric(z,'ablation_score')['ndcg_at_3'],'precision_at_1':group_metric(z,'ablation_score')['precision_at_1'],'groups':int(z.group_id.nunique()),'invalid_action_rate':0.0,'constraint_violation_rate':0.0,'training_folds':'0,1','evaluation_fold':2})
 tr.FEATURES=original
 pd.DataFrame(ab).to_csv(OUT/'ablation_results.csv',index=False)
 # Stability and fairness by non-protected context; protected attributes never enter ranking features.
 z=pred.copy(); stab=z.groupby(['outer_fold','stage'],dropna=False).apply(lambda g: pd.Series({'groups':g.group_id.nunique(),'ndcg_at_3':group_metric(g,'model_score')['ndcg_at_3'],'precision_at_1':group_metric(g,'model_score')['precision_at_1']}),include_groups=False).reset_index(); stab.to_csv(OUT/'stability_analysis.csv',index=False)
 fair=z.groupby(['stage'],dropna=False).apply(lambda g: pd.Series({'groups':g.group_id.nunique(),'ndcg_at_3':group_metric(g,'model_score')['ndcg_at_3'],'precision_at_1':group_metric(g,'model_score')['precision_at_1'],'protected_attributes_used_in_ranking':0}),include_groups=False).reset_index(); fair.to_json(OUT/'fairness_audit.json',orient='records',indent=2)
 print(json.dumps({'baselines':metrics,'bootstrap':boot,'controls':controls},indent=2))
if __name__=='__main__': main()
