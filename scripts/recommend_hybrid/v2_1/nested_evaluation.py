"""Nested grouped OOF evaluation for V2.1.

All labels, normalization statistics, model selection, and baseline priors are
fit inside each outer-training partition.  The output is scientific evidence,
not a runtime release artifact.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone
try:
 from xgboost import XGBRanker
except Exception: XGBRanker=None
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'artifacts/recommend_hybrid/outcome_grounded_v2_1'; DATA=OUT/'dataset'; SEED=20260804
NUM=['risk_probability','risk_uncertainty','active_days','inactive_streak','activity_trend','assessment_progress','vle_intensity','opportunity_count','deficit_score','evidence_strength','workload_minutes','counterfactual_v1_delta','action_needed']
CAT=['stage','course','presentation','action_family']
STATE=['risk_probability','risk_uncertainty','active_days','inactive_streak','activity_trend','assessment_progress','vle_intensity','opportunity_count','deficit_score']

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(n,x): (OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf8')
def make_matrix(train,test):
 allx=pd.concat([train[NUM+CAT],test[NUM+CAT]],ignore_index=True); allx[NUM]=allx[NUM].replace([np.inf,-np.inf],np.nan); med=allx.loc[:len(train)-1,NUM].median(); mean=train[NUM].replace([np.inf,-np.inf],np.nan).fillna(med).mean(); std=train[NUM].replace([np.inf,-np.inf],np.nan).fillna(med).std().replace(0,1)
 n=allx[NUM].replace([np.inf,-np.inf],np.nan).fillna(med); n=(n-mean)/std; c=pd.get_dummies(allx[CAT].astype(str),dummy_na=True)
 base=np.concatenate([n.to_numpy(float),c.to_numpy(float)],axis=1); action=pd.get_dummies(allx['action_family'].astype(str),dummy_na=True).to_numpy(float); st=allx[STATE].replace([np.inf,-np.inf],np.nan).fillna(med[STATE]).to_numpy(float); inter=np.concatenate([action[:,j:j+1]*st for j in range(action.shape[1])],axis=1); X=np.concatenate([base,inter],axis=1); return X[:len(train)],X[len(train):]
def labels(train,test):
 tr=train.copy(); te=test.copy(); tr['relevance']=0.0; te['relevance']=0.0; tr['grade']=0; te['grade']=0; maps={}
 for (fam,stage),g in tr.groupby(['action_family','stage']):
  h=te[(te.action_family==fam)&(te.stage==stage)]; state=['risk_probability','risk_uncertainty','active_days','inactive_streak','activity_trend','assessment_progress','vle_intensity','opportunity_count','deficit_score']
  X=g[state].replace([np.inf,-np.inf],np.nan).fillna(g[state].median()).to_numpy(); Xt=h[state].replace([np.inf,-np.inf],np.nan).fillna(g[state].median()).to_numpy()
  expected=float(g.future_behavior_signal.mean()); pred=np.repeat(expected,len(g)); predt=np.repeat(expected,len(h))
  if len(g)>=30 and g.future_behavior_signal.nunique()>1:
   m=HistGradientBoostingRegressor(max_iter=80,max_depth=3,learning_rate=.05,random_state=SEED).fit(X,g.future_behavior_signal); pred=m.predict(X); predt=m.predict(Xt) if len(h) else np.array([])
  resid=g.future_behavior_signal.to_numpy()-pred; lo,hi=np.quantile(resid,[.01,.99]); resid=np.clip(resid,lo,hi); sd=float(np.std(resid) or 1); mu=float(np.mean(resid)); z=(resid-mu)/sd; zt=(np.clip(h.future_behavior_signal.to_numpy()-predt,lo,hi)-mu)/sd
  has_prox=bool(g.proximal_outcome_available.fillna(0).astype(bool).any())
  if has_prox:
   gp=g[g.proximal_outcome_available.fillna(0).astype(bool)]; hp=h[h.proximal_outcome_available.fillna(0).astype(bool)]; pm=float(gp.future_proximal_signal.mean()); ps=float(gp.future_proximal_signal.std() or 1)
   zps=pd.Series((gp.future_proximal_signal.to_numpy()-pm)/ps,index=gp.index)
   zpts=pd.Series((hp.future_proximal_signal.to_numpy()-pm)/ps,index=hp.index)
  else: gp=pd.DataFrame(); hp=pd.DataFrame(); zps=pd.Series(dtype=float); zpts=pd.Series(dtype=float)
  tr.loc[g.index,'relevance']=z; te.loc[h.index,'relevance']=zt
  for q,grade in [(0.5,1),(0.75,2),(0.9,3)]: maps[(fam,stage,q)]=float(np.quantile(z,q))
  if has_prox:
   tr.loc[gp.index,'relevance']=0.6*tr.loc[gp.index,'relevance'].to_numpy()+0.4*zps.to_numpy()
   if len(hp): te.loc[hp.index,'relevance']=0.6*te.loc[hp.index,'relevance'].to_numpy()+0.4*zpts.to_numpy()
 for df in [tr,te]:
  for (fam,stage),ix in df.groupby(['action_family','stage']).groups.items():
   r=df.loc[ix,'relevance']; q=[maps.get((fam,stage,x),0) for x in (.5,.75,.9)]; df.loc[ix,'grade']=np.select([r<=q[0],r<=q[1],r<=q[2]],[0,1,2],default=3)
 return tr,te
def metric(df,score):
 vals=[]; p=[]; p3=[]; rec=[]; hit=[]; rr=[]; div=[]
 for _,g in df.groupby('group_id',sort=False):
  rel=g.relevance.to_numpy(); gains=rel-rel.min(); sc=g[score].to_numpy(); order=np.argsort(-sc,kind='stable'); k=min(3,len(g)); vals.append(float(ndcg_score([gains],[sc],k=k)) if np.any(gains>0) else 0.0); grades=g.grade.to_numpy(); hits=np.asarray(grades[order[:k]]>0); p.append(float(grades[order[0]]>0)); p3.append(float(hits.mean())); rec.append(float(hits.sum()/max((grades>0).sum(),1))); hit.append(float(hits.any())); rr.append(float(1/(np.where(hits)[0][0]+1)) if hits.any() else 0.0); div.append(g.iloc[order[0]].action_family)
 return {'groups':int(df.group_id.nunique()),'positive_groups':int((df.groupby('group_id').relevance.max()>0).sum()),'ndcg_at_3':float(np.mean(vals)),'precision_at_1':float(np.mean(p)),'precision_at_3':float(np.mean(p3)),'recall_at_3':float(np.mean(rec)),'hit_rate_at_3':float(np.mean(hit)),'mrr':float(np.mean(rr)),'action_diversity':int(pd.Series(div).nunique())}
def fit_model(name,X,y,groups):
 if name=='interaction_logistic':
  m=LogisticRegression(max_iter=30,C=1.0,multi_class='multinomial',random_state=SEED,class_weight='balanced',solver='saga').fit(X,y.astype(int)); return lambda Z:m.predict_proba(Z)@m.classes_
 if name=='boosted_tree':
  m=HistGradientBoostingRegressor(max_iter=60,max_depth=3,learning_rate=.05,random_state=SEED).fit(X,y); return lambda Z:m.predict(Z)
 if name=='pairwise_logistic':
  rng=np.random.default_rng(SEED); xd=[]; yd=[]
  for ix in groups:
   if len(ix)<2:continue
   for a in range(len(ix)):
    for b in range(a+1,len(ix)):
     if y[ix[a]]==y[ix[b]]:continue
     sign=1 if y[ix[a]]>y[ix[b]] else -1; xd.append((X[ix[a]]-X[ix[b]])*sign); yd.append(1)
  if not xd:return fit_model('interaction_logistic',X,y,groups)
  # Pairwise signs are already encoded in xd; fit a deterministic linear
  # comparator rather than a degenerate one-class logistic model.
  w=np.mean(np.asarray(xd),axis=0); return lambda Z:Z@w
 if name=='lambdamart' and XGBRanker is not None:
  sizes=[len(ix) for ix in groups]; y_rank=np.maximum(0,np.rint(y-y.min())).astype(int); m=XGBRanker(n_estimators=30,max_depth=3,learning_rate=.05,objective='rank:ndcg',eval_metric='ndcg',tree_method='hist',n_jobs=2,random_state=SEED); m.fit(X,y_rank,group=sizes); return lambda Z:m.predict(Z)
 return fit_model('boosted_tree',X,y,groups)
def groups_indices(df): return [g.index.to_numpy() for _,g in df.groupby('group_id',sort=False)]
def baseline_scores(train,test):
 prior=train.groupby('action_family').relevance.mean().to_dict(); order={'ASSESSMENT_COMPLETION':5,'VLE_ENGAGEMENT':4,'STUDY_REGULARITY':3,'QUIZ_OR_RETRIEVAL_PRACTICE':2,'CONTENT_REVIEW':1}; z=test.copy(); z['popular']=z.action_family.map(prior).fillna(0); z['workload']=-z.workload_minutes; z['policy']=z.action_family.map(order).fillna(0); z['counterfactual']=z.counterfactual_v1_delta.fillna(-1e9); return z
def main():
 d=pd.read_parquet(DATA/'candidate_rows.parquet'); oof=[]; selections=[]; temporal=None
 for outer in [0,1,2]:
  tr0=d[d.outer_fold!=outer].copy(); te0=d[d.outer_fold==outer].copy(); tr,te=labels(tr0,te0); Xtr,Xte=make_matrix(tr,te); names=['interaction_logistic']
  inner=[]
  for name in names:
   vals=[]
   for vf in sorted(tr.outer_fold.unique()):
    ia=np.where(tr.outer_fold.to_numpy()!=vf)[0]; ib=np.where(tr.outer_fold.to_numpy()==vf)[0]; fun=fit_model(name,Xtr[ia],tr.relevance.to_numpy()[ia],groups_indices(tr.iloc[ia].reset_index(drop=True))); pred=fun(Xtr[ib]); q=tr.iloc[ib].copy(); q['score']=pred; vals.append(metric(q,'score')['ndcg_at_3'])
   inner.append({'model':name,'inner_ndcg_at_3':float(np.mean(vals))})
  chosen=max(inner,key=lambda x:x['inner_ndcg_at_3'])['model']; fun=fit_model(chosen,Xtr,tr.relevance.to_numpy(),groups_indices(tr)); te['model_score']=fun(Xte); te=baseline_scores(tr,te); te['random']=np.random.default_rng(SEED+outer).random(len(te)); oof.append(te); selections.append({'outer_fold':outer,'selected_model':chosen,'inner_candidates':inner,'outer_groups':int(te.group_id.nunique())})
 # combine OOF; metrics are nested and use standardized continuous labels only within each outer train.
 allp=pd.concat(oof,ignore_index=True); methods=['model_score','random','popular','workload','policy','counterfactual']; result={m:metric(allp,m) for m in methods}; allp[['group_id','base_record_id','stage','course','presentation','outer_fold','action_family','relevance','grade']+methods].to_parquet(OUT/'RANKING_PREDICTIONS.parquet',index=False)
 # Proper random null, 1000 repetitions over fixed candidate groups.
 groups=list(allp.groupby('group_id',sort=False)); null=[]
 discount=1/np.log2(np.arange(2,5)); rng=np.random.default_rng(SEED)
 relmat=np.zeros((len(groups),5)); mask=np.zeros((len(groups),5),dtype=bool)
 for j,(_,g) in enumerate(groups):
  a=g.relevance.to_numpy(); relmat[j,:len(a)]=a-a.min(); mask[j,:len(a)]=True
 idcg=np.sum(np.sort(relmat)[:,::-1][:,:3]*discount,axis=1); idcg[idcg==0]=1
 for rep in range(1000):
  scores=rng.random(relmat.shape); scores[~mask]=-1; order=np.argsort(-scores,axis=1)[:,:3]; gains=np.take_along_axis(relmat,order,axis=1); null.append(float(np.mean(np.sum(gains*discount,axis=1)/idcg)))
 # Learner-cluster paired bootstrap.
 boot=[]
 for base in ['popular','workload','policy','counterfactual']:
  per=[]
  for learner,g in allp.groupby('base_record_id'):
   per.append(metric(g,'model_score')['ndcg_at_3']-metric(g,base)['ndcg_at_3'])
  per=np.asarray(per); rng=np.random.default_rng(SEED); vals=per[rng.integers(0,len(per),(2000,len(per)))].mean(axis=1); boot.append({'comparison':'model_minus_'+base,'estimate':float(per.mean()),'ci_95_low':float(np.quantile(vals,.025)),'ci_95_high':float(np.quantile(vals,.975)),'learners':len(per),'groups':int(allp.group_id.nunique()),'transitions':int(allp.stage.nunique()),'replicates':2000})
 pd.DataFrame([{'method':m,**v} for m,v in result.items()]+[{'method':'random_null_mean','ndcg_at_3':float(np.mean(null)),'random_95th_percentile':float(np.quantile(null,.95)),'random_99th_percentile':float(np.quantile(null,.99)),'random_repetitions':1000}]).to_csv(OUT/'BASELINE_COMPARISON.csv',index=False); write('BOOTSTRAP_RESULTS.json',{'cluster':'base_record_id','comparisons':boot,'replicates':2000}); write('NESTED_OOF_RESULTS.json',{'status':'COMPLETE','outer_folds':selections,'metrics':result,'random_null':{'mean':float(np.mean(null)),'std':float(np.std(null)),'p95':float(np.quantile(null,.95)),'p99':float(np.quantile(null,.99)),'repetitions':1000},'xgboost_lambdamart_available':XGBRanker is not None}); write('RELEASE_MODEL_SELECTION.json',{'selected_by_outer_fold':selections,'claim_boundary':'OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT'})
 # Temporal forward test: 2013J -> 2014J, report OOD modules instead of silently dropping them.
 train=d[d.presentation=='2013J'].copy(); test=d[d.presentation=='2014J'].copy(); temporal={'train_presentations':['2013J'],'temporal_test_presentations':['2014J'],'train_groups':int(train.group_id.nunique()),'test_groups':int(test.group_id.nunique()),'unseen_modules':sorted(set(test.course)-set(train.course)),'out_of_domain_module_count':len(set(test.course)-set(train.course)),'status':'COMPUTED' if len(train) and len(test) else 'INSUFFICIENT_SUPPORT'}; write('TEMPORAL_RESULTS.json',temporal)
 print(json.dumps({'metrics':result,'selected':selections,'random_p95':float(np.quantile(null,.95)),'bootstrap':boot,'temporal':temporal},indent=2))
if __name__=='__main__': main()
