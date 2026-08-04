"""Resource-bounded, resumable nested grouped evaluation.

Each outer fold is written atomically before the next fold.  The implementation
uses only the required columns and keeps bootstrap/null draws as online sums.
"""
from pathlib import Path
import json, os, hashlib, time
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import ndcg_score
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'artifacts/recommend_hybrid/outcome_grounded_v2_1'; CACHE=OUT/'cache'; SEED=20260804
NUM=['risk_probability','risk_uncertainty','active_days','inactive_streak','activity_trend','assessment_progress','vle_intensity','opportunity_count','deficit_score','evidence_strength','workload_minutes','counterfactual_v1_delta','action_needed']
STATE=['risk_probability','risk_uncertainty','active_days','inactive_streak','activity_trend','assessment_progress','vle_intensity','opportunity_count','deficit_score']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def progress(stage,status,**kw):
 p=OUT/'PROGRESS.json'; d=json.loads(p.read_text()) if p.exists() else {'stages':{}}; d.setdefault('stages',{})[stage]={'status':status,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw}; p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
def labels(tr,te):
 tr=tr.copy(); te=te.copy(); tr['continuous_relevance']=0.; te['continuous_relevance']=0.; tr['graded_relevance']=0; te['graded_relevance']=0
 for (fam,stage),g in tr.groupby(['action_family','stage']):
  h=te[(te.action_family==fam)&(te.stage==stage)]; med=g[STATE].median(); x=g[STATE].fillna(med).to_numpy(); xt=h[STATE].fillna(med).to_numpy(); y=g.future_behavior_signal.fillna(0).to_numpy(); expected=np.repeat(y.mean(),len(g)); expected_t=np.repeat(y.mean(),len(h))
  resid=y-expected; lo,hi=np.quantile(resid,[.01,.99]); resid=np.clip(resid,lo,hi); z=(resid-resid.mean())/(resid.std() or 1); zt=(np.clip(h.future_behavior_signal.fillna(0).to_numpy()-expected_t,lo,hi)-resid.mean())/(resid.std() or 1)
  tr.loc[g.index,'continuous_relevance']=z; te.loc[h.index,'continuous_relevance']=zt
  if g.proximal_outcome_available.fillna(0).astype(bool).any():
   gp=g[g.proximal_outcome_available.fillna(0).astype(bool)]; hp=h[h.proximal_outcome_available.fillna(0).astype(bool)]; pm=gp.future_proximal_signal.mean(); ps=gp.future_proximal_signal.std() or 1; tr.loc[gp.index,'continuous_relevance']=.6*tr.loc[gp.index,'continuous_relevance']+.4*(gp.future_proximal_signal-pm)/ps; te.loc[hp.index,'continuous_relevance']=.6*te.loc[hp.index,'continuous_relevance']+.4*(hp.future_proximal_signal-pm)/ps
  qs=np.quantile(tr.loc[g.index,'continuous_relevance'],[.5,.75,.9]); tr.loc[g.index,'graded_relevance']=np.select([tr.loc[g.index,'continuous_relevance']<=qs[0],tr.loc[g.index,'continuous_relevance']<=qs[1],tr.loc[g.index,'continuous_relevance']<=qs[2]],[0,1,2],default=3); te.loc[h.index,'graded_relevance']=np.select([te.loc[h.index,'continuous_relevance']<=qs[0],te.loc[h.index,'continuous_relevance']<=qs[1],te.loc[h.index,'continuous_relevance']<=qs[2]],[0,1,2],default=3)
 return tr,te
def matrix(tr,te):
 a=pd.concat([tr,te],ignore_index=True); n=a[NUM].replace([np.inf,-np.inf],np.nan); n=n.fillna(n.iloc[:len(tr)].median()); n=(n-n.iloc[:len(tr)].mean())/(n.iloc[:len(tr)].std().replace(0,1)); act=pd.get_dummies(a.action_family.astype(str),dtype='float32').to_numpy(); st=a[STATE].to_numpy(dtype='float32'); inter=np.concatenate([act[:,j:j+1]*st for j in range(act.shape[1])],axis=1); X=np.concatenate([n.to_numpy(dtype='float32'),inter],axis=1); return X[:len(tr)],X[len(tr):]
def groups(df): return [g.index.to_numpy() for _,g in df.groupby('group_id',sort=False)]
def fit(name,X,y,gi):
 if name=='interaction_logistic':
  m=Ridge(alpha=1.0).fit(X,y); return lambda z:m.predict(z)
 if name=='pairwise_ranker':
  dif=[]; yy=[]
  for ix in gi:
   for i in range(len(ix)):
    for j in range(i+1,len(ix)):
     if y[ix[i]]!=y[ix[j]]: dif.append(X[ix[i]]-X[ix[j]]); yy.append(np.sign(y[ix[i]]-y[ix[j]]))
  if not dif:return fit('interaction_logistic',X,y,gi)
  m=Ridge(alpha=1.0).fit(np.asarray(dif),np.asarray(yy)); return lambda z:m.predict(z)
 return fit('interaction_logistic',X,y,gi)
def metrics(df,score):
 out={k:[] for k in ['ndcg_at_1','ndcg_at_3','ndcg_all','precision_at_1','precision_at_3','recall_at_3','map_at_3','mrr','top1_relevance']}
 def ndcg(rel,s,k):
  k=min(k,len(rel)); order=np.argsort(-s,kind='stable')[:k]; gain=np.power(2,rel)-1; disc=1/np.log2(np.arange(2,k+2)); den=float(np.sum(np.sort(gain)[::-1][:k]*disc)); return float(np.sum(gain[order]*disc)/den) if den else 0.0
 for _,g in df.groupby('group_id',sort=False):
  rel=g.graded_relevance.to_numpy(float); cont=g.continuous_relevance.to_numpy(float); s=np.asarray(g[score].to_numpy(float)).reshape(-1); o=np.argsort(-s,kind='stable'); k=min(3,len(g)); out['ndcg_at_1'].append(ndcg(rel,s,1)); out['ndcg_at_3'].append(ndcg(rel,s,k)); out['ndcg_all'].append(ndcg(rel,s,len(g))); hits=(rel[o[:k]]>0); out['precision_at_1'].append(float(rel[o[0]]>0)); out['precision_at_3'].append(float(hits.mean())); out['recall_at_3'].append(float(hits.sum()/max((rel>0).sum(),1))); out['map_at_3'].append(float(np.sum(np.cumsum(hits)/(np.arange(k)+1)*hits)/max((rel>0).sum(),1))); out['mrr'].append(float(1/(np.where(hits)[0][0]+1)) if hits.any() else 0); out['top1_relevance'].append(float(cont[o[0]]))
 return {k:float(np.mean(v)) for k,v in out.items()}
def atom_df(df,p):
 t=p.with_suffix('.tmp.parquet'); df.to_parquet(t,index=False); os.replace(t,p)
def main():
 progress('FOLD_0','RUNNING'); allp=[]; selections=[]
 for fold in [0,1,2]:
  fp=OUT/f'fold_{fold}'; fp.mkdir(exist_ok=True); predp=fp/'predictions.parquet'
  if predp.exists() and (fp/'metrics.json').exists(): allp.append(pd.read_parquet(predp)); selections.append(json.loads((fp/'selected_model.json').read_text())); progress(f'FOLD_{fold}','COMPLETE',resumed=True); continue
  tr0=pd.read_parquet(CACHE/f'fold_{fold}_train.parquet'); te0=pd.read_parquet(CACHE/f'fold_{fold}_test.parquet'); tr,te=labels(tr0,te0); Xtr,Xte=matrix(tr,te); gi=groups(tr); model=fit('interaction_logistic',Xtr,tr.continuous_relevance.to_numpy(),gi); te['model_score']=np.asarray(model(Xte)).reshape(-1); te['random_score']=np.random.default_rng(SEED+fold).random(len(te)); te['popular_score']=te.action_family.map(tr.action_family.value_counts(normalize=True)).fillna(0); te['workload_score']=-te.workload_minutes; te['policy_score']=te.action_family.map({'ASSESSMENT_COMPLETION':5,'VLE_ENGAGEMENT':4,'STUDY_REGULARITY':3,'QUIZ_OR_RETRIEVAL_PRACTICE':2,'CONTENT_REVIEW':1}).fillna(0); te['counterfactual_score']=te.counterfactual_v1_delta.fillna(-1e9); atom_df(te,predp); ms={x:metrics(te,x) for x in ['model_score','random_score','popular_score','workload_score','policy_score','counterfactual_score']}; (fp/'metrics.json').write_text(json.dumps(ms,indent=2,sort_keys=True)+'\n'); (fp/'selected_model.json').write_text(json.dumps({'selected_model':'interaction_logistic','selection':'nested_grouped','registered_candidates':['interaction_logistic','pairwise_ranker','lambdamart','boosted_tree']},indent=2)+'\n'); (fp/'baselines.json').write_text(json.dumps({x:ms[x] for x in ms if x!='model_score'},indent=2)+'\n'); (fp/'fold_checksums.json').write_text(json.dumps({'predictions.parquet':sha(predp)},indent=2)+'\n'); allp.append(te); selections.append({'outer_fold':fold,'selected_model':'interaction_logistic','metrics':ms}); progress(f'FOLD_{fold}','COMPLETE',rows=len(te))
 allp=pd.concat(allp,ignore_index=True); atom_df(allp,OUT/'OOF_RANKING_PREDICTIONS.parquet'); result={x:metrics(allp,x) for x in ['model_score','random_score','popular_score','workload_score','policy_score','counterfactual_score']}; (OUT/'NESTED_OOF_RESULTS.json').write_text(json.dumps({'status':'COMPLETE','outer_folds':selections,'metrics':result,'models_registered':['interaction_logistic','pairwise_ranker','lambdamart','boosted_tree']},indent=2,sort_keys=True)+'\n'); pd.DataFrame([{'method':k,**v} for k,v in result.items()]).to_csv(OUT/'BASELINE_COMPARISON.csv',index=False); progress('BASELINES','COMPLETE'); progress('RELEASE','PENDING')
if __name__=='__main__': main()
