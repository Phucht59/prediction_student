"""Train the preregistered V2 ranker on development folds only."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'artifacts/recommend_hybrid/outcome_grounded'; DATA=OUT/'dataset'; SEED=20260804
NUM=['risk_probability','risk_uncertainty','vle_intensity','active_days','inactive_streak','activity_trend','assessment_completion','assessment_availability','studied_credits','previous_attempts','cutoff_day','workload_minutes','evidence_strength','action_availability','counterfactual_v1_delta']
CAT=['stage','course','presentation','action_id','action_family']
FEATURES=NUM+CAT

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def model(feature_list=None):
 fs=FEATURES if feature_list is None else feature_list
 nums=[x for x in NUM if x in fs]; cats=[x for x in CAT if x in fs]
 pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore'))]),cats)])
 return Pipeline([('pre',pre),('clf',LogisticRegression(max_iter=1000,C=1.0,class_weight='balanced',multi_class='multinomial',random_state=SEED,n_jobs=1))])
def fit(train,feature_list=None):
 fs=FEATURES if feature_list is None else feature_list
 m=model(fs); m.fit(train[fs],train.relevance_grade.astype(int)); return m
def ndcg(values,order,k=3):
 x=np.asarray(values)[np.asarray(order)[:k]]; den=np.sum((2**np.sort(np.asarray(values))[-k:][::-1]-1)/np.log2(np.arange(2,k+2))); num=np.sum((2**x-1)/np.log2(np.arange(2,len(x)+2))); return float(num/den) if den else 0.0
def group_metric(d,score_col,seed=SEED):
 vals=[]; p=[]; ap=[]
 for _,g in d.groupby('group_id',sort=False):
  g=g.reset_index(drop=True); order=np.argsort(-g[score_col].to_numpy(),kind='stable'); rel=g.relevance_grade.to_numpy(); vals.append(ndcg(rel,order)); p.append(float(rel[order[0]]>0)); hits=0; prec=[]
  for j,ix in enumerate(order[:3],1):
   if rel[ix]>0: hits+=1; prec.append(hits/j)
  ap.append(float(np.mean(prec)) if prec else 0.0)
 return {'groups':len(vals),'ndcg_at_3':float(np.mean(vals)) if vals else 0.0,'precision_at_1':float(np.mean(p)) if p else 0.0,'map_at_3':float(np.mean(ap)) if ap else 0.0,'coverage':float(np.mean(d[score_col].notna())) if len(d) else 0.0,'action_diversity':int(d.loc[d.groupby('group_id')[score_col].idxmax(),'action_id'].nunique()) if len(d) else 0}
def score_model(m,d): return m.predict_proba(d[FEATURES])@m.classes_
def base_scores(d,train):
 popular=train.groupby('action_id').relevance_grade.mean().to_dict(); policy={a:len(policy_order)-i for i,a in enumerate(policy_order)}
 d=d.copy(); d['random_score']=np.random.default_rng(SEED).random(len(d)); d['popular_score']=d.action_id.map(popular).fillna(0); d['workload_score']=-d.workload_minutes; d['policy_score']=d.action_id.map(policy).fillna(0); d['counterfactual_score']=d.counterfactual_v1_delta.fillna(-1e9); return d
policy_order=['ASSESSMENT_COMPLETION','VLE_ENGAGEMENT','STUDY_SCHEDULE','TARGETED_PRACTICE','RETRIEVAL_PRACTICE','LEARNING_CONSOLIDATION']
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['development','lockbox'],required=True); args=ap.parse_args()
 d=pd.read_parquet(DATA/'candidate_rows.parquet'); d=d[d.relevance_grade.notna()].copy(); dev=d[d.outer_fold.isin([0,1])].copy()
 if args.phase=='development':
  result=[]
  for train_fold,test_fold in [(0,1),(1,0)]:
   m=fit(dev[dev.outer_fold==train_fold]); test=base_scores(dev[dev.outer_fold==test_fold],dev[dev.outer_fold==train_fold]); test['model_score']=score_model(m,test); result.append({'train_fold':train_fold,'test_fold':test_fold,'metrics':{k:group_metric(test,k) for k in ['model_score','random_score','popular_score','workload_score','policy_score','counterfactual_score']}})
  (OUT/'development_results.json').write_text(json.dumps({'status':'DEVELOPMENT_COMPLETE','model':'pointwise_logistic','fold_results':result},indent=2,sort_keys=True)+'\n',encoding='utf8'); m=fit(dev); (OUT/'models').mkdir(parents=True,exist_ok=True); joblib.dump(m,OUT/'models/selected_model.joblib'); (OUT/'models/feature_schema.json').write_text(json.dumps({'features':FEATURES,'numeric':NUM,'categorical':CAT},indent=2)+'\n',encoding='utf8'); (OUT/'models/label_schema.json').write_text((DATA/'label_schema.json').read_text(encoding='utf8'),encoding='utf8'); print(json.dumps({'phase':'development','folds':2},indent=2)); return
 # lockbox is opened exactly once, after development result exists and this commit is fixed.
 if not (OUT/'development_results.json').is_file(): raise SystemExit('development must complete before lockbox')
 reg=json.loads((OUT/'LOCKBOX_REGISTRY.json').read_text());
 if reg.get('status')!='NOT_OPENED': raise SystemExit('lockbox already opened')
 m=joblib.load(OUT/'models/selected_model.joblib'); test=base_scores(d[d.outer_fold==2].copy(),dev); test['model_score']=score_model(m,test); metrics={k:group_metric(test,k) for k in ['model_score','random_score','popular_score','workload_score','policy_score','counterfactual_score']}; reg.update({'status':'OPENED_AND_EVALUATED_ONCE','opened_at':pd.Timestamp.utcnow().isoformat(),'prior_executions':1,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'protocol_sha256':json.loads((OUT/'INPUT_AUTHORITY.json').read_text())['protocol_sha256'],'lockbox_fold':2,'groups':int(test.group_id.nunique()),'candidate_rows':len(test)}); (OUT/'LOCKBOX_REGISTRY.json').write_text(json.dumps(reg,indent=2,sort_keys=True)+'\n',encoding='utf8'); (OUT/'lockbox_results.json').write_text(json.dumps({'status':'LOCKBOX_EVALUATED_ONCE','metrics':metrics,'groups':int(test.group_id.nunique()),'candidate_rows':len(test)},indent=2,sort_keys=True)+'\n',encoding='utf8'); test[['group_id','base_record_id','stage','course','presentation','action_id','outer_fold','relevance_grade','model_score','random_score','popular_score','workload_score','policy_score','counterfactual_score']].to_parquet(OUT/'ranking_predictions.parquet',index=False); print(json.dumps({'phase':'lockbox','metrics':metrics},indent=2))
if __name__=='__main__': main()
