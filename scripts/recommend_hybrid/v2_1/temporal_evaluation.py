"""Chronological 2013J+2014B to 2014J evaluation, frozen after protocol."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from full_evaluation import labels,matrix,groups,metrics,CACHE,OUT
def main():
 tr0=pd.read_parquet(CACHE/'temporal_train.parquet'); te0=pd.read_parquet(CACHE/'temporal_test.parquet');
 if len(te0)==0:
  d={'status':'COMPLETE_INSUFFICIENT_SUPPORT','train_presentations':sorted(tr0.presentation.unique().tolist()),'test_presentations':[],'overlapping_modules':0,'unseen_modules':[],'overall':None,'overlap_metrics':None,'ood_metrics':None,'coverage':0.0}
 else:
  tr,te=labels(tr0,te0); Xtr,Xte=matrix(tr,te); m=Ridge(alpha=1).fit(Xtr,tr.continuous_relevance); te['model_score']=np.asarray(m.predict(Xte)).reshape(-1); te['random_score']=np.random.default_rng(20260804).random(len(te)); te['workload_score']=-te.workload_minutes; te['policy_score']=te.action_family.map({'ASSESSMENT_COMPLETION':5,'VLE_ENGAGEMENT':4,'STUDY_REGULARITY':3,'QUIZ_OR_RETRIEVAL_PRACTICE':2,'CONTENT_REVIEW':1}).fillna(0); res=metrics(te,'model_score'); overlap=te[te.course.isin(set(tr.course))]; ood=te[~te.course.isin(set(tr.course))]; d={'status':'COMPLETE','train_presentations':sorted(tr.presentation.unique().tolist()),'test_presentations':sorted(te.presentation.unique().tolist()),'overlapping_modules':int(te.course.isin(set(tr.course)).sum()),'unseen_modules':sorted(set(te.course)-set(tr.course)),'overall':res,'overlap_metrics':metrics(overlap,'model_score') if len(overlap) else None,'ood_metrics':metrics(ood,'model_score') if len(ood) else None,'coverage':float(te.group_id.nunique()/max(len(te0.group_id.unique()),1))}
 (OUT/'TEMPORAL_RESULTS.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
