"""Learner-cluster paired bootstrap with online/vectorized resampling."""
from pathlib import Path
import json,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'artifacts/recommend_hybrid/outcome_grounded_v2_1'; R=2000
def ndcg(g,s):
 rel=g.graded_relevance.to_numpy(float); x=np.asarray(g[s]).reshape(-1); k=min(3,len(g)); o=np.argsort(-x)[:k]; gain=2**rel-1; d=1/np.log2(np.arange(2,k+2)); den=np.sum(np.sort(gain)[::-1][:k]*d); return float(np.sum(gain[o]*d)/den) if den else 0.0
def main():
 p=pd.read_parquet(OUT/'OOF_RANKING_PREDICTIONS.parquet'); rows=[]
 for learner,g in p.groupby('base_record_id'):
  vals={x:ndcg(g,x) for x in ['model_score','random_score','popular_score','workload_score','policy_score','counterfactual_score']}; rows.append(vals)
 a=pd.DataFrame(rows); rng=np.random.default_rng(20260804); idx=rng.integers(0,len(a),(R,len(a))); out=[]
 for b in ['random_score','popular_score','workload_score','policy_score','counterfactual_score']:
  diff=a.model_score-a[b]; boots=diff.to_numpy()[idx].mean(axis=1); out.append({'comparison':'V2.1_minus_'+b,'estimate':float(diff.mean()),'ci95_low':float(np.quantile(boots,.025)),'ci95_high':float(np.quantile(boots,.975)),'probability_difference_le_zero':float(np.mean(boots<=0)),'learners':len(a),'groups':int(p.group_id.nunique()),'replicates':R})
 (OUT/'BOOTSTRAP_RESULTS.json').write_text(json.dumps({'status':'COMPLETE','cluster':'base_record_id','comparisons':out},indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
