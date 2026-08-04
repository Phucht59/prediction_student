"""Fast aggregation for completed corrected outer-fold predictions."""
from pathlib import Path
import json, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/recommend_hybrid/outcome_grounded_v2_1'; FINAL=OUT/'final_oof'
def metric(df,col):
 vals=[]; p1=[]; p3=[]; rec=[]; maps=[]; mrr=[]; top=[]; topa=[]
 d=1/np.log2(np.arange(2,5))
 for _,g in df.groupby('group_id',sort=False):
  rel=g.graded_relevance.to_numpy(float); cont=g.continuous_relevance.to_numpy(float); s=np.asarray(g[col],float); o=np.argsort(-s,kind='stable'); k=min(3,len(g)); gain=2**rel-1; den=np.sum(np.sort(gain)[::-1][:k]*d[:k]); hits=rel[o[:k]]>0
  vals.append(float(np.sum(gain[o[:k]]*d[:k])/den) if den else 0); p1.append(float(rel[o[0]]>0)); p3.append(float(hits.mean())); rec.append(float(hits.sum()/max((rel>0).sum(),1))); maps.append(float(np.sum(np.cumsum(hits)/(np.arange(k)+1)*hits)/max((rel>0).sum(),1))); mrr.append(float(1/(np.where(hits)[0][0]+1)) if hits.any() else 0); top.append(float(cont[o[0]])); topa.append(str(g.iloc[o[0]].action_family))
 return {'ndcg_at_3':float(np.mean(vals)),'precision_at_1':float(np.mean(p1)),'precision_at_3':float(np.mean(p3)),'recall_at_3':float(np.mean(rec)),'map_at_3':float(np.mean(maps)),'mrr':float(np.mean(mrr)),'top1_relevance':float(np.mean(top)),'groups':int(df.group_id.nunique()),'learners':int(df.base_record_id.nunique()),'action_diversity':len(set(topa)),'top_action_concentration':float(pd.Series(topa).value_counts(normalize=True).max())}
def main():
 p=pd.read_parquet(FINAL/'OOF_RANKING_PREDICTIONS.parquet'); methods=['model_score','random_debug_score','popular_score','workload_score','policy_score','counterfactual_score']; metrics={m:metric(p,m) for m in methods}; groups=list(p.groupby('group_id',sort=False)); maxa=max(len(g) for _,g in groups); rel=np.zeros((len(groups),maxa),np.float32); mask=np.zeros((len(groups),maxa),bool)
 for i,(_,g) in enumerate(groups): a=g.graded_relevance.to_numpy(float); rel[i,:len(a)]=a-a.min(); mask[i,:len(a)]=1
 k=min(3,maxa); disc=1/np.log2(np.arange(2,k+2)); idcg=np.sum(np.sort(2**rel-1,axis=1)[:,::-1][:,:k]*disc,axis=1); idcg[idcg==0]=1; rng=np.random.default_rng(20260804); null=[]
 for _ in range(1000):
  s=rng.random(rel.shape); s[~mask]=-1; o=np.argsort(-s,axis=1)[:,:k]; gains=np.take_along_axis(2**rel-1,o,axis=1); null.append(float(np.mean(np.sum(gains*disc,axis=1)/idcg)))
 random={'repetitions':1000,'mean':float(np.mean(null)),'p95':float(np.quantile(null,.95)),'p99':float(np.quantile(null,.99))}; result={'status':'CORRECTED_NESTED_OOF_COMPLETE','claim_boundary':'OFFLINE_PREDICTIVE_RELEVANCE_NOT_CAUSAL_EFFECT','models_actually_evaluated':['interaction_logistic','pairwise_logistic','lambdamart','boosted_tree'],'metrics':metrics,'random_null':random,'learners':int(p.base_record_id.nunique()),'groups':int(p.group_id.nunique()),'candidate_rows':len(p)}; (FINAL/'NESTED_OOF_RESULTS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); pd.DataFrame([{'method':m,**v} for m,v in metrics.items()]+[{'method':'random_null_distribution',**random}]).to_csv(FINAL/'BASELINE_COMPARISON.csv',index=False)
if __name__=='__main__': main()
