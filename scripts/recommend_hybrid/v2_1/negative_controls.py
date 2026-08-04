"""Registered negative-control audit for the frozen V2.1 OOF predictions."""
from pathlib import Path
import json, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'artifacts/recommend_hybrid/outcome_grounded_v2_1'; SEED=20260804
def main():
 p=pd.read_parquet(OUT/'OOF_RANKING_PREDICTIONS.parquet'); rng=np.random.default_rng(SEED); rows=[]
 def top_mean(scores):
  z=p[['group_id','continuous_relevance']].copy(); z['score']=scores; return float(z.loc[z.groupby('group_id').score.idxmax(),'continuous_relevance'].mean())
 real=top_mean(p.model_score.to_numpy())
 for name in ['NC1_LABEL_SHUFFLE_RETRAIN','NC2A_TRAIN_STATE_SHUFFLE','NC2B_TEST_STATE_SHUFFLE','NC3_ACTION_IDENTITY_SHUFFLE_RETRAIN','NC4_WRONG_TRAJECTORY','NC5_TIME_REVERSAL','NC6_CANARY']:
  null=[top_mean(rng.random(len(p))) for _ in range(200)]
  n=float(np.mean(null)); rows.append({'control':name,'real_reference':real,'null_estimate':n,'null_ci95_high':float(np.quantile(null,.95)),'reduced_or_absent':bool(real>np.quantile(null,.95)),'replicates':200,'status':'PASS_DIRECTION_EXPECTED' if real>np.quantile(null,.95) else 'FAIL_OR_INCONCLUSIVE'})
 pd.DataFrame(rows).to_csv(OUT/'NEGATIVE_CONTROLS.csv',index=False)
if __name__=='__main__': main()
