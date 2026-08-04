"""Fixed-name V2.1 ablation ledger; model reruns are intentionally separate."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'artifacts/recommend_hybrid/outcome_grounded_v2_1'
def main():
 base=pd.read_csv(OUT/'BASELINE_COMPARISON.csv') if (OUT/'BASELINE_COMPARISON.csv').exists() else pd.DataFrame()
 names=['full','no_risk','no_behavior','no_opportunity','no_deficit','no_cf','no_interactions','no_workload','no_constraints','action_prior']
 rows=[]
 for n in names: rows.append({'ablation':n,'status':'REGISTERED_REQUIRES_NESTED_RERUN' if n!='full' else 'FULL_SYSTEM','ndcg_at_3':float(base.loc[base.method=='model_score','ndcg_at_3'].iloc[0]) if n=='full' and 'ndcg_at_3' in base and (base.method=='model_score').any() else None})
 pd.DataFrame(rows).to_csv(OUT/'ABLATION_RESULTS.csv',index=False)
if __name__=='__main__': main()
