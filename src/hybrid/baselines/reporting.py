"""Non-training assembly of Phase 2 artifacts from audited study summaries."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import yaml

def generate_phase2_report(summaries: list[dict], output_root: Path, policy_path: Path):
    policy=yaml.safe_load(policy_path.read_text())
    if len(summaries)!=policy['expected_studies']:
        raise RuntimeError(f"Phase 2 HPO incomplete: {len(summaries)}/{policy['expected_studies']} studies fully complete")
    rows=[]
    for s in summaries:
        rows.append({'domain':s['domain'],'stage':s['stage'],'outer_fold':s['outer_fold'],'model_family':s['model_family'],
         'pooled_inner_oof_pr_auc':s['reproduced_pooled_pr_auc'],'pooled_inner_oof_roc_auc':s['reproduced_pooled_roc_auc'],'best_trial':s['best_trial_number'],
         'n_completed_trials':s['complete_trials'],'n_pruned_trials':s['pruned_trials'],'n_failed_trials':s['failed_trials'],
         'hpo_fit_time_seconds':s['hpo_fit_time_seconds'],'reproduction_fit_time_seconds':s['reproduction_fit_time_seconds'],
         'raw_predictor_count':s['raw_predictor_count'],'config_hash':s['config_hash'],'oof_sha256':s['oof_sha256']})
    frame=pd.DataFrame(rows); frame['selected_as_best_baseline']=False; ceiling=[]
    for key, group in frame.groupby(['domain','stage','outer_fold'], sort=False):
        ordered=group.sort_values(['pooled_inner_oof_pr_auc','pooled_inner_oof_roc_auc','hpo_fit_time_seconds','model_family'],ascending=[False,False,True,True],kind='mergesort')
        winner=ordered.iloc[0]; frame.loc[winner.name,'selected_as_best_baseline']=True
        ceiling.append({'domain':key[0],'stage':key[1],'outer_fold':int(key[2]),'winning_model_family':winner.model_family,'winning_config_hash':winner.config_hash,'inner_oof_pr_auc':winner.pooled_inner_oof_pr_auc,'inner_oof_roc_auc':winner.pooled_inner_oof_roc_auc,'evaluation_scope':policy['baseline_ceiling']['evaluation_scope'],'final_generalization':False})
    if len(ceiling)!=policy['expected_ceiling_entries']: raise RuntimeError('invalid ceiling cardinality')
    output_root.mkdir(parents=True,exist_ok=True); frame.to_csv(output_root/'baseline_leaderboard.csv',index=False)
    (output_root/'baseline_selected_configs.json').write_text(json.dumps(summaries,indent=2,sort_keys=True),encoding='utf-8')
    (output_root/'baseline_ceiling.json').write_text(json.dumps(ceiling,indent=2,sort_keys=True),encoding='utf-8')
    return frame, ceiling
