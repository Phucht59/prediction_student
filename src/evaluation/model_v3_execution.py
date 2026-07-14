"""Shared contract-driven V3 execution helpers for rehearsal and future full runs."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from src.evaluation.protocol import canonical_json
from src.evaluation.metrics import classification_metrics
from src.evaluation.model_v3_protocol import checksum, regression_metric_summary

def semantic_checksum(payload: dict[str,Any]) -> str:
    body={k:v for k,v in payload.items() if k!='semantic_checksum'}
    return hashlib.sha256(canonical_json(body).encode()).hexdigest()

def require_semantic_checksum(payload: dict[str,Any]) -> None:
    if 'semantic_checksum' not in payload or payload['semantic_checksum'] != semantic_checksum(payload):
        raise ValueError('Stored semantic checksum does not match canonical contract content.')

def authorization_checksum(payload: dict[str,Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()

def require_authorization_sidecar(payload:dict[str,Any], sidecar:dict[str,Any]) -> None:
    if sidecar.get('authorization_checksum') != authorization_checksum(payload):
        raise ValueError('Authorization sidecar checksum mismatch.')

def atomic_json(path:Path,payload:Any)->None:
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8');tmp.replace(path)

def build_outer_execution_contract(run_id:str, jobs:list[dict], selection_checksum:str, source_commit:str)->dict:
    result={'contract_version':'model_v3_3','run_id':run_id,'created_after_selection_before_outer_refit':True,'selection_evidence_checksum':selection_checksum,'source_commit':source_commit,'jobs':jobs}
    result['semantic_checksum']=semantic_checksum(result);return result

def strict_prediction_validation(predictions:pd.DataFrame, metrics:pd.DataFrame, expected_by_job:dict[tuple,set[str]])->dict:
    out={'missing_jobs':0,'duplicate_rows':0,'coverage_errors':0,'probability_errors':0,'argmax_errors':0,'metric_errors':0}
    keys=['model_family','track','outer_fold','training_seed']
    out['duplicate_rows']=int(predictions.duplicated(keys+['record_id']).sum())
    for key,g in predictions.groupby(keys):
        if set(g.record_id)!=expected_by_job.get(key,set()):out['coverage_errors']+=1
        prob=g[['probability_low','probability_medium','probability_high']].to_numpy(float)
        out['probability_errors']+=int(not(np.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all() and np.max(np.abs(prob.sum(1)-1))<=1e-6))
        out['argmax_errors']+=int(not np.array_equal(prob.argmax(1),g.predicted_label.to_numpy(int)))
        stored=metrics
        match=stored[(stored.model_family==key[0])&(stored.track==key[1])&(stored.outer_fold==key[2])&(stored.training_seed==key[3])]
        if len(match)!=1:out['metric_errors']+=1;continue
        rec=classification_metrics(g.true_label,g.predicted_label,prob)
        out['metric_errors']+=int(not np.isclose(rec['macro_f1'],match.iloc[0].macro_f1))
    out['overall_validation_status']='valid' if not any(out.values()) else 'invalid';return out

def report_rehearsal(predictions:pd.DataFrame, metrics:pd.DataFrame, out:Path)->None:
    out.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(out/'ranking_by_track.csv',index=False)
    summary=metrics.groupby(['track','model_family']).macro_f1.mean().reset_index()
    lines=['# Rehearsal ranking (non-scientific)','', '| track | model_family | macro_f1 |','|---|---|---:|']
    lines += [f'| {r.track} | {r.model_family} | {r.macro_f1:.6f} |' for r in summary.itertuples(index=False)]
    (out/'ranking_by_track.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    regression=metrics[[c for c in metrics.columns if c in {'model_family','track','outer_fold','training_seed','mae_raw','rmse_raw','r2_raw'}]].copy();regression.to_csv(out/'regression_metrics_fold_seed.csv',index=False)
    (out/'model_v3_scientific_decision.md').write_text('# Rehearsal only\n\nNo model ranking or scientific decision is permitted from this execution rehearsal.\n',encoding='utf-8')
