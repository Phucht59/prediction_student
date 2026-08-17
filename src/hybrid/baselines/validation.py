"""Artifact-only validation; intentionally imports no estimator or objective code."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

def audit_summary(summary_path: Path, oof_path: Path, tolerance: float=1e-10):
    result={'summary_exists':summary_path.is_file(),'oof_exists':oof_path.is_file(),'oof_hash_matches':False,'reproduced_pr_auc_matches':False,'reproduced_roc_auc_matches':False,'valid':False}
    if not result['summary_exists'] or not result['oof_exists']: return result
    summary=json.loads(summary_path.read_text()); digest=hashlib.sha256(oof_path.read_bytes()).hexdigest(); result['oof_hash_matches']=digest==summary.get('oof_sha256')
    frame=pd.read_parquet(oof_path); pr=average_precision_score(frame.target,frame.ranking_score); roc=roc_auc_score(frame.target,frame.ranking_score)
    result['reproduced_pr_auc_matches']=abs(pr-summary['reproduced_pooled_pr_auc'])<=tolerance; result['reproduced_roc_auc_matches']=abs(roc-summary['reproduced_pooled_roc_auc'])<=tolerance
    result['valid']=result['oof_hash_matches'] and result['reproduced_pr_auc_matches'] and result['reproduced_roc_auc_matches']
    return result
