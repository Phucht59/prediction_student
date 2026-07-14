"""Strict artifact validator shared by rehearsal and future full execution."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.evaluation.model_v3_execution import require_authorization_sidecar,strict_prediction_validation
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--authorization',required=True);p.add_argument('--authorization-sidecar',required=True);a=p.parse_args();root=Path(a.run_dir)
 manifest=json.loads((root/'run_manifest.json').read_text())
 if manifest.get('status')!='completed':raise RuntimeError('Partial run cannot be ranking eligible.')
 auth=json.loads(Path(a.authorization).read_text());side=json.loads(Path(a.authorization_sidecar).read_text());require_authorization_sidecar(auth,side)
 pred=pd.read_csv(root/'rehearsal_predictions.csv');metrics=pd.read_csv(root/'rehearsal_metrics.csv');keys=['model_family','track','outer_fold','training_seed'];expected={key:set(g.record_id) for key,g in pred.groupby(keys)}
 result=strict_prediction_validation(pred,metrics,expected);result['ranking_eligible']=False;result['scientific_eligibility']='rehearsal_only';(root/'strict_validation.json').write_text(json.dumps(result,indent=2));(root/'strict_validation.md').write_text('# Strict validation\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
 if result['overall_validation_status']!='valid':raise RuntimeError('Strict validation failed.')
if __name__=='__main__':main()
