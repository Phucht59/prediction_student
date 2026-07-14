"""Strict V3.2 validator for a materialized full run; partial runs cannot rank."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.evaluation.model_v3_2 import validate_pooled_oof_exact

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--expected-ids',required=True);a=p.parse_args();root=Path(a.run_dir)
 manifest=json.loads((root/'run_manifest.json').read_text())
 if manifest.get('status')!='completed':raise RuntimeError('Partial/failed runs cannot be validated for ranking.')
 pred=pd.read_csv(root/'outer_validation_predictions.csv');expected=set(json.loads(Path(a.expected_ids).read_text()))
 validate_pooled_oof_exact(pred,expected)
 print(json.dumps({'overall_validation_status':'valid','ranking_eligible':True}))
if __name__=='__main__':main()
