"""Report entry point; rehearsal reports are explicitly non-scientific."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.evaluation.model_v3_execution import report_rehearsal
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();root=Path(a.run_dir);v=json.loads((root/'strict_validation.json').read_text())
 if v.get('overall_validation_status')!='valid':raise RuntimeError('Invalid execution cannot report.')
 report_rehearsal(pd.read_csv(root/'rehearsal_predictions.csv'),pd.read_csv(root/'rehearsal_metrics.csv'),root/'rehearsal_report')
if __name__=='__main__':main()
