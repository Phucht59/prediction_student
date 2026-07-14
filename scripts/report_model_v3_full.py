"""Refuse ranking until a strict validator has marked the run eligible."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--validation',required=True);a=p.parse_args();v=json.loads(Path(a.validation).read_text())
 if v.get('overall_validation_status')!='valid' or not v.get('ranking_eligible'):raise RuntimeError('Partial or invalid V3 runs cannot produce rankings.')
 print('Validated reporting entry point ready.')
if __name__=='__main__':main()
