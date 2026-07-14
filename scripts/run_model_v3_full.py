"""V3.2 full-run entry point. It never bypasses the execution authorization gate."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import ROOT_DIR
from src.evaluation.model_v3_2 import git_tree_clean,validate_authorization

def load(path):return json.loads(Path(path).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--contract-dir',required=True);p.add_argument('--authorization',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();d=Path(a.contract_dir);source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT_DIR,text=True).strip()
 auth=load(a.authorization);expected=load(d/'expected_job_contract.json');neural=load(d/'selection_study_contract.json');b0=load(d/'b0_selection_contract.json');inner=load(d/'shared_inner_split_manifest.json');feature=load(d/'feature_contracts.json');target=load(d/'target_contract.json');search=load(d/'search_contract.json');acceptance=load(d/'acceptance_criteria.json');m4=load(d/'fixed_m4_config_contract.json')
 validate_authorization(auth,expected,neural,b0,inner,feature,target,search,acceptance,m4,source_commit=source,tree_clean=git_tree_clean(str(ROOT_DIR)))
 if not a.execute: print(json.dumps({'preflight':'valid','execute':False,'message':'No compute started.'}));return
 raise NotImplementedError('Full compute deliberately requires a separately reviewed execution implementation and authorization.')
if __name__=='__main__':main()
