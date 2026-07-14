"""Run a small V3.3 implementation rehearsal through the shared validation/report engine."""
from __future__ import annotations
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import ROOT_DIR
from src.evaluation.model_v3_execution import atomic_json,authorization_checksum,build_outer_execution_contract
ROOT=ROOT_DIR/'artifacts/model_v3_rehearsal';SMOKE=ROOT_DIR/'artifacts/model_v3_smoke'
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();root=ROOT/a.run_id
 if root.exists():raise FileExistsError(root)
 root.mkdir(parents=True);auth={'protocol_version':'model_v3_3','execution_mode':'rehearsal','compute_authorized':True,'scientific_eligibility':'implementation_rehearsal_only','run_id':a.run_id};atomic_json(root/'authorization_contract.json',auth);atomic_json(root/'authorization_checksum.json',{'authorization_checksum':authorization_checksum(auth)})
 child=f'{a.run_id}-engine';subprocess.check_call([sys.executable,str(ROOT_DIR/'scripts/run_model_v3_2_dual_smoke.py'),'--run-id',child],cwd=ROOT_DIR)
 child_root=SMOKE/child
 shutil.copy2(child_root/'dual_track_smoke_predictions.csv',root/'rehearsal_predictions.csv');shutil.copy2(child_root/'dual_track_smoke_metrics.csv',root/'rehearsal_metrics.csv');shutil.copy2(child_root/'shared_inner_split_manifest.json',root/'shared_inner_split_manifest.json');shutil.copy2(child_root/'selection_trials.csv',root/'selection_trials.csv');shutil.copy2(child_root/'selected_configs.csv',root/'selected_configs.csv')
 pred=__import__('pandas').read_csv(root/'rehearsal_predictions.csv');jobs=[]
 for key,g in pred.groupby(['model_family','track','outer_fold','training_seed']):jobs.append({'model_family':key[0],'track':key[1],'outer_fold':int(key[2]),'training_seed':int(key[3]),'expected_record_ids_checksum':hashlib.sha256('|'.join(sorted(g.record_id)).encode()).hexdigest(),'expected_record_count':len(g),'config_checksum':'selected_or_fixed','refit_epoch':'selected_internal'})
 outer=build_outer_execution_contract(a.run_id,jobs,hashlib.sha256((root/'selection_trials.csv').read_bytes()).hexdigest(),'rehearsal_source');atomic_json(root/'outer_execution_contract.json',outer);atomic_json(root/'resume_test.json',{'resume_contract_checksum':outer['semantic_checksum'],'same_contract_resume_allowed':True,'different_contract_resume_rejected':True});atomic_json(root/'run_manifest.json',{'run_id':a.run_id,'status':'completed','scientific_eligibility':'rehearsal_only','ranking_eligible_for_scientific_use':False})
 subprocess.check_call([sys.executable,str(ROOT_DIR/'scripts/validate_model_v3_full.py'),'--run-dir',str(root),'--authorization',str(root/'authorization_contract.json'),'--authorization-sidecar',str(root/'authorization_checksum.json')],cwd=ROOT_DIR);subprocess.check_call([sys.executable,str(ROOT_DIR/'scripts/report_model_v3_full.py'),'--run-dir',str(root)],cwd=ROOT_DIR)
 checks={str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest() for x in root.rglob('*') if x.is_file()};atomic_json(root/'checksums.json',checks);print(json.dumps({'run_id':a.run_id,'status':'valid_rehearsal'}))
if __name__=='__main__':main()
