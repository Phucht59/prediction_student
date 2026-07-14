"""Materialize read-only V2.2 closure and frozen pre-full V3 job contracts."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import ROOT_DIR
from src.evaluation.model_v3_protocol import build_expected_jobs,checksum
from src.evaluation.protocol import file_checksum,load_fold_manifest

def main():
 report=ROOT_DIR/'reports/model_v3_protocol';closure=ROOT_DIR/'reports/neural_sanity_v2_2/v2_2_1';fold=load_fold_manifest();source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT_DIR,text=True).strip()
 features=json.loads((report/'feature_contracts.json').read_text());target=json.loads((report/'target_supervision_contract.json').read_text())
 for name in ('late_stage','early_warning'):
  features[name]['fold_manifest_checksum']=fold['manifest_checksum'];features[name]['semantic_checksum']=checksum(features[name])
 target['semantic_checksum']=checksum(target)
 counts={i:sum(1 for x in fold['assignments'] if x['outer_role']=='validation' and int(x['outer_fold'])==i) for i in range(5)}
 expected=build_expected_jobs('MODEL_V3_FULL_RUN_ID_TO_BE_ASSIGNED_AFTER_REVIEW',counts,fold['manifest_checksum'],source,features,target,smoke=False)
 (report/'expected_job_contract.json').write_text(json.dumps(expected,indent=2),encoding='utf-8')
 manifest={'patch_version':'v2.2.1','source_run':'neural-sanity-v2-2-20260714','original_artifacts_modified':False,'patch_source_commit':source,'input_training_diagnostics_checksum':file_checksum(ROOT_DIR/'artifacts/neural_sanity_v2_2/neural-sanity-v2-2-20260714/training_diagnostics.csv'),'input_control_predictions_checksum':file_checksum(ROOT_DIR/'artifacts/neural_sanity_v2_2/neural-sanity-v2-2-20260714/outer_validation_predictions.csv'),'input_v2_predictions_checksum':file_checksum(ROOT_DIR/'artifacts/benchmark_v2/benchmark-v2-full-20260713c/predictions/outer_validation_predictions.csv')}
 manifest['output_checksums']={p.name:file_checksum(p) for p in closure.iterdir() if p.is_file() and p.name!='closure_patch_manifest.json'}
 (closure/'closure_patch_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print(json.dumps({'full_expected_jobs':len(expected['jobs']),'closure_outputs':len(manifest['output_checksums'])},indent=2))
if __name__=='__main__':main()
