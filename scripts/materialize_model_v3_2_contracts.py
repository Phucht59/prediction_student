"""Create V3.2 readiness contracts only; compute authorization stays false."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT_DIR
from src.evaluation.model_v3_protocol import build_expected_jobs, checksum
from src.evaluation.model_v3_2 import V3_2_PROTOCOL_VERSION, build_b0_selection_contract, build_shared_inner_split_manifest, git_tree_clean
from src.evaluation.protocol import load_fold_manifest

OUT=ROOT_DIR/'reports/model_v3_protocol/v3_2'
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',default='model-v3-full-v3-2-20260714');a=p.parse_args()
 if OUT.exists():raise FileExistsError(OUT)
 clean=git_tree_clean(str(ROOT_DIR));
 if not clean:raise RuntimeError('Refuse to materialize contracts from a dirty tracked tree.')
 OUT.mkdir(parents=True);source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT_DIR,text=True).strip();fm=load_fold_manifest()
 labels={x['source_record_identity']:x['true_label'] for x in fm['development_records']}; outer_train={};outer_valid={}
 for fold in range(5):
  train=[x['source_record_identity'] for x in fm['assignments'] if x['outer_fold']==fold and x['outer_role']=='train'];valid=[x['source_record_identity'] for x in fm['assignments'] if x['outer_fold']==fold and x['outer_role']=='validation']
  import pandas as pd
  outer_train[fold]=pd.DataFrame({'record_id':train,'true_label':[labels[x] for x in train]});outer_valid[fold]=valid
 inner=build_shared_inner_split_manifest(outer_train,fm['manifest_checksum'])
 feature={}
 for track,features,cutoff in [('late_stage',['G1','G2'],'after_G2'),('early_warning',['G1'],'after_G1')]:
  item={'contract_version':'v3_2_feature_1','scenario':track,'cutoff':cutoff,'feature_set_id':'+'.join(features),'ordered_features':features,'preprocessing_contract':'StandardScaler fit on current training partition only','scaler_contract':'train_only_standard_scaler','target_excluded':True,'temporal_availability_status':'allowed_by_frozen_feature_allowlist','class_order':['Low','Medium','High'],'dataset_version':1,'fold_manifest_checksum':fm['manifest_checksum']};item['semantic_checksum']=checksum(item);feature[track]=item
 target={'contract_version':'v3_2_target_1','class_order':['Low','Medium','High'],'continuous_g3':{'scaler':'fit on current training partition only','primary_scale':'raw_0_20','clip_before_primary_rmse_r2':False}};target['semantic_checksum']=checksum(target)
 search={'contract_version':'v3_2_search_1','objective':'inner_mean_macro_f1','inner_folds':3,'trials_per_study':20,'shared_pytorch_m0_m3':{'hidden_width':[8,16,32],'hidden_layers':[1,2],'dropout':[0,.15,.30],'learning_rate':{'low':.0005,'high':.005,'distribution':'log_uniform'},'weight_decay':{'low':1e-6,'high':1e-3,'distribution':'log_uniform'},'batch_size':[16,32],'max_epochs':60,'patience':10,'drop_last':False},'multitask_lambda':[.1,.3,1.0]};search['semantic_checksum']=checksum(search)
 neural=[]
 for fam in ['M0','M1','M2','M3']:
  for track in ['late_stage','early_warning']:
   for fold in range(5):
    neural.append({'study_id':f'{a.run_id}:{fam}:{track}:outer{fold}','model_family':fam,'track':track,'outer_fold':fold,'trial_budget':20,'expected_inner_evaluations':60,'objective':'maximize_inner_mean_macro_f1','inner_split_manifest_checksum':inner['semantic_checksum'],'search_space_checksum':search['semantic_checksum'],'target_supervision_checksum':target['semantic_checksum'],'source_commit':source})
 neural={'contract_version':V3_2_PROTOCOL_VERSION,'run_id':a.run_id,'created_before_compute':True,'studies':neural};neural['semantic_checksum']=checksum(neural)
 b0=build_b0_selection_contract(a.run_id,source,inner['semantic_checksum'],feature)
 s3=json.loads((ROOT_DIR/'artifacts/benchmark_v2/benchmark-v2-full-20260713c/configs/selected_configs.json').read_text())
 m4=[]
 for fold in range(5):
  cfg={**s3[f'late_stage/cnn_bilstm_v2_tuned/fold{fold}']['config'],'max_epochs':40,'patience':8,'scheduler_patience':3};m4.append({'outer_fold':fold,'source_s3_run_id':'neural-sanity-v2-2-20260714','source_s3_artifact_path':f'artifacts/benchmark_v2/benchmark-v2-full-20260713c/configs/selected_configs.json','source_s3_config_checksum':checksum(s3[f'late_stage/cnn_bilstm_v2_tuned/fold{fold}']['config']),'config':cfg,'fixed_config_checksum':checksum(cfg),'meaning':'architecture and hyperparameters fixed; weights are reinitialized and trained with ordinal loss'})
 m4={'contract_version':V3_2_PROTOCOL_VERSION,'run_id':a.run_id,'fixed_sequence_backbone_not_frozen_weights':True,'configs':m4};m4['semantic_checksum']=checksum(m4)
 counts={f:len(outer_valid[f]) for f in range(5)};expected=build_expected_jobs(a.run_id,counts,fm['manifest_checksum'],source,feature,target,selection_contract_checksum=neural['semantic_checksum'])
 expected['contract_version']=V3_2_PROTOCOL_VERSION
 for job in expected['jobs']:
  job['smoke']=False;job['inner_split_manifest_checksum']=inner['semantic_checksum']
  if job['model_family']=='M4':
   fixed=m4['configs'][job['outer_fold']];job['fixed_config_checksum']=fixed['fixed_config_checksum'];job['source_s3_config_checksum']=fixed['source_s3_config_checksum']
 expected['semantic_checksum']=checksum({k:v for k,v in expected.items() if k!='semantic_checksum'})
 acceptance={'contract_version':V3_2_PROTOCOL_VERSION,'comparison_pairs':[['M1','M0'],['M4','REF_CNN_S3_NOMINAL'],['M2','M3'],['M2','M1'],['M3','M0']],'paired_non_decreasing_tolerance':-1e-12,'seed_sd_relative_increase_max':.25,'seed_sd_absolute_increase_max':.01,'strong_baseline_gap_material':.03};acceptance['semantic_checksum']=checksum(acceptance)
 auth={'protocol_version':V3_2_PROTOCOL_VERSION,'execution_mode':'full','run_id':a.run_id,'source_commit':source,'source_tree_clean':clean,'expected_job_contract_checksum':expected['semantic_checksum'],'selection_study_contract_checksum':neural['semantic_checksum'],'b0_selection_contract_checksum':b0['semantic_checksum'],'inner_split_manifest_checksum':inner['semantic_checksum'],'feature_contract_checksum':checksum(feature),'target_contract_checksum':target['semantic_checksum'],'search_contract_checksum':search['semantic_checksum'],'acceptance_contract_checksum':acceptance['semantic_checksum'],'fixed_m4_config_contract_checksum':m4['semantic_checksum'],'compute_authorized':False,'authorized_at':None,'authorization_reason':'Awaiting scientific review; materialization does not grant compute authority.'}
 schema={'outer_execution_contract':'created after validated selection and before outer refits','required_job_fields':['selected_config_checksum_or_fixed_config_checksum','refit_epochs','feature_contract_checksum','target_contract_checksum','fold_manifest_checksum','inner_split_manifest_checksum','selection_study_checksum','source_commit'],'five_seeds':[42,52,62,72,82],'b0_seed':0}
 epoch={'rule':'round_half_up(median(best_epoch_inner_fold_0,best_epoch_inner_fold_1,best_epoch_inner_fold_2))','bounds':'1 <= refit_epochs <= max_epochs','m4':'internal outer-train epoch selection; no nominal-S3 epoch reuse'}
 validator={'reject_partial_run_for_ranking':True,'require_pooled_oof_exact_316_ids':True,'require_m4_fixed_config_binding':True,'require_authorization_manifest':True}
 for name,obj in [('full_execution_authorization.json',auth),('shared_inner_split_manifest.json',inner),('selection_study_contract.json',neural),('b0_selection_contract.json',b0),('fixed_m4_config_contract.json',m4),('expected_job_contract.json',expected),('outer_execution_contract_schema.json',schema),('final_refit_epoch_contract.json',epoch),('full_validator_contract.json',validator),('feature_contracts.json',feature),('target_contract.json',target),('search_contract.json',search),('acceptance_criteria.json',acceptance)]:dump(name,obj)
 (OUT/'readiness_audit.md').write_text('# V3.2 readiness audit\n\nExecution remains unauthorized. Shared inner assignments are record-ID based and independent of model/trial/seed. M0–M3 use common inner folds; B0 has 10 deterministic alpha-selection studies; M4 binds fixed per-fold S3 architecture/hyperparameter checksums.\n')
 (OUT/'compute_estimate.md').write_text('# Compute estimate\n\nM0–M3: 40 × 20 × 3 = 2,400 inner evaluations. B0: 10 × 4 × 3 = 120. Total: 2,520. Outer evaluation: 235 jobs, 14,852 prediction rows. Not authorized.\n')
 (OUT/'pre_full_run_decision.md').write_text('# Decision\n\nStatus: **not authorized**. `compute_authorized` remains false. A future execution authorization must be separately signed/materialized from a clean source revision.\n')
 manifest={'protocol_version':V3_2_PROTOCOL_VERSION,'source_commit':source,'source_tree_clean_verified_by_git':clean,'run_id':a.run_id,'compute_authorized':False,'expected_jobs':235,'expected_predictions':14852,'created_at':datetime.now(timezone.utc).isoformat()};dump('protocol_manifest.json',manifest)
 checks={x.name:sha(x) for x in OUT.iterdir() if x.is_file() and x.name!='checksums.json'};dump('checksums.json',checks)
if __name__=='__main__':main()
