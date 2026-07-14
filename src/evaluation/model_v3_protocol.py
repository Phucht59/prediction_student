"""Frozen Model V3 registry, contracts, and evidence validators."""
from __future__ import annotations
import hashlib,json
from typing import Any
import pandas as pd
from src.evaluation.protocol import canonical_json

MODEL_REGISTRY={
 "M0":{"name":"nominal_small_mlp_control","ordinal":False,"regression":False,"target_supervision":"classification_only","tracks":["late_stage","early_warning"]},
 "M1":{"name":"ordinal_mlp","ordinal":True,"regression":False,"target_supervision":"classification_only","tracks":["late_stage","early_warning"]},
 "M2":{"name":"multitask_ordinal_mlp","ordinal":True,"regression":True,"target_supervision":"continuous_g3_enriched","tracks":["late_stage","early_warning"]},
 "M3":{"name":"multitask_nominal_mlp","ordinal":False,"regression":True,"target_supervision":"continuous_g3_enriched","tracks":["late_stage","early_warning"]},
 "M4":{"name":"s3_cnn_bilstm_ordinal_comparator","ordinal":True,"regression":False,"target_supervision":"classification_only","tracks":["late_stage"]},
}
SEEDS=(42,52,62,72,82)
def checksum(payload:Any)->str:return hashlib.sha256(canonical_json(payload).encode()).hexdigest()
def build_expected_jobs(run_id:str,fold_counts:dict[int,int],fold_checksum:str,source_commit:str,feature_contracts:dict,target_contract:dict,*,smoke:bool=False,config_checksums:dict[str,str]|None=None):
 jobs=[];active_folds={0:fold_counts[0]} if smoke else fold_counts;active_seeds=(42,) if smoke else SEEDS;active_tracks=("late_stage",) if smoke else ("late_stage","early_warning")
 for model_id,model in MODEL_REGISTRY.items():
  for track in active_tracks:
   if track not in model['tracks']:continue
   feature=feature_contracts[track]
   config={"model_family":model_id,"track":track,"smoke":smoke}
   for fold,count in active_folds.items():
    for seed in active_seeds:jobs.append({"run_id":run_id,"model_family":model_id,"track":track,"scenario":track,"feature_set_id":feature['feature_set_id'],"target_supervision_type":model['target_supervision'],"outer_fold":fold,"training_seed":seed,"expected_record_count":count,"config_checksum":config_checksums[model_id] if config_checksums else checksum(config),"fold_manifest_checksum":fold_checksum,"feature_contract_checksum":feature['semantic_checksum'],"target_contract_checksum":target_contract['semantic_checksum'],"source_commit":source_commit})
 contract={"contract_version":"model_v3_protocol_1","created_before_compute":True,"jobs":jobs};contract['semantic_checksum']=checksum(contract);return contract
def duplicate_jobs(frame:pd.DataFrame)->int:
 cols=['model_family','track','outer_fold','training_seed'];return int(frame.duplicated(cols,keep=False).sum())
def legacy_intersection(development_ids:set[str],legacy_ids:set[str])->set[str]:return development_ids&legacy_ids
def validate_shape_rows(frame:pd.DataFrame)->bool:
 for r in frame.itertuples():
  expected=2 if int(r.cnn_kernel_size)==1 else 3 if int(r.cnn_kernel_size)==2 else None
  if expected is None or int(r.cnn_output_sequence_length)!=expected or int(r.bilstm_input_sequence_length)!=expected:return False
 return True
def validate_loader_rows(frame:pd.DataFrame)->bool:
 for r in frame.itertuples():
  n,b=int(r.dataset_size),int(r.batch_size);expected_dropped=0 if not bool(r.drop_last_train) else n%b;expected_consumed=n-expected_dropped
  if int(r.samples_dropped_per_epoch)!=expected_dropped or int(r.samples_consumed_per_epoch)!=expected_consumed:return False
 return True
