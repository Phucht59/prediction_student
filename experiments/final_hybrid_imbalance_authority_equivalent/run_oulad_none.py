"""One isolated FINAL H1 authority-policy (uniform NONE) replay job."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'oulad_none_runs'
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0,str(ROOT/'oulad_structured_resampling'))
from src.benchmark.oulad_data import build_canonical_bundle,single_stage_rows
from src.training.release_freeze import _predict_deep_payload,_train_deep_final

def run(fold:int,seed:int,mode:str='FIXED_NONE')->None:
 d=(ROOT/('oulad_class_weight_runs' if mode=='FIXED_CLASS_WEIGHT' else 'oulad_none_runs'))/f'oulad__{mode.lower()}__fold{fold}__seed{seed}'; d.mkdir(parents=True,exist_ok=True); m=d/'run_manifest.json'
 if m.is_file() and json.loads(m.read_text()).get('status')=='COMPLETE' and (d/'checkpoint.pt').is_file() and (d/'predictions.npz').is_file():return
 m.write_text(json.dumps({'status':'RUNNING','fold':fold,'seed':seed}),encoding='utf-8')
 ref=torch.load(ROOT.parents[1]/'artifacts'/'canonical_v3'/'checkpoints'/'oulad_h1_final'/f'outer{fold}_seed{seed}.pt',map_location='cpu',weights_only=False)
 bundle=build_canonical_bundle(); base=bundle.base[['base_record_id','outer_fold']].drop_duplicates(); tr=set(base.loc[base.outer_fold.ne(fold),'base_record_id'].astype(str)); te=set(base.loc[base.outer_fold.eq(fold),'base_record_id'].astype(str)); train=single_stage_rows(bundle,'FINAL',tr); test=single_stage_rows(bundle,'FINAL',te)
 manifest={k:ref[k] for k in ('final_candidate_hash','architecture_hash','feature_schema_hash','training_policy_hash','evaluation_protocol_hash')}
 config=dict(ref['config'])
 if mode=='FIXED_CLASS_WEIGHT': config.update({'loss_policy':'weighted_bce','pos_weight_strategy':'full_ratio'})
 payload=_train_deep_final('H1_TABULAR_RESIDUAL_EXPERT',train,config,seed,int(ref['fixed_epochs']),d/'checkpoint.pt',manifest)
 p=_predict_deep_payload(d/'checkpoint.pt',test[0],test[1],test[2],test[3],test[4])
 np.savez_compressed(d/'predictions.npz',record_id=test[0].base_record_id.astype(str).to_numpy(),target=test[5],probability=p,outer_fold=np.full(len(p),fold))
 weight=float((train[5]==0).sum()/max((train[5]==1).sum(),1)) if mode=='FIXED_CLASS_WEIGHT' else 1.0
 m.write_text(json.dumps({'status':'COMPLETE','mode':mode,'fold':fold,'seed':seed,'parameter_count':payload['parameter_count'],'fixed_epochs':ref['fixed_epochs'],'train_only_pos_weight':weight}),encoding='utf-8')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--fold',type=int,required=True);a.add_argument('--seed',type=int,required=True);a.add_argument('--mode',default='FIXED_NONE',choices=('FIXED_NONE','FIXED_CLASS_WEIGHT'));x=a.parse_args();run(x.fold,x.seed,x.mode)
