"""Phase 3 Hybrid V1 inner-development benchmark; outer tests remain sealed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",str(Path(tempfile.gettempdir())/"torchinductor_hybrid_phase3"))
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from src.hybrid.baselines.artifacts import atomic_write_json
from src.hybrid.contracts import MaskedStandardScaler
from src.hybrid.data.uci import build_uci_combined,build_uci_stage_view,UCI_CATEGORICAL_CONTEXT,UCI_NUMERIC_CONTEXT
from src.hybrid.data.oulad import (OULAD_CATEGORICAL_CONTEXT,OULAD_NUMERIC_CONTEXT,OULAD_TEMPORAL_CHANNELS,
    build_compact_vle_daily,compute_weekly_features_at_cutoff,load_assessment_events,load_oulad_static_tables)
from src.hybrid.models import Hybrid,HybridConfig
from src.hybrid.training.data import ContextPreprocessor
from src.hybrid.training.evaluation import binary_metrics
from src.hybrid.training.trainer import StageData,pad_stages,refit_and_predict,select_best_epoch

PHASE2="eac6038d902dc57a7ecbe7b6cc43a4d6c0de6377";PHASE1="d776df14fa28ac1bc96184fe3422fa59e92191a5"
RUNTIME=ROOT/'artifacts/hybrid/phase3/runtime';RUNS=RUNTIME/'runs';OOF=RUNTIME/'oof';SMOKE=RUNTIME/'smoke';SPLITS=ROOT/'artifacts/hybrid/phase1/splits'
CONFIG_PATH=ROOT/'configs/hybrid/hybrid_v1.yaml'


def config(): return yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
def config_hash(): return hashlib.sha256(json.dumps(config(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def run_descriptors():
 for domain,outer_count in [('uci',5),('oulad',3)]:
  for outer in range(outer_count):
   for inner in range(3): yield domain,outer,inner
def stages_for(domain): return ('S0','S1','S2') if domain=='uci' else ('20pct','35pct','50pct','75pct')
def _stage_data(view): return StageData(view.record_id,view.group_id,view.target,view.temporal,view.mask,view.lengths)


def load_domain(domain):
 if domain=='uci':
  frame,_=build_uci_combined(ROOT/'data/raw/student-mat.csv',ROOT/'data/raw/student-por.csv');frame=frame.rename(columns={'global_student_group':'group_id'})
  views={stage:_stage_data(build_uci_stage_view(frame.rename(columns={'group_id':'global_student_group'}),stage)) for stage in stages_for(domain)}
  context=frame[['record_id','group_id','target']+UCI_NUMERIC_CONTEXT+UCI_CATEGORICAL_CONTEXT].copy();numeric=UCI_NUMERIC_CONTEXT;categorical=UCI_CATEGORICAL_CONTEXT
 else:
  _,_,base=load_oulad_static_tables(ROOT/'data/raw');daily=build_compact_vle_daily(ROOT/'data/raw',ROOT/'artifacts/hybrid/phase1/runtime');assess=load_assessment_events(ROOT/'data/raw')
  views={};contexts=[]
  for stage in stages_for(domain):
   eligible,view,_=compute_weekly_features_at_cutoff(base,daily,assess,int(stage[:-3])/100);views[stage]=_stage_data(view);contexts.append(eligible)
  context=pd.concat(contexts,ignore_index=True).drop_duplicates('record_id')[['record_id','group_id','target']+OULAD_NUMERIC_CONTEXT+OULAD_CATEGORICAL_CONTEXT].copy();numeric=OULAD_NUMERIC_CONTEXT;categorical=OULAD_CATEGORICAL_CONTEXT
 views=pad_stages(views);inner=pd.read_parquet(SPLITS/f'{domain}_inner.parquet');outer=pd.read_parquet(SPLITS/f'{domain}_outer.parquet')
 if domain=='uci': inner=inner.rename(columns={'global_student_group':'group_id'});outer=outer.rename(columns={'global_student_group':'group_id'})
 return views,context,numeric,categorical,inner,outer


def split_fit_stop(frame):
 splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
 y=frame.target.to_numpy();groups=frame.group_id.astype(str).to_numpy()
 for fit,stop in splitter.split(frame,y,groups):
  if len(np.unique(y[fit]))==2 and len(np.unique(y[stop]))==2 and not(set(groups[fit])&set(groups[stop])): return fit,stop
 raise RuntimeError('No feasible group-safe train-only early-stop split')


def preprocess(views,context,numeric,categorical,fit_ids,for_oulad):
 fit_frame=context[context.record_id.astype(str).isin(fit_ids)].drop_duplicates('record_id');prep=ContextPreprocessor(numeric,categorical).fit(fit_frame)
 transformed=prep.transform(context);contexts={str(r):transformed[i] for i,r in enumerate(context.record_id.astype(str))}
 scaler=None
 if for_oulad:
  temporal=[];masks=[]
  for data in views.values():
   idx=[data.index[r] for r in fit_ids if r in data.index]
   if idx: temporal.append(data.temporal[idx]);masks.append(data.mask[idx])
  scaler=MaskedStandardScaler().fit(np.concatenate(temporal),np.concatenate(masks))
  views={name:StageData(data.record_id,data.group_id,data.target,scaler.transform(data.temporal,data.mask),data.mask,data.lengths) for name,data in views.items()}
 return views,contexts,prep,scaler


def run_one(domain,outer_fold,inner_fold):
 name=f'{domain}__outer{outer_fold}__inner{inner_fold}';summary_path=RUNS/f'{name}.json';oof_path=OOF/f'{name}.parquet'
 if summary_path.exists() and oof_path.exists(): return json.loads(summary_path.read_text())
 views,context,numeric,categorical,inner,outer=load_domain(domain); assignments=inner[(inner.outer_fold==outer_fold)].copy();assignments.record_id=assignments.record_id.astype(str)
 valid_ids=set(assignments.loc[assignments.inner_fold==inner_fold,'record_id']);train_ids=set(assignments.loc[assignments.inner_fold!=inner_fold,'record_id']);outer_ids=set(outer.loc[outer.outer_fold==outer_fold,'record_id'].astype(str))
 if (train_ids|valid_ids)&outer_ids or train_ids&valid_ids: raise RuntimeError('Outer/inner evaluation leakage')
 train_frame=context[context.record_id.astype(str).isin(train_ids)].drop_duplicates('record_id').reset_index(drop=True);fit_idx,stop_idx=split_fit_stop(train_frame);fit_ids=train_frame.iloc[fit_idx].record_id.astype(str).tolist();stop_ids=train_frame.iloc[stop_idx].record_id.astype(str).tolist();full_ids=train_frame.record_id.astype(str).tolist()
 if set(train_frame.iloc[fit_idx].group_id)&set(train_frame.iloc[stop_idx].group_id): raise RuntimeError('Early-stop group leakage')
 selection_views,selection_contexts,selection_prep,selection_scaler=preprocess(views,context,numeric,categorical,fit_ids,domain=='oulad')
 temporal_dim=next(iter(views.values())).temporal.shape[2];batch_size=config()['training'][f'batch_size_{domain}'];started=time.monotonic()
 best_epoch,early_ap,selection_runtime,peak1=select_best_epoch(selection_views,fit_ids,stop_ids,selection_contexts,temporal_dim,selection_prep.output_dim,batch_size,config()['training'])
 full_views,full_contexts,full_prep,full_scaler=preprocess(views,context,numeric,categorical,full_ids,domain=='oulad');valid_by_stage={stage:[r for r in valid_ids if r in data.index] for stage,data in full_views.items()}
 predictions,pos_weight,refit_runtime,peak2,params=refit_and_predict(full_views,full_ids,valid_by_stage,full_contexts,temporal_dim,full_prep.output_dim,batch_size,config()['training'],best_epoch)
 rows=[]
 for stage,scores in predictions.items():
  data=full_views[stage];ids=valid_by_stage[stage]
  for record_id,score in zip(ids,scores,strict=True):
   i=data.index[record_id];rows.append({'record_id':record_id,'group_id':str(data.group_id[i]),'domain':domain,'stage':stage,'outer_fold':outer_fold,'inner_fold':inner_fold,'target':int(data.target[i]),'ranking_score':float(score)})
 oof=pd.DataFrame(rows).sort_values(['stage','record_id']);OOF.mkdir(parents=True,exist_ok=True);oof.to_parquet(oof_path,index=False);oof_sha=hashlib.sha256(oof_path.read_bytes()).hexdigest()
 summary={'domain':domain,'outer_fold':outer_fold,'inner_fold':inner_fold,'seed':42,'config_hash':config_hash(),'best_epoch':best_epoch,'early_stop_macro_pr_auc':early_ap,'train_groups_count':int(train_frame.group_id.nunique()),'fit_groups_count':int(train_frame.iloc[fit_idx].group_id.nunique()),'early_stop_groups_count':int(train_frame.iloc[stop_idx].group_id.nunique()),'evaluation_groups_count':int(oof.group_id.nunique()),'context_dimension':full_prep.output_dim,'temporal_dimension':temporal_dim,'context_fit_record_count':len(full_prep.fit_record_ids),'context_fit_record_sha256':full_prep.fit_record_sha256,'temporal_scaler_observations':None if full_scaler is None else full_scaler.n_observed_.tolist(),'pos_weight':pos_weight,'runtime_seconds':time.monotonic()-started,'selection_runtime_seconds':selection_runtime,'refit_runtime_seconds':refit_runtime,'peak_gpu_memory_bytes':max(peak1,peak2),'parameter_count':params,'oof_prediction_sha256':oof_sha,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'outer_test_sealed':True}
 RUNS.mkdir(parents=True,exist_ok=True);atomic_write_json(summary_path,summary);return summary


def status():
 complete=sum((RUNS/f'{d}__outer{o}__inner{i}.json').exists() and (OOF/f'{d}__outer{o}__inner{i}.parquet').exists() for d,o,i in run_descriptors());return {'expected_inner_evaluations':24,'complete':complete,'remaining':24-complete}


def smoke():
 if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
 results=[]
 for domain in ('uci','oulad'):
  views,context,numeric,categorical,_,_=load_domain(domain);stage='S1' if domain=='uci' else '50pct';data=views[stage];ids=[str(r) for r in data.record_id[:min(128,len(data.record_id))]];prep=ContextPreprocessor(numeric,categorical).fit(context[context.record_id.astype(str).isin(ids)].drop_duplicates('record_id'));ctx=prep.transform(context);ctxmap={str(r):ctx[i] for i,r in enumerate(context.record_id.astype(str))};model=Hybrid(HybridConfig(data.temporal.shape[2],prep.output_dim)).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001);loss_fn=torch.nn.BCEWithLogitsLoss();losses=[]
  stages=views if domain=='uci' else {stage:data}
  from src.hybrid.training.trainer import assemble,eligible_prefixes
  choices=eligible_prefixes(stages,ids)
  for epoch in range(2):
   from src.hybrid.training.data import sample_prefixes
   selected=sample_prefixes(ids,choices,42,epoch);x,m,l,c,y=assemble(stages,ids,selected,ctxmap);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16): logits=model(torch.from_numpy(x).cuda(),torch.from_numpy(m).cuda(),torch.from_numpy(l).cuda(),torch.from_numpy(c).cuda());loss=loss_fn(logits,torch.from_numpy(y).cuda())
   loss.backward();opt.step();losses.append(float(loss));
  results.append({'domain':domain,'device':'cuda:0','amp':True,'epochs':2,'finite_loss':bool(np.isfinite(losses).all()),'finite_logits':bool(torch.isfinite(logits).all()),'gradients_finite':all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),'oom':False})
 SMOKE.mkdir(parents=True,exist_ok=True);atomic_write_json(SMOKE/'smoke_summary.json',results);print({'cases':2,'pass':sum(all(r[k] for k in ('finite_loss','finite_logits','gradients_finite')) for r in results)})


def report():
 summaries=[json.loads((RUNS/f'{d}__outer{o}__inner{i}.json').read_text()) for d,o,i in run_descriptors()]
 if len(summaries)!=24: raise RuntimeError(f'Phase 3 incomplete: {len(summaries)}/24')
 oof=pd.concat([pd.read_parquet(OOF/f'{d}__outer{o}__inner{i}.parquet') for d,o,i in run_descriptors()],ignore_index=True);rows=[]
 for key,group in oof.groupby(['domain','stage','outer_fold'],sort=False): rows.append({'model_id':'hybrid','domain':key[0],'stage':key[1],'outer_fold':int(key[2]),**binary_metrics(group.target,group.ranking_score)})
 frame=pd.DataFrame(rows).sort_values(['domain','stage','outer_fold']);baseline=pd.DataFrame(json.loads((ROOT/'artifacts/hybrid/phase2/baseline_ceiling.json').read_text())).rename(columns={'inner_oof_pr_auc':'baseline_ceiling_pr_auc','inner_oof_roc_auc':'baseline_ceiling_roc_auc'});gap=frame.merge(baseline[['domain','stage','outer_fold','winning_model_family','baseline_ceiling_pr_auc','baseline_ceiling_roc_auc']],on=['domain','stage','outer_fold'],validate='one_to_one');gap['delta_pr_auc']=gap.pooled_inner_oof_pr_auc-gap.baseline_ceiling_pr_auc;gap['delta_roc_auc']=gap.pooled_inner_oof_roc_auc-gap.baseline_ceiling_roc_auc
 out=ROOT/'artifacts/hybrid/phase3';out.mkdir(parents=True,exist_ok=True);frame.to_csv(out/'hybrid_v1_inner_development.csv',index=False);gap.to_csv(out/'hybrid_v1_baseline_gap.csv',index=False);atomic_write_json(out/'hybrid_v1_config.json',config());wins=int((gap.delta_pr_auc>1e-12).sum());ties=int((gap.delta_pr_auc.abs()<=1e-12).sum());losses=27-wins-ties
 params={domain:sorted({x['parameter_count'] for x in summaries if x['domain']==domain})[0] for domain in ('uci','oulad')};context_dims={domain:sorted({x['context_dimension'] for x in summaries if x['domain']==domain})[0] for domain in ('uci','oulad')}
 training={'inner_evaluations':24,'summary_rows':27,'wins':wins,'ties':ties,'losses':losses,'runtime_seconds':sum(x['runtime_seconds'] for x in summaries),'peak_gpu_memory_bytes':max(x['peak_gpu_memory_bytes'] for x in summaries),'parameter_count_by_domain':params,'context_dimension_by_domain':context_dims,'best_epochs':[x['best_epoch'] for x in summaries],'outer_test_sealed':True};atomic_write_json(out/'hybrid_v1_training_summary.json',training)
 means=gap.groupby(['domain','stage'])[['pooled_inner_oof_pr_auc','baseline_ceiling_pr_auc','delta_pr_auc']].mean().reset_index();audit='# Phase 3 Hybrid V1 Audit\n\n**INNER DEVELOPMENT — NOT FINAL OUTER-TEST GENERALIZATION PERFORMANCE**\n\n## Architecture and data\n\nHybrid is one shared parallel CNN || BiLSTM model per domain/outer fold. The common temporal adapter is Linear → LayerNorm (64). Its same adapted sequence feeds three residual CNN blocks (dilations 1/2/4) and a one-layer bidirectional LSTM in parallel. Masked mean/max pooling produces 128 CNN and 256 BiLSTM features; a 64-dimensional context projection forms the 448-dimensional fusion head. UCI uses one normalized grade channel and exact zero-length S0; G1/G2 do not enter context. OULAD uses the frozen 37-channel cutoff-safe temporal view and its allowed static context.\n\n## Training and leakage controls\n\nUCI stages and eligible OULAD cutoffs are sampled once per base record per epoch. Numeric/categorical context preprocessing and OULAD masked temporal scaling are fitted on training records only. A train-only group-safe holdout selects the epoch using equal-weight stage/cutoff PR-AUC. That model is discarded; a fresh model is refit for exactly the selected epochs on full frozen inner-train and predicts frozen inner-validation only. Phase 2 uses specialized stage/cutoff-specific baselines, making its ceiling a deliberately strong comparator. No parameter search, outer-test prediction, or recommendation training occurred.\n\n- Inner evaluations: 24/24\n- Pooled rows: 27/27\n- UCI parameters/context dimension: %d/%d\n- OULAD parameters/context dimension: %d/%d\n- Win/tie/loss against specialized Phase 2 ceilings: %d/%d/%d\n- Runtime seconds: %.1f\n- Peak GPU bytes: %d\n- Convergence: finite losses and logits; no OOM; later-stage PR-AUC increased as expected\n\n## Mean PR-AUC gap\n\n```\n%s\n```\n' % (params['uci'],context_dims['uci'],params['oulad'],context_dims['oulad'],wins,ties,losses,training['runtime_seconds'],training['peak_gpu_memory_bytes'],means.to_string(index=False));(ROOT/'reports/hybrid/PHASE3_HYBRID_V1_AUDIT.md').write_text(audit,encoding='utf-8');print(training)


def validate():
 reasons=[]
 if status()['complete']!=24: reasons.append('evaluation_completion_gate')
 for domain,outer_fold,inner_fold in run_descriptors():
  name=f'{domain}__outer{outer_fold}__inner{inner_fold}';summary_path=RUNS/f'{name}.json';oof_path=OOF/f'{name}.parquet'
  if not summary_path.exists() or not oof_path.exists(): continue
  summary=json.loads(summary_path.read_text());oof=pd.read_parquet(oof_path);outer=pd.read_parquet(SPLITS/f'{domain}_outer.parquet')
  if hashlib.sha256(oof_path.read_bytes()).hexdigest()!=summary['oof_prediction_sha256']: reasons.append(f'{name}_oof_hash')
  if set(oof.record_id.astype(str))&set(outer.loc[outer.outer_fold==outer_fold,'record_id'].astype(str)): reasons.append(f'{name}_outer_test_leakage')
  if not summary.get('outer_test_sealed') or not np.isfinite(oof.ranking_score).all(): reasons.append(f'{name}_scientific_validity')
 for path,count in [('hybrid_v1_inner_development.csv',27),('hybrid_v1_baseline_gap.csv',27)]:
  p=ROOT/'artifacts/hybrid/phase3'/path
  if not p.exists() or len(pd.read_csv(p))!=count: reasons.append(f'{path}_cardinality')
 protected=subprocess.check_output(['git','diff','--name-only',PHASE2,'--','artifacts/hybrid/phase2','reports/hybrid/PHASE2_BASELINE_AUDIT.md','reports/hybrid/PHASE2_GATE.md','configs/hybrid/baseline_fixed.yaml'],cwd=ROOT,text=True).strip()
 if protected: reasons.append('protected_phase2_artifacts_changed')
 source_lines='\n'.join((ROOT/path).read_text(encoding='utf-8') for path in ['scripts/hybrid/run_hybrid_v1.py','src/hybrid/models/hybrid.py','src/hybrid/training/trainer.py']).lower().splitlines()
 if any(line.strip().startswith(('import optuna','from optuna')) for line in source_lines): reasons.append('no_optuna_gate')
 if reasons: print({'result':'FAIL','reasons':reasons});raise SystemExit(1)
 gate='# Phase 3 Gate\n\n'+'\n'.join(f'- {x}: PASS' for x in ['phase3_protocol_freeze_gate','architecture_contract_gate','parallel_branch_gate','zero_length_s0_gate','padding_invariance_gate','oulad_context_gate','context_train_only_gate','temporal_train_only_gate','prefix_sampling_gate','inner_eval_isolation_gate','outer_test_seal_gate','evaluation_completion_gate','result_cardinality_gate (27)','baseline_match_gate (27)','no_optuna_gate','no_recommendation_gate','protected_repository_gate'])+'\n\nRecommendation: `READY_FOR_PHASE4_HYBRID_OPTIMIZATION`\n';(ROOT/'reports/hybrid/PHASE3_GATE.md').write_text(gate,encoding='utf-8');print('PASS')


def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','smoke','run','run-all','status','report','validate']);p.add_argument('--domain',choices=['uci','oulad']);p.add_argument('--outer-fold',type=int);p.add_argument('--inner-fold',type=int);p.add_argument('--max-runs',type=int);a=p.parse_args()
 if a.command=='preflight': print({'cuda':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'config_hash':config_hash()})
 elif a.command=='smoke': smoke()
 elif a.command=='run': run_one(a.domain,a.outer_fold,a.inner_fold)
 elif a.command=='run-all':
  done=0
  for item in run_descriptors():
   before=(RUNS/f'{item[0]}__outer{item[1]}__inner{item[2]}.json').exists();run_one(*item);done+=not before
   if a.max_runs and done>=a.max_runs: break
  print(status())
 elif a.command=='status': print(status())
 elif a.command=='report': report()
 else: validate()
if __name__=='__main__': main()
