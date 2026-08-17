"""Phase7B deterministic job identity, cache, and aggregation primitives."""
from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np, torch, time, copy
from sklearn.metrics import average_precision_score
from scripts.hybrid.run_hybrid_v1 import split_fit_stop
from src.hybrid.data.uci import build_uci_combined,UCI_NUMERIC_CONTEXT,UCI_CATEGORICAL_CONTEXT
from src.hybrid.phase7.data import build_uci_phase7_view
from src.hybrid.phase7.model import UnifiedHybrid,UnifiedHybridConfig
from src.hybrid.training.data import ContextPreprocessor
from src.hybrid.training.evaluation import binary_classification_metrics
from scripts.hybrid.run_hybrid_phase4b import prepare as prepare_phase4b
from src.hybrid.optimization.phase4b import fit_screen_fold

ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'artifacts/hybrid/phase7'; JOBS=OUT/'runtime/jobs'
def canonical(value: Any)->str:return json.dumps(value,sort_keys=True,separators=(',',':'))
def job_identity(*,domain,candidate,fold,seed,data_manifest_hash,cohort_hash,split_hash,model_source_hash,training_config_hash):
 return {'phase':'phase7b','domain':domain,'candidate':candidate,'fold':int(fold),'seed':int(seed),'data_manifest_hash':data_manifest_hash,'cohort_hash':cohort_hash,'split_hash':split_hash,'model_source_hash':model_source_hash,'training_config_hash':training_config_hash,'outer_test_used':False}
def identity_hash(identity):return hashlib.sha256(canonical(identity).encode()).hexdigest()
def job_path(identity):return JOBS/f"{identity['domain']}__{identity['candidate']}__f{identity['fold']}__s{identity['seed']}__{identity_hash(identity)}.json"
def atomic_json(path:Path,value:dict):
 path.parent.mkdir(parents=True,exist_ok=True); fd,name=tempfile.mkstemp(dir=path.parent,suffix='.tmp');
 with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(value,f,indent=2,sort_keys=True)
 os.replace(name,path)
def load_completed(identity):
 path=job_path(identity)
 if not path.exists():return None
 value=json.loads(path.read_text(encoding='utf-8'))
 if value.get('identity')!=identity or value.get('identity_hash')!=identity_hash(identity):raise RuntimeError('STALE_CACHE_REJECTED')
 return value if value.get('status')=='completed' else None
def write_completed(identity,result):
 payload={'identity':identity,'identity_hash':identity_hash(identity),'status':'completed',**result};atomic_json(job_path(identity),payload);atomic_json(OUT/'progress.json',{'state':'SMOKE_COMPLETED','last_identity':identity,'last_identity_hash':identity_hash(identity),'outer_test_used':False});return payload
def candidate_flags(candidate):
 if candidate=='A0':return {'historical':True,'aggregate_branch':False,'last':False,'progress':False,'interaction':False}
 if candidate=='A1':return {'historical':True,'aggregate_branch':False,'last':False,'progress':False,'interaction':False}
 if candidate=='A2':return {'historical':False,'aggregate_branch':True,'last':False,'progress':False,'interaction':False}
 if candidate=='A3':return {'historical':False,'aggregate_branch':True,'last':True,'progress':True,'interaction':False}
 if candidate=='A4':return {'historical':False,'aggregate_branch':True,'last':True,'progress':True,'interaction':True}
 raise ValueError(candidate)
def aggregate_jobs():
 rows=[]
 for path in JOBS.glob('*.json'):
  job=json.loads(path.read_text());
  for stage,metric in job.get('metrics',{}).items():rows.append({'domain':job['identity']['domain'],'candidate':job['identity']['candidate'],'fold':job['identity']['fold'],'seed':job['identity']['seed'],'stage':stage,**metric,'parameter_count':job.get('parameter_count'),'runtime_seconds':job.get('runtime_seconds')})
 if not rows:return pd.DataFrame()
 return pd.DataFrame(rows)

def historical_oulad_a0_config():
 """Return the immutable Phase4B-screening/R0 configuration used as OULAD A0.

 This is deliberately read from the historical selected-config artefact, rather
 than copied into Phase7.  A0 must be reproducible independently of Phase7
 representation changes.
 """
 # Phase4B representation screening calls ``base_params(domain)``.  The
 # later Phase4B selected-config is a targeted-HPO output and is therefore
 # not the A0 screening reference.
 payload=json.loads((ROOT/'artifacts/hybrid/phase4/oulad_phase4a_selected_config.json').read_text(encoding='utf-8'))
 if payload.get('domain')!='oulad' or payload.get('parameter_count')!=1270145:
  raise RuntimeError('HISTORICAL_OULAD_A0_SOURCE_MISMATCH')
 return payload

def run_historical_oulad_a0(*,fold:int,seed:int,frozen_inputs:dict):
 """Run the exact Phase4B OULAD R0 screening path for a Phase7 A0 gate.

 The Phase4B helper owns the historical FIT/STOP/VALID split, train-only
 preprocessing, scheduler, loss weighting, checkpoint selection and prefix
 sampling.  This adapter only maps its result into the Phase7 atomic-job
 envelope; it does not introduce a second training protocol.
 """
 historic=historical_oulad_a0_config(); params=historic['params']; candidate='A0'
 training_hash=hashlib.sha256(canonical({'historical_source':'phase4b_r0','params':params,'representation':'R0'}).encode()).hexdigest()
 identity=job_identity(domain='oulad',candidate=candidate,fold=fold,seed=seed,**frozen_inputs,training_config_hash=training_hash)
 cached=load_completed(identity)
 if cached:
  cached['cache_hit']=True
  return cached
 # `screening=True` is the historic development scope: one inner fold for
 # FIT/STOP, a disjoint inner VALID fold, and the remaining confirmation fold
 # excluded.  The helper also asserts no outer IDs enter the split.
 _,_,_,_,stages,contexts,prep,fit,stop,_,valid_by=prepare_phase4b('oulad',fold,True)
 result=fit_screen_fold(
  stages,fit,stop,valid_by,contexts,
  next(iter(stages.values())).temporal.shape[2],prep.output_dim,params,'R0',seed,
 )
 metrics={stage:{
  'pr_auc':float(value),'roc_auc':None,'risk_precision':None,'risk_recall':None,
  'risk_f1':None,'balanced_accuracy':None,'train_pr_auc':None,
  'validation_pr_auc':float(value),'train_validation_gap':None,
  'selected_threshold':None,'threshold_source':'historical_phase4b_screening_pr_only',
 } for stage,value in result['per_stage'].items()}
 return write_completed(identity,{
  'metrics':metrics,'parameter_count':int(result['parameter_count']),
  'runtime_seconds':float(result['runtime_seconds']),
  'peak_gpu_memory_bytes':int(result['peak_gpu_memory_bytes']),
  'best_epoch':int(result['best_epoch']),'early_stop_macro':float(result['early_stop_macro']),
  'historical_source':{
  'artifact':'artifacts/hybrid/phase4/oulad_phase4a_selected_config.json',
   'implementation':'src/hybrid/optimization/phase4b.py:fit_screen_fold',
   'representation':'R0','preprocessing':'scripts/hybrid/run_hybrid_phase4b.py:prepare',
   'split_scope':'historic_screening_fit_stop_valid',
  },'cache_hit':False,
 })

def run_hybrid_job(domain,candidate,fold,seed,frozen_inputs,training_config):
 """BCE-only shared-head inner-development runner; smoke currently exercises UCI A2."""
 if candidate in {'A1','A2','A3','A4'}:
  from src.hybrid.phase7.execution import run_redesigned
  identity=job_identity(domain=domain,candidate=candidate,fold=fold,seed=seed,**frozen_inputs,training_config_hash=hashlib.sha256(canonical(training_config).encode()).hexdigest())
  cached=load_completed(identity)
  if cached: cached['cache_hit']=True;return cached
  result=run_redesigned(domain,candidate,fold,seed,training_config)
  result['cache_hit']=False
 return write_completed(identity,result)
 if domain!='uci' or candidate!='A2': raise NotImplementedError('engine contract is installed; smoke scope is UCI/A2 only')
 identity=job_identity(domain=domain,candidate=candidate,fold=fold,seed=seed,**frozen_inputs,training_config_hash=hashlib.sha256(canonical(training_config).encode()).hexdigest())
 cached=load_completed(identity)
 if cached: cached['cache_hit']=True;return cached
 torch.manual_seed(seed);np.random.seed(seed);device='cuda' if torch.cuda.is_available() else 'cpu';start=time.monotonic()
 frame,_=build_uci_combined(ROOT/'data/raw/student-mat.csv',ROOT/'data/raw/student-por.csv');inner=pd.read_parquet(ROOT/'artifacts/hybrid/phase1/splits/uci_inner.parquet');assign=inner[inner.outer_fold==0].copy();assign.record_id=assign.record_id.astype(str)
 valid=set(assign.loc[assign.inner_fold==fold,'record_id']);train=set(assign.loc[assign.inner_fold!=fold,'record_id']);ctx=frame.rename(columns={'global_student_group':'group_id'})[['record_id','group_id','target']+UCI_NUMERIC_CONTEXT+UCI_CATEGORICAL_CONTEXT];train_frame=ctx[ctx.record_id.astype(str).isin(train)].reset_index(drop=True);fi,si=split_fit_stop(train_frame);fit=train_frame.iloc[fi].record_id.astype(str).tolist();stop=train_frame.iloc[si].record_id.astype(str).tolist()
 if set(fit)&set(stop) or (set(fit)|set(stop))&valid:raise RuntimeError('STOP_VALID_LEAKAGE')
 prep=ContextPreprocessor(UCI_NUMERIC_CONTEXT,UCI_CATEGORICAL_CONTEXT).fit(ctx[ctx.record_id.astype(str).isin(fit)]);static=prep.transform(ctx);sm={r:static[i] for i,r in enumerate(ctx.record_id.astype(str))}
 views={s:build_uci_phase7_view(frame,s) for s in ('S0','S1','S2')};model=UnifiedHybrid(UnifiedHybridConfig(prep.output_dim,1,5,use_last_state=False,use_progress=False,use_interaction=False,dropout=training_config['dropout'])).to(device);count=sum(p.numel() for p in model.parameters());
 if count>=600000:raise RuntimeError('PARAMETER_CAP')
 y=ctx[ctx.record_id.astype(str).isin(fit)].target.to_numpy();pos=(len(y)-y.sum())/max(1,y.sum());loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos],device=device));opt=torch.optim.AdamW(model.parameters(),lr=training_config['learning_rate'],weight_decay=training_config['weight_decay']);best=-np.inf;best_state=None;best_epoch=0;stale=0
 def score(ids,stage):
  v=views[stage];ix=[v.record_id.tolist().index(r) for r in ids];out=model(torch.tensor([sm[r] for r in ids],dtype=torch.float32,device=device),torch.tensor(v.temporal[ix],device=device),torch.tensor(v.temporal_mask[ix],device=device),torch.tensor(v.lengths[ix],device=device),torch.tensor(v.aggregate[ix],device=device),torch.tensor(v.aggregate_available[ix],device=device),torch.tensor(v.progress[ix],device=device));return torch.sigmoid(out).detach().cpu().numpy(),v.target[ix]
 for epoch in range(training_config['max_epochs']):
  model.train();opt.zero_grad();losses=[]
  for stage,v in views.items():
   ix=[v.record_id.tolist().index(r) for r in fit];logit=model(torch.tensor([sm[r] for r in fit],dtype=torch.float32,device=device),torch.tensor(v.temporal[ix],device=device),torch.tensor(v.temporal_mask[ix],device=device),torch.tensor(v.lengths[ix],device=device),torch.tensor(v.aggregate[ix],device=device),torch.tensor(v.aggregate_available[ix],device=device),torch.tensor(v.progress[ix],device=device));losses.append(loss_fn(logit,torch.tensor(v.target[ix],dtype=torch.float32,device=device)))
  loss=torch.stack(losses).mean();
  if not torch.isfinite(loss):raise RuntimeError('NONFINITE_LOSS')
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),training_config['gradient_clip']);opt.step();model.eval();stop_ap=np.mean([average_precision_score(score([r for r in stop if r in set(v.record_id.astype(str))],s)[1],score([r for r in stop if r in set(v.record_id.astype(str))],s)[0]) for s,v in views.items()])
  if stop_ap>best:best,best_epoch,best_state,stale=stop_ap,epoch+1,copy.deepcopy(model.state_dict()),0
  else:stale+=1
  if stale>=training_config['patience']:break
 model.load_state_dict(best_state);metrics={}
 for s,v in views.items():
  vi=[r for r in valid if r in set(v.record_id.astype(str))];sc,yy=score(vi,s);tr,ty=score(fit,s);m=binary_classification_metrics(yy,sc,threshold=.5);m.update({'train_pr_auc':float(average_precision_score(ty,tr)),'validation_pr_auc':m['pr_auc'],'train_validation_gap':float(average_precision_score(ty,tr)-m['pr_auc']),'selected_threshold':.5,'threshold_source':'fixed_not_valid_tuned'});metrics[s]=m
 return write_completed(identity,{'metrics':metrics,'parameter_count':count,'runtime_seconds':time.monotonic()-start,'peak_gpu_memory_bytes':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,'best_epoch':best_epoch,'cache_hit':False})

def run_baseline_job(domain,family,fold,seed,frozen_inputs):
 from src.hybrid.phase7.execution import run_parity_baseline
 training_config={'fixed_family':family,'protocol':'phase7_parity_train_only'}
 identity=job_identity(domain=domain,candidate=f'baseline_{family}',fold=fold,seed=seed,**frozen_inputs,training_config_hash=hashlib.sha256(canonical(training_config).encode()).hexdigest())
 cached=load_completed(identity)
 if cached:cached['cache_hit']=True;return cached
 result=run_parity_baseline(domain,family,fold,seed);result['cache_hit']=False
 return write_completed(identity,result)

def run_controlled_hybrid_job(domain,candidate,fold,seed,frozen_inputs,training_config):
 """Standardized Phase7B A0--A4 job; H0 uses its separate historical adapter."""
 from src.hybrid.phase7.execution import run_redesigned
 protocol={**training_config,'training_protocol':'phase7_standard_3fold_fit_stop_valid_v1'}
 identity=job_identity(domain=domain,candidate=candidate,fold=fold,seed=seed,**frozen_inputs,training_config_hash=hashlib.sha256(canonical(protocol).encode()).hexdigest())
 cached=load_completed(identity)
 if cached:cached['cache_hit']=True;return cached
 result=run_redesigned(domain,candidate,fold,seed,training_config);result['cache_hit']=False
 return write_completed(identity,result)
