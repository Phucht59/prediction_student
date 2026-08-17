"""Phase 4B representation screening and confirmation helpers."""
from __future__ import annotations
import copy,gc,time
import numpy as np
import optuna
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader,TensorDataset
from src.hybrid.models import Hybrid,HybridConfig
from src.hybrid.optimization.phase4a import model_kwargs
from src.hybrid.training.data import sample_prefixes,sample_prefixes_stage_balanced
from src.hybrid.training.losses import binary_pos_weight
from src.hybrid.training.trainer import assemble,eligible_prefixes,seed_everything,_scheduler

PROGRESS={'S0':0.,'S1':.5,'S2':1.,'20pct':.2,'35pct':.35,'50pct':.5,'75pct':.75,'FINAL':1.}
CANDIDATES=('R0','R1','R2','R3','R4')
SEEDS=(42,1201,2026)
def progress_value(stage:str)->float:return PROGRESS[stage]
def candidate_model_kwargs(params,candidate):
 out=model_kwargs(params);out.update({'representation':candidate,'summary_hidden':params.get('summary_hidden',64),'progress_hidden':params.get('progress_hidden',16)});return out
def _assemble(stages,ids,chosen,contexts):
 x,m,l,c,y=assemble(stages,ids,chosen,contexts);p=np.asarray([progress_value(s) for s in chosen],np.float32);return x,m,l,c,y,p
def _loader(arrays,batch,seed,shuffle):
 tensors=tuple(torch.from_numpy(x) for x in arrays);return DataLoader(TensorDataset(*tensors),batch_size=batch,shuffle=shuffle,generator=torch.Generator().manual_seed(seed),pin_memory=True)
def _sample(ids,available,seed,epoch,mode):return sample_prefixes_stage_balanced(ids,available,seed,epoch) if mode=='stage_balanced' else sample_prefixes(ids,available,seed,epoch)
@torch.no_grad()
def predict(model,stages,stage,ids,contexts,batch):
 arrays=_assemble(stages,ids,[stage]*len(ids),contexts);scores=[];gates=[];model.eval()
 for temporal,mask,lengths,context,_,progress in _loader(arrays,batch,42,False):
  with torch.autocast('cuda',dtype=torch.float16):logits=model(temporal.cuda(non_blocking=True),mask.cuda(non_blocking=True),lengths.cuda(non_blocking=True),context.cuda(non_blocking=True),progress.cuda(non_blocking=True))
  scores.append(torch.sigmoid(logits).float().cpu().numpy())
  if hasattr(model,'last_gate_weights'):gates.append(model.last_gate_weights.cpu().numpy())
 return np.concatenate(scores),None if not gates else np.concatenate(gates).mean(0)
def fit_screen_fold(stages,fit_ids,stop_ids,valid_by,contexts,temporal_dim,context_dim,params,candidate,seed,trial=None,step_offset=0):
 seed_everything(seed);torch.cuda.reset_peak_memory_stats();started=time.monotonic();model=Hybrid(HybridConfig(temporal_dim,context_dim,**candidate_model_kwargs(params,candidate))).cuda();nparams=sum(p.numel() for p in model.parameters());
 if nparams>2_000_000:raise optuna.TrialPruned(f'parameter_cap:{nparams}')
 optimizer=torch.optim.AdamW(model.parameters(),lr=params['learning_rate'],weight_decay=params['weight_decay']);scheduler=_scheduler(optimizer,80,params.get('warmup_epochs',0),1e-5,params['learning_rate']);scaler=torch.amp.GradScaler('cuda');targets=np.asarray([next(d.target[d.index[r]] for d in stages.values() if r in d.index) for r in fit_ids]);loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=binary_pos_weight(targets).cuda());available=eligible_prefixes(stages,fit_ids);best=-1.;best_epoch=1;best_state=None;stale=0
 for epoch in range(80):
  chosen=_sample(fit_ids,available,seed,epoch,params.get('prefix_sampling','record_uniform'));arrays=_assemble(stages,fit_ids,chosen,contexts);model.train()
  for temporal,mask,lengths,context,target,progress in _loader(arrays,params['batch_size'],seed+epoch,True):
   temporal=temporal.cuda(non_blocking=True);mask=mask.cuda(non_blocking=True);lengths=lengths.cuda(non_blocking=True);context=context.cuda(non_blocking=True);target=target.cuda(non_blocking=True);progress=progress.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):loss=loss_fn(model(temporal,mask,lengths,context,progress),target)
   if not torch.isfinite(loss):raise optuna.TrialPruned('non_finite_loss')
   scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),params['gradient_clip_norm']);scaler.step(optimizer);scaler.update()
   if not all(torch.isfinite(p).all() for p in model.parameters()):raise optuna.TrialPruned('non_finite_parameter')
  scheduler.step();stop_metrics={}
  for stage,data in stages.items():
   ids=[r for r in stop_ids if r in data.index]
   if ids:
    y=np.asarray([data.target[data.index[r]] for r in ids]);score,_=predict(model,stages,stage,ids,contexts,params['batch_size'])
    if len(np.unique(y))==2:stop_metrics[stage]=float(average_precision_score(y,score))
  macro=float(np.mean(list(stop_metrics.values())))
  if trial is not None:trial.report(macro,step_offset+epoch)
  if trial is not None and epoch>=10 and trial.should_prune():raise optuna.TrialPruned('median_pruner')
  if macro>best:best=macro;best_epoch=epoch+1;best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()});stale=0
  else:stale+=1
  if stale>=10:break
 model.load_state_dict(best_state);metrics={};gate_means={}
 for stage,ids in valid_by.items():
  data=stages[stage];y=np.asarray([data.target[data.index[r]] for r in ids]);score,gates=predict(model,stages,stage,ids,contexts,params['batch_size']);metrics[stage]=float(average_precision_score(y,score));
  if gates is not None:gate_means[stage]=gates.tolist()
 result={'per_stage':metrics,'macro':float(np.mean(list(metrics.values()))) if metrics else None,'best_epoch':best_epoch,'early_stop_macro':best,'parameter_count':nparams,'peak_gpu_memory_bytes':int(torch.cuda.max_memory_allocated()),'runtime_seconds':time.monotonic()-started,'gate_means':gate_means};del model,optimizer,scheduler,scaler,loss_fn;gc.collect();torch.cuda.empty_cache();return result
def refit_predict(stages,train_ids,valid_by,contexts,temporal_dim,context_dim,params,candidate,seed,epochs):
 seed_everything(seed);torch.cuda.reset_peak_memory_stats();started=time.monotonic();model=Hybrid(HybridConfig(temporal_dim,context_dim,**candidate_model_kwargs(params,candidate))).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=params['learning_rate'],weight_decay=params['weight_decay']);scheduler=_scheduler(optimizer,80,params.get('warmup_epochs',0),1e-5,params['learning_rate']);scaler=torch.amp.GradScaler('cuda');targets=np.asarray([next(d.target[d.index[r]] for d in stages.values() if r in d.index) for r in train_ids]);loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=binary_pos_weight(targets).cuda());available=eligible_prefixes(stages,train_ids)
 for epoch in range(epochs):
  chosen=_sample(train_ids,available,seed,epoch,params.get('prefix_sampling','record_uniform'));arrays=_assemble(stages,train_ids,chosen,contexts);model.train()
  for temporal,mask,lengths,context,target,progress in _loader(arrays,params['batch_size'],seed+epoch,True):
   temporal=temporal.cuda(non_blocking=True);mask=mask.cuda(non_blocking=True);lengths=lengths.cuda(non_blocking=True);context=context.cuda(non_blocking=True);target=target.cuda(non_blocking=True);progress=progress.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):loss=loss_fn(model(temporal,mask,lengths,context,progress),target)
   scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),params['gradient_clip_norm']);scaler.step(optimizer);scaler.update()
  scheduler.step()
 predictions={};gates={}
 for stage,ids in valid_by.items():predictions[stage],gates[stage]=predict(model,stages,stage,ids,contexts,params['batch_size'])
 return predictions,gates,{'runtime_seconds':time.monotonic()-started,'peak_gpu_memory_bytes':int(torch.cuda.max_memory_allocated()),'parameter_count':sum(p.numel() for p in model.parameters())}
def select_candidate(rows,domain,floors,tolerance=.001):
 import pandas as pd
 frame=pd.DataFrame(rows);mean=frame.groupby(['candidate','stage']).pr_auc.mean().unstack();macro=mean.mean(1);r0=mean.loc['R0'];eligible=[]
 for candidate in CANDIDATES:
  if all(mean.loc[candidate,stage]>=r0[stage]+floors[stage] for stage in mean.columns):eligible.append(candidate)
 best=max(macro[c] for c in eligible);return next(c for c in CANDIDATES if c in eligible and best-macro[c]<tolerance)
