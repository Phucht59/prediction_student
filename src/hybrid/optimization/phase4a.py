"""Frozen Phase 4A Optuna search helpers."""
from __future__ import annotations

import copy
import gc
import time
import numpy as np
import optuna
import torch
from sklearn.metrics import average_precision_score

from src.hybrid.models import Hybrid, HybridConfig
from src.hybrid.training.data import sample_prefixes, sample_prefixes_stage_balanced
from src.hybrid.training.losses import binary_pos_weight
from src.hybrid.training.trainer import _loader, assemble, eligible_prefixes, predict, seed_everything


def equal_weight_macro_pr_auc(per_stage: dict[str, float]) -> float:
    if not per_stage or not np.isfinite(list(per_stage.values())).all():
        raise ValueError("All stage PR-AUC values must be finite")
    return float(np.mean(list(per_stage.values())))


def suggest_trial_config(trial: optuna.Trial, domain: str) -> dict:
    result={
        'd_model':trial.suggest_categorical('d_model',[64,96,128,160]),
        'cnn_channels':trial.suggest_categorical('cnn_channels',[64,96,128,160]),
        'cnn_blocks':trial.suggest_categorical('cnn_blocks',[2,3]),
        'bilstm_hidden':trial.suggest_categorical('bilstm_hidden',[64,96,128,160]),
        'bilstm_layers':trial.suggest_categorical('bilstm_layers',[1,2]),
        'context_hidden':trial.suggest_categorical('context_hidden',[64,96,128,192]),
        'head_hidden':trial.suggest_categorical('head_hidden',[128,192,256]),
        'dropout':trial.suggest_float('dropout',.05,.35),
        'learning_rate':trial.suggest_float('learning_rate',1e-4,2e-3,log=True),
        'weight_decay':trial.suggest_float('weight_decay',1e-6,3e-3,log=True),
        'gradient_clip_norm':trial.suggest_categorical('gradient_clip_norm',[.5,1.,2.]),
        'warmup_epochs':trial.suggest_categorical('warmup_epochs',[0,3,5]),
        'batch_size':trial.suggest_categorical('batch_size',[64,128,256] if domain=='uci' else [128,256,512]),
        'prefix_sampling':'record_uniform' if domain=='uci' else trial.suggest_categorical('prefix_sampling',['record_uniform','stage_balanced']),
    }
    return result


def model_kwargs(params: dict) -> dict:
    return {key:params[key] for key in ('d_model','cnn_channels','cnn_blocks','bilstm_hidden','bilstm_layers','context_hidden','head_hidden','dropout')}


def training_config(params: dict) -> dict:
    return {'learning_rate':params['learning_rate'],'weight_decay':params['weight_decay'],
            'gradient_clip_norm':params['gradient_clip_norm'],'warmup_epochs':params['warmup_epochs'],
            'max_epochs':80,'patience':10,'eta_min':1e-5}


def parameter_count(temporal_dim: int, context_dim: int, params: dict) -> int:
    return sum(p.numel() for p in Hybrid(HybridConfig(temporal_dim,context_dim,**model_kwargs(params))).parameters())


def _scheduler(optimizer, params):
    warmup=params['warmup_epochs'];base=params['learning_rate']
    def factor(epoch):
        if warmup and epoch<warmup:return (epoch+1)/warmup
        progress=(epoch-warmup)/max(1,80-warmup)
        return 1e-5/base+(1-1e-5/base)*.5*(1+np.cos(np.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,factor)


def fit_hpo_fold(trial, fold_index, stages, fit_ids, stop_ids, valid_by_stage, contexts,
                 temporal_dim, context_dim, params, seed=42):
    seed_everything(seed);torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    model=Hybrid(HybridConfig(temporal_dim,context_dim,**model_kwargs(params))).cuda()
    count=sum(p.numel() for p in model.parameters())
    if count>2_000_000: raise optuna.TrialPruned(f'parameter_cap:{count}')
    optimizer=torch.optim.AdamW(model.parameters(),lr=params['learning_rate'],weight_decay=params['weight_decay']);scheduler=_scheduler(optimizer,params);scaler=torch.amp.GradScaler('cuda')
    y=np.asarray([next(data.target[data.index[r]] for data in stages.values() if r in data.index) for r in fit_ids]);loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=binary_pos_weight(y).cuda());available=eligible_prefixes(stages,fit_ids);best=-1.;best_epoch=1;best_state=None;stale=0
    sampler=sample_prefixes_stage_balanced if params['prefix_sampling']=='stage_balanced' else sample_prefixes
    for epoch in range(80):
        chosen=sampler(fit_ids,available,seed,epoch);arrays=assemble(stages,fit_ids,chosen,contexts);model.train()
        for temporal,mask,lengths,context,target in _loader(arrays,params['batch_size'],seed+epoch,True):
            temporal=temporal.cuda(non_blocking=True);mask=mask.cuda(non_blocking=True);lengths=lengths.cuda(non_blocking=True);context=context.cuda(non_blocking=True);target=target.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=loss_fn(model(temporal,mask,lengths,context),target)
            if not torch.isfinite(loss):raise optuna.TrialPruned('non_finite_loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise optuna.TrialPruned('non_finite_gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),params['gradient_clip_norm']);scaler.step(optimizer);scaler.update()
        scheduler.step();per_stop={}
        for stage,data in stages.items():
            ids=[r for r in stop_ids if r in data.index]
            if ids:
                targets=np.asarray([data.target[data.index[r]] for r in ids]);scores=predict(model,stages,stage,ids,contexts,params['batch_size'])
                if len(np.unique(targets))==2:per_stop[stage]=float(average_precision_score(targets,scores))
        current=equal_weight_macro_pr_auc(per_stop)
        trial.report(current,fold_index*80+epoch)
        if epoch>=10 and trial.should_prune():raise optuna.TrialPruned('median_pruner')
        if current>best:best=current;best_epoch=epoch+1;best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()});stale=0
        else:stale+=1
        if stale>=10:break
    model.load_state_dict(best_state);per_valid={}
    for stage,ids in valid_by_stage.items():
        data=stages[stage];targets=np.asarray([data.target[data.index[r]] for r in ids]);scores=predict(model,stages,stage,ids,contexts,params['batch_size']);per_valid[stage]=float(average_precision_score(targets,scores))
    peak=int(torch.cuda.max_memory_allocated());elapsed=time.monotonic()-started
    del model,optimizer,scheduler,scaler,loss_fn;gc.collect();torch.cuda.empty_cache()
    return per_valid,best_epoch,peak,elapsed,count
