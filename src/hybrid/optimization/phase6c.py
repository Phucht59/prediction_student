"""Phase 6C UCI full multi-stage supervision on inner-development data only."""
from __future__ import annotations

import copy
import gc
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.hybrid.optimization.phase6 import (
    ExponentialMovingAverage,
    STAGES,
    _arrays,
    _metrics,
    class_pos_weight,
    deterministic_pairwise_loss,
    model_config,
    predict,
)
from src.hybrid.models import StageConditionedHybrid
from src.hybrid.training.trainer import _scheduler, seed_everything


def multistage_arrays(stages, record_ids, contexts):
    """Return aligned S0/S1/S2 arrays without increasing dataset length."""
    if tuple(stages) != STAGES["uci"]:
        raise ValueError("Phase6C accepts UCI S0/S1/S2 only")
    result={stage:_arrays(stages,record_ids,[stage]*len(record_ids),contexts,"uci") for stage in STAGES["uci"]}
    targets=[result[stage][4] for stage in STAGES["uci"]]
    if not all(np.array_equal(targets[0],target) for target in targets[1:]):
        raise RuntimeError("Stage targets differ for identical UCI records")
    return result


def average_stage_loss(stage_losses):
    if len(stage_losses)!=3:
        raise ValueError("Phase6C requires exactly three stage losses")
    return torch.stack(tuple(stage_losses)).mean()


def perform_optimizer_update(stage_losses, model, optimizer, clip, scaler=None):
    """Perform exactly one optimizer update for one original record batch."""
    optimizer.zero_grad(set_to_none=True);loss=average_stage_loss(stage_losses)
    if scaler is None:
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),clip);optimizer.step()
    else:
        scaler.scale(loss).backward();scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
        if not torch.isfinite(norm):raise RuntimeError("non_finite_multistage_gradient")
        scaler.step(optimizer);scaler.update()
    return loss


def _tensors(arrays_by_stage):
    return {stage:tuple(torch.from_numpy(value) for value in arrays) for stage,arrays in arrays_by_stage.items()}


def fit_multistage_development_fold(stages,fit_ids,stop_ids,valid_by_stage,contexts,temporal_dim,context_dim,params,seed=42,max_epochs=80,patience=10,model_factory=StageConditionedHybrid,config_factory=model_config,expected_parameter_count=282481,parameter_cap=None,checkpoint_path=None):
    """Train one UCI Hybrid with equal S0/S1/S2 loss per optimizer step."""
    if set(fit_ids)&set(stop_ids):raise RuntimeError("fit/early-stop record leakage")
    valid_ids=set().union(*map(set,valid_by_stage.values()))
    if (set(fit_ids)|set(stop_ids))&valid_ids:raise RuntimeError("training/evaluation record leakage")
    seed_everything(seed);torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    model=model_factory(config_factory("uci",temporal_dim,context_dim,params)).cuda();nparams=sum(p.numel() for p in model.parameters())
    if expected_parameter_count is not None and nparams!=expected_parameter_count:raise RuntimeError(f"Phase6C parameter count changed: {nparams}")
    if parameter_cap is not None and nparams>parameter_cap:raise RuntimeError(f"Phase6E parameter cap exceeded: {nparams}")
    optimizer=torch.optim.AdamW(model.parameters(),lr=params["learning_rate"],weight_decay=params["weight_decay"])
    scheduler=_scheduler(optimizer,max_epochs,params.get("warmup_epochs",0),1e-5,params["learning_rate"])
    target=np.asarray([stages["S0"].target[stages["S0"].index[record]] for record in fit_ids]);weight=class_pos_weight(target,params["class_weight_mode"])
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight],device="cuda"));scaler=torch.amp.GradScaler("cuda");ema=ExponentialMovingAverage(model,params.get("ema_decay"))
    stage_tensors=_tensors(multistage_arrays(stages,fit_ids,contexts));best,best_epoch,best_state,stale=-1.0,1,None,0;optimizer_steps=0
    for epoch in range(max_epochs):
        indices=TensorDataset(torch.arange(len(fit_ids)))
        loader=DataLoader(indices,batch_size=params["batch_size"],shuffle=True,generator=torch.Generator().manual_seed(seed+epoch),pin_memory=True)
        model.train()
        for batch_index,(index,) in enumerate(loader):
            stage_losses=[]
            for stage_number,stage in enumerate(STAGES["uci"]):
                temporal,mask,lengths,context,batch_target,stage_index=(tensor[index].cuda(non_blocking=True) for tensor in stage_tensors[stage])
                with torch.autocast("cuda",dtype=torch.float16):
                    logits=model(temporal,mask,lengths,context,stage_index)
                    bce=loss_fn(logits,batch_target)
                    rank=deterministic_pairwise_loss(logits.float(),batch_target,seed+epoch*10000+batch_index*3+stage_number)
                    stage_losses.append(bce+float(params["lambda_rank"])*rank)
            loss=perform_optimizer_update(stage_losses,model,optimizer,params["gradient_clip_norm"],scaler)
            if not torch.isfinite(loss):raise RuntimeError("non_finite_multistage_loss")
            ema.update(model);optimizer_steps+=1
        scheduler.step();raw_state=copy.deepcopy(model.state_dict());model.load_state_dict(ema.state_dict(model))
        stop_by={stage:[record for record in stop_ids if record in data.index] for stage,data in stages.items()}
        stop_metrics=_metrics(model,"uci",stages,stop_by,contexts,params["batch_size"]);macro=float(np.mean([row["pr_auc"] for row in stop_metrics.values()]));model.load_state_dict(raw_state)
        if macro>best:best,best_epoch,best_state,stale=macro,epoch+1,ema.state_dict(model),0
        else:stale+=1
        if stale>=patience:break
    if best_state is None:raise RuntimeError("no_finite_multistage_epoch")
    model.load_state_dict(best_state);train_by={stage:[record for record in fit_ids if record in data.index] for stage,data in stages.items()};stop_by={stage:[record for record in stop_ids if record in data.index] for stage,data in stages.items()}
    train_metrics=_metrics(model,"uci",stages,train_by,contexts,params["batch_size"]);stop_metrics=_metrics(model,"uci",stages,stop_by,contexts,params["batch_size"]);valid_metrics=_metrics(model,"uci",stages,valid_by_stage,contexts,params["batch_size"])
    predictions={"stop":{stage:predict(model,"uci",stages,stage,ids,contexts,params["batch_size"]) for stage,ids in stop_by.items()},"validation":{stage:predict(model,"uci",stages,stage,ids,contexts,params["batch_size"]) for stage,ids in valid_by_stage.items()}}
    train_macro=float(np.mean([row["pr_auc"] for row in train_metrics.values()]));valid_macro=float(np.mean([row["pr_auc"] for row in valid_metrics.values()]))
    result={"best_epoch":best_epoch,"parameter_count":nparams,"optimizer_steps":optimizer_steps,"records_per_epoch":len(fit_ids),"dataset_tripled":False,"train_macro_pr_auc":train_macro,"validation_macro_pr_auc":valid_macro,"train_validation_gap":train_macro-valid_macro,"per_stage_train":train_metrics,"per_stage_stop":stop_metrics,"per_stage_validation":valid_metrics,"runtime_seconds":time.monotonic()-started,"peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated()),"pos_weight":weight}
    if checkpoint_path is not None:
        torch.save({"model_state":best_state,"best_epoch":best_epoch,"parameter_count":nparams,"seed":seed,"params":params},checkpoint_path)
    del model,optimizer,scheduler,loss_fn,scaler,ema;gc.collect();torch.cuda.empty_cache();return result,predictions
