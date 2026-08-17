"""Phase 6G SAM pretraining and uniform checkpoint averaging contracts."""
from __future__ import annotations
import copy, time
from pathlib import Path
import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel
from libauc.optimizers import SOAP

from src.hybrid.models import SharedHeadHybrid
from src.hybrid.optimization.phase6 import STAGES,_metrics,class_pos_weight,predict
from src.hybrid.optimization.phase6c import multistage_arrays
from src.hybrid.optimization.phase6e import shared_config
from src.hybrid.optimization.phase6f import make_stage_ap_losses,positive_safe_batches,recall_constrained_threshold,ap_blended_stage_loss
from src.hybrid.training.evaluation import binary_classification_metrics
from src.hybrid.training.trainer import _scheduler,seed_everything

SAM_GRID=(None,.01,.03,.05)
WA_GRID=(None,.40,.60,.80)
GRID={f"G{i:02d}":{"sam_rho":rho,"wa_start":wa} for i,(rho,wa) in enumerate((r,w) for r in SAM_GRID for w in WA_GRID)}
SCREEN_SEEDS=(42,1201);ROBUST_SEED=2026

class SAMController:
    """Canonical perturb/restore controller around an unchanged base optimizer."""
    def __init__(self,params,rho):self.params=list(params);self.rho=float(rho);self.perturbations={}
    @torch.no_grad()
    def perturb(self):
        grads=[p.grad for p in self.params if p.grad is not None]
        norm=torch.norm(torch.stack([g.norm(2) for g in grads]),2) if grads else torch.tensor(0.)
        scale=self.rho/(norm+1e-12);self.perturbations={}
        for p in self.params:
            if p.grad is not None:
                e=p.grad*scale.to(p);p.add_(e);self.perturbations[id(p)]=e
    @torch.no_grad()
    def restore(self):
        for p in self.params:
            if id(p) in self.perturbations:p.sub_(self.perturbations[id(p)])
        self.perturbations={}

def _target(stages,stage,ids):
    d=stages[stage];return np.asarray([d.target[d.index[r]] for r in ids],dtype=int)
def _tensors(stages,ids,contexts):return {s:tuple(torch.from_numpy(x) for x in a) for s,a in multistage_arrays(stages,ids,contexts).items()}

def _stage_bce_losses(model,tensors,batch,loss_fn):
    losses=[]
    for stage in STAGES["uci"]:
        temporal,mask,lengths,context,target,stage_index=(x[batch].cuda(non_blocking=True) for x in tensors[stage])
        with torch.autocast("cuda",dtype=torch.float16):logits=model(temporal,mask,lengths,context,stage_index);losses.append(loss_fn(logits,target))
    return torch.stack(losses).mean()

def fit_sam_pretrain_fold(stages,fit_ids,stop_ids,valid_by_stage,contexts,temporal_dim,context_dim,params,seed,rho,checkpoint_path,max_epochs=80,patience=10):
    if rho not in SAM_GRID[1:]:raise ValueError("SAM rho must be enabled Phase6G value")
    valid_ids=set().union(*map(set,valid_by_stage.values()))
    if set(fit_ids)&set(stop_ids) or (set(fit_ids)|set(stop_ids))&valid_ids:raise RuntimeError("Phase6G split leakage")
    seed_everything(seed);torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    model=SharedHeadHybrid(shared_config("uci",temporal_dim,context_dim,params)).cuda();nparams=sum(p.numel() for p in model.parameters())
    if nparams!=494795:raise RuntimeError(f"Phase6G parameter count changed:{nparams}")
    optimizer=torch.optim.AdamW(model.parameters(),lr=params["learning_rate"],weight_decay=params["weight_decay"])
    scheduler=_scheduler(optimizer,max_epochs,params.get("warmup_epochs",0),1e-5,params["learning_rate"])
    target=_target(stages,"S0",fit_ids);weight=class_pos_weight(target,"full");loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight],device="cuda"))
    tensors=_tensors(stages,fit_ids,contexts);scaler=torch.amp.GradScaler("cuda");sam=SAMController(model.parameters(),rho)
    best,best_epoch,best_state,stale=-np.inf,0,None,0;loss_evaluations=0;scheduler_steps=0
    for epoch in range(max_epochs):
        rng=np.random.default_rng(seed+epoch);order=rng.permutation(len(fit_ids));model.train()
        for start in range(0,len(order),params["batch_size"]):
            batch=order[start:start+params["batch_size"]]
            optimizer.zero_grad(set_to_none=True);first=_stage_bce_losses(model,tensors,batch,loss_fn);loss_evaluations+=1
            scaler.scale(first).backward();inverse_scale=1.0/scaler.get_scale()
            for parameter in model.parameters():
                if parameter.grad is not None:parameter.grad.mul_(inverse_scale)
            torch.nn.utils.clip_grad_norm_(model.parameters(),params["gradient_clip_norm"]);sam.perturb();optimizer.zero_grad(set_to_none=True)
            second=_stage_bce_losses(model,tensors,batch,loss_fn);loss_evaluations+=1
            scaler.scale(second).backward();scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),params["gradient_clip_norm"]);sam.restore()
            if not torch.isfinite(norm):raise RuntimeError("non_finite_sam_gradient")
            scaler.step(optimizer);scaler.update()
        scheduler.step();scheduler_steps+=1
        stop_by={s:[r for r in stop_ids if r in stages[s].index] for s in STAGES["uci"]};m=_metrics(model,"uci",stages,stop_by,contexts,params["batch_size"]);macro=float(np.mean([m[s]["pr_auc"] for s in STAGES["uci"]]))
        if macro>best:best,best_epoch,best_state,stale=macro,epoch+1,copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()}),0
        else:stale+=1
        if stale>=patience:break
    payload={"model_state":best_state,"best_epoch":best_epoch,"parameter_count":nparams,"seed":seed,"params":params,"sam_rho":rho}
    torch.save(payload,checkpoint_path)
    return {"best_epoch":best_epoch,"parameter_count":nparams,"runtime_seconds":time.monotonic()-started,"peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated()),"loss_evaluations":loss_evaluations,"scheduler_steps":scheduler_steps}

def fine_tune_with_uniform_average(stages,fit_ids,stop_ids,valid_by_stage,contexts,temporal_dim,context_dim,pretrain_params,checkpoint,fine_params,seed,wa_start):
    valid_ids=set().union(*map(set,valid_by_stage.values()))
    if set(fit_ids)&set(stop_ids) or (set(fit_ids)|set(stop_ids))&valid_ids:raise RuntimeError("Phase6G split leakage")
    seed_everything(seed);torch.cuda.reset_peak_memory_stats();started=time.monotonic();model=SharedHeadHybrid(shared_config("uci",temporal_dim,context_dim,pretrain_params)).cuda();model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)["model_state"])
    if sum(p.numel() for p in model.parameters())!=494795:raise RuntimeError("Phase6G architecture changed")
    optimizer=SOAP(model.parameters(),lr=fine_params["ap_lr"],mode="adam",clip_value=fine_params["gradient_clip"],weight_decay=fine_params["ap_weight_decay"],device="cuda",verbose=False)
    target=_target(stages,"S0",fit_ids);bce=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_pos_weight(target,"full")],device="cuda"));ap=make_stage_ap_losses(len(fit_ids),"cuda",fine_params["gamma"]);tensors=_tensors(stages,fit_ids,contexts)
    averaged=AveragedModel(model,use_buffers=True) if wa_start is not None else None;included=0;best,best_epoch,best_state=-np.inf,0,None;epochs=int(fine_params["ap_epochs"])
    for epoch in range(epochs):
        model.train()
        for batch in positive_safe_batches(target,pretrain_params["batch_size"],seed,epoch):
            index=torch.as_tensor(batch,dtype=torch.long,device="cuda");optimizer.zero_grad(set_to_none=True);losses=[]
            for stage in STAGES["uci"]:
                temporal,mask,lengths,context,y,stage_index=(x[batch].cuda(non_blocking=True) for x in tensors[stage]);logits=model(temporal,mask,lengths,context,stage_index);losses.append(ap_blended_stage_loss(logits,y,index,ap[stage],bce,fine_params["alpha"]))
            loss=torch.stack(losses).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),fine_params["gradient_clip"]);optimizer.step()
        if averaged is not None and (epoch+1)/epochs>=wa_start:averaged.update_parameters(model);included+=1
        candidate=averaged if averaged is not None and included else model;stop_by={s:[r for r in stop_ids if r in stages[s].index] for s in STAGES["uci"]};m=_metrics(candidate,"uci",stages,stop_by,contexts,pretrain_params["batch_size"]);macro=float(np.mean([m[s]["pr_auc"] for s in STAGES["uci"]]))
        if macro>best:best,best_epoch=macro,epoch+1;source=candidate.module if isinstance(candidate,AveragedModel) else candidate;best_state=copy.deepcopy({k:v.detach().cpu() for k,v in source.state_dict().items()})
    model.load_state_dict(best_state);train_by={s:[r for r in fit_ids if r in stages[s].index] for s in STAGES["uci"]};stop_by={s:[r for r in stop_ids if r in stages[s].index] for s in STAGES["uci"]};train_m=_metrics(model,"uci",stages,train_by,contexts,pretrain_params["batch_size"]);stop_m=_metrics(model,"uci",stages,stop_by,contexts,pretrain_params["batch_size"]);rows=[]
    for stage in STAGES["uci"]:
        ss=predict(model,"uci",stages,stage,stop_by[stage],contexts,pretrain_params["batch_size"]);vs=predict(model,"uci",stages,stage,valid_by_stage[stage],contexts,pretrain_params["batch_size"]);selected,feasible,reason=recall_constrained_threshold(_target(stages,stage,stop_by[stage]),ss,stage);vm=binary_classification_metrics(_target(stages,stage,valid_by_stage[stage]),vs,threshold=selected["threshold"])
        rows.append({"stage":stage,**vm,"selected_threshold":selected["threshold"],"stop_threshold_feasible":feasible,"threshold_reason":reason,"stop_metrics":selected,"train_pr_auc":train_m[stage]["pr_auc"],"stop_pr_auc":stop_m[stage]["pr_auc"],"validation_pr_auc":vm["pr_auc"],"train_validation_gap":train_m[stage]["pr_auc"]-vm["pr_auc"],"train_stop_gap":train_m[stage]["pr_auc"]-stop_m[stage]["pr_auc"],"stop_validation_gap":stop_m[stage]["pr_auc"]-vm["pr_auc"]})
    train_macro=float(np.mean([train_m[s]["pr_auc"] for s in STAGES["uci"]]));valid_macro=float(np.mean([r["pr_auc"] for r in rows]));return {"best_epoch":best_epoch,"averaged_checkpoints":included,"parameter_count":494795,"train_macro_pr_auc":train_macro,"validation_macro_pr_auc":valid_macro,"train_validation_gap":train_macro-valid_macro,"runtime_seconds":time.monotonic()-started,"peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated()),"rows":rows}
