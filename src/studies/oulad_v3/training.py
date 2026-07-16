from __future__ import annotations

import copy, gc, time
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .data import OULADV3Data
from .models import PreparedInputs, V3Preprocessors, build_model, prepare_inputs, set_deterministic_seed, state_dict_sha256

@dataclass
class FitResult:
    probabilities: np.ndarray; selected_epoch: int; epochs_ran: int; history: list[dict[str, Any]]
    parameter_count: int; runtime_seconds: float; state_dict: dict[str, torch.Tensor]; state_dict_sha256: str
    preprocessors: V3Preprocessors; reproduction_max_abs_difference: float; device: str
    attention_entropy_mean: float | None; attention_padding_max: float | None

def _dataset(x: PreparedInputs):
    return TensorDataset(*[torch.from_numpy(v) for v in [x.sequence,x.lengths,x.mask,x.aggregate,x.static,x.target]])

def _loader(x, batch_size, shuffle, seed, device):
    generator=torch.Generator(); generator.manual_seed(seed)
    return DataLoader(_dataset(x),batch_size=batch_size,shuffle=shuffle,num_workers=0,generator=generator,
                      pin_memory=device.type=="cuda",drop_last=False)

def predict(model, inputs, batch_size, device, diagnostics=False):
    model.eval(); probabilities=[]; entropies=[]; padding=[]
    with torch.no_grad():
        for sequence,lengths,mask,aggregate,static,_ in _loader(inputs,batch_size,False,0,device):
            result=model(sequence.to(device),lengths.to(device),mask.to(device),aggregate.to(device),static.to(device),return_attention=diagnostics)
            logits, attention = result if diagnostics else (result,None)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            if attention is not None:
                valid=mask.to(device).bool(); p=attention.clamp_min(1e-12)
                entropies.extend((-(p*torch.log(p))*valid).sum(1).cpu().tolist())
                if (~valid).any(): padding.append(float(attention.masked_select(~valid).max().cpu()))
    p=np.concatenate(probabilities).astype(float)
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any(): raise RuntimeError("Probability contract failed")
    return p,(float(np.mean(entropies)) if entropies else None),(max(padding) if padding else 0.0 if diagnostics else None)

def _nll(y,p):
    p=np.clip(p,1e-7,1-1e-7); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def _positive_weight(policy,y):
    ratio=float((len(y)-y.sum())/max(y.sum(),1))
    return {"none":1.0,"sqrt_balanced":float(np.sqrt(ratio)),"fully_balanced":ratio}[policy]

def fit_candidate(data: OULADV3Data,candidate_id:str,train_indices,evaluation_indices,*,temporal_config,
                  aggregate_config,seed:int,fixed_epochs:int|None=None,device_name:str|None=None)->FitResult:
    started=time.perf_counter(); config=dict(temporal_config or aggregate_config)
    max_epochs=int(config.get("max_epochs",40)); patience=int(config.get("patience",6)); batch_size=int(config["batch_size"])
    set_deterministic_seed(seed); torch.set_num_threads(min(6,max(1,torch.get_num_threads())))
    device=torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    train=prepare_inputs(data,train_indices,train_indices,candidate_id); evaluation=prepare_inputs(data,train_indices,evaluation_indices,candidate_id,train.preprocessors)
    model=build_model(candidate_id,train,temporal_config,aggregate_config).to(device)
    count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count>=300000: raise RuntimeError(f"Parameter guardrail exceeded: {count}")
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(_positive_weight(config["positive_weight"],train.target),device=device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(config["learning_rate"]),weight_decay=float(config["weight_decay"]))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max_epochs) if config.get("scheduler")=="deterministic_cosine" else None
    if config.get("scheduler","fixed_lr") not in {"fixed_lr","deterministic_cosine"}: raise RuntimeError("Non-replayable scheduler")
    best_state=copy.deepcopy(model.state_dict()); best_loss=float("inf"); best_epoch=1; stale=0; history=[]
    for epoch in range(1,int(fixed_epochs or max_epochs)+1):
        model.train(); losses=[]; norms=[]
        for sequence,lengths,mask,aggregate,static,target in _loader(train,batch_size,True,seed+epoch,device):
            optimizer.zero_grad(set_to_none=True)
            logits=model(sequence.to(device),lengths.to(device),mask.to(device),aggregate.to(device),static.to(device))
            loss=criterion(logits,target.to(device));
            if not torch.isfinite(loss): raise RuntimeError("Non-finite training loss")
            loss.backward(); norm=nn.utils.clip_grad_norm_(model.parameters(),float(config.get("gradient_clip",1.0))); optimizer.step()
            losses.append(float(loss.detach().cpu())); norms.append(float(norm.detach().cpu()))
        if scheduler: scheduler.step()
        row={"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_nll":None,"learning_rate":float(optimizer.param_groups[0]["lr"]),"gradient_norm_mean":float(np.mean(norms))}
        if fixed_epochs is None:
            p,_,_=predict(model,evaluation,batch_size,device); loss=_nll(evaluation.target,p); row["validation_nll"]=loss
            if loss<best_loss-1e-5: best_loss=loss; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict()); stale=0
            else: stale+=1
        else: best_epoch=epoch; best_state=copy.deepcopy(model.state_dict())
        history.append(row)
        if fixed_epochs is None and stale>=patience: break
    model.load_state_dict(best_state); p,entropy,padding=predict(model,evaluation,batch_size,device,True)
    cpu_state={k:v.detach().cpu().clone() for k,v in best_state.items()}; state_hash=state_dict_sha256(cpu_state)
    replay=build_model(candidate_id,train,temporal_config,aggregate_config).to(device); replay.load_state_dict(cpu_state)
    replay_p,_,_=predict(replay,evaluation,batch_size,device); difference=float(np.max(np.abs(p-replay_p)))
    if difference>1e-7: raise RuntimeError(f"Checkpoint reproduction failed: {difference}")
    runtime=time.perf_counter()-started; del replay,model
    if device.type=="cuda": torch.cuda.empty_cache()
    gc.collect()
    return FitResult(p,best_epoch,len(history),history,count,runtime,cpu_state,state_hash,train.preprocessors,difference,str(device),entropy,padding)
