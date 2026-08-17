"""Deterministic CUDA/AMP training for the frozen Hybrid V1."""
from __future__ import annotations

from dataclasses import dataclass
import gc
import random
import time
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from src.hybrid.models import Hybrid, HybridConfig
from .data import sample_prefixes, sample_prefixes_stage_balanced
from .losses import binary_pos_weight


@dataclass
class StageData:
    record_id: np.ndarray
    group_id: np.ndarray
    target: np.ndarray
    temporal: np.ndarray
    mask: np.ndarray
    lengths: np.ndarray

    def __post_init__(self):
        self.index = {str(value): i for i, value in enumerate(self.record_id)}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def pad_stages(stages: dict[str, StageData]) -> dict[str, StageData]:
    max_t = max(item.temporal.shape[1] for item in stages.values())
    result = {}
    for name, item in stages.items():
        if item.temporal.shape[1] == max_t:
            result[name] = item; continue
        temporal = np.zeros((len(item.record_id), max_t, item.temporal.shape[2]), np.float32)
        mask = np.zeros((len(item.record_id), max_t), bool)
        temporal[:, :item.temporal.shape[1]] = item.temporal
        mask[:, :item.mask.shape[1]] = item.mask
        result[name] = StageData(item.record_id, item.group_id, item.target, temporal, mask, item.lengths)
    return result


def eligible_prefixes(stages: dict[str, StageData], record_ids: list[str]) -> list[list[str]]:
    return [[stage for stage, data in stages.items() if record_id in data.index] for record_id in record_ids]


def assemble(stages: dict[str, StageData], record_ids: list[str], selected: list[str], contexts: dict[str, np.ndarray]):
    temporal=[]; mask=[]; lengths=[]; target=[]; context=[]
    for record_id, stage in zip(record_ids, selected, strict=True):
        data=stages[stage]; i=data.index[record_id]
        temporal.append(data.temporal[i]); mask.append(data.mask[i]); lengths.append(data.lengths[i]); target.append(data.target[i]); context.append(contexts[record_id])
    return (np.stack(temporal).astype(np.float32), np.stack(mask), np.asarray(lengths,np.int64),
            np.stack(context).astype(np.float32), np.asarray(target,np.float32))


def _loader(arrays, batch_size: int, seed: int, shuffle: bool):
    tensors = (torch.from_numpy(arrays[0]), torch.from_numpy(arrays[1]), torch.from_numpy(arrays[2]),
               torch.from_numpy(arrays[3]), torch.from_numpy(arrays[4]))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), pin_memory=True)


@torch.no_grad()
def predict(model: Hybrid, stages: dict[str, StageData], stage: str, record_ids: list[str], contexts: dict[str,np.ndarray], batch_size: int) -> np.ndarray:
    model.eval(); data=stages[stage]; chosen=[stage]*len(record_ids); arrays=assemble(stages,record_ids,chosen,contexts); scores=[]
    for temporal,mask,lengths,context,_ in _loader(arrays,batch_size,42,False):
        with torch.autocast("cuda",dtype=torch.float16):
            logits=model(temporal.cuda(non_blocking=True),mask.cuda(non_blocking=True),lengths.cuda(non_blocking=True),context.cuda(non_blocking=True))
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(scores)


def _train_epoch(model, loader, optimizer, loss_fn, scaler, clip):
    model.train(); losses=[]
    for temporal,mask,lengths,context,target in loader:
        temporal=temporal.cuda(non_blocking=True); mask=mask.cuda(non_blocking=True); lengths=lengths.cuda(non_blocking=True); context=context.cuda(non_blocking=True); target=target.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.float16): loss=loss_fn(model(temporal,mask,lengths,context),target)
        if not torch.isfinite(loss): raise RuntimeError("Non-finite Hybrid loss")
        scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
        scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
    return float(np.mean(losses))


def _scheduler(optimizer, max_epochs, warmup_epochs, eta_min, base_lr):
    def factor(epoch):
        if warmup_epochs and epoch < warmup_epochs: return (epoch + 1) / warmup_epochs
        progress=(epoch-warmup_epochs)/max(1,max_epochs-warmup_epochs)
        return eta_min/base_lr+(1-eta_min/base_lr)*0.5*(1+np.cos(np.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,factor)


def _sample(ids, available, seed, epoch, mode):
    return sample_prefixes_stage_balanced(ids,available,seed,epoch) if mode=='stage_balanced' else sample_prefixes(ids,available,seed,epoch)


def select_best_epoch(stages, fit_ids, stop_ids, contexts, temporal_dim, context_dim, batch_size, config, seed=42, model_kwargs=None, prefix_sampling='record_uniform'):
    seed_everything(seed); torch.cuda.reset_peak_memory_stats()
    model=Hybrid(HybridConfig(temporal_dim,context_dim,**(model_kwargs or {}))).cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    scheduler=_scheduler(optimizer,config["max_epochs"],config.get("warmup_epochs",0),config.get("scheduler_eta_min",config.get("eta_min",1e-5)),config["learning_rate"])
    fit_target=np.asarray([next(data.target[data.index[r]] for data in stages.values() if r in data.index) for r in fit_ids])
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=binary_pos_weight(fit_target).cuda()); scaler=torch.amp.GradScaler("cuda")
    available=eligible_prefixes(stages,fit_ids); best=-1.0; best_epoch=1; stale=0; started=time.monotonic()
    for epoch in range(config["max_epochs"]):
        selected=_sample(fit_ids,available,seed,epoch,prefix_sampling); arrays=assemble(stages,fit_ids,selected,contexts)
        _train_epoch(model,_loader(arrays,batch_size,seed+epoch,True),optimizer,loss_fn,scaler,config["gradient_clip_norm"]); scheduler.step()
        aps=[]
        for stage,data in stages.items():
            ids=[r for r in stop_ids if r in data.index]
            if ids:
                y=np.asarray([data.target[data.index[r]] for r in ids]); score=predict(model,stages,stage,ids,contexts,batch_size)
                if len(np.unique(y))==2: aps.append(float(average_precision_score(y,score)))
        macro=float(np.mean(aps)) if aps else float("nan")
        if not np.isfinite(macro): raise RuntimeError("Early-stop macro PR-AUC is not finite")
        if macro>best: best=macro;best_epoch=epoch+1;stale=0
        else: stale+=1
        if stale>=config["patience"]: break
    peak=int(torch.cuda.max_memory_allocated()); runtime=time.monotonic()-started
    del model,optimizer,scheduler,loss_fn,scaler;gc.collect();torch.cuda.empty_cache()
    return best_epoch,best,runtime,peak


def refit_and_predict(stages, train_ids, valid_ids_by_stage, contexts, temporal_dim, context_dim, batch_size, config, epochs, seed=42, model_kwargs=None, prefix_sampling='record_uniform'):
    seed_everything(seed); torch.cuda.reset_peak_memory_stats(); started=time.monotonic()
    model=Hybrid(HybridConfig(temporal_dim,context_dim,**(model_kwargs or {}))).cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    scheduler=_scheduler(optimizer,config["max_epochs"],config.get("warmup_epochs",0),config.get("scheduler_eta_min",config.get("eta_min",1e-5)),config["learning_rate"])
    target=np.asarray([next(data.target[data.index[r]] for data in stages.values() if r in data.index) for r in train_ids])
    pos_weight=float(binary_pos_weight(target).item()); loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight],device="cuda")); scaler=torch.amp.GradScaler("cuda")
    available=eligible_prefixes(stages,train_ids)
    for epoch in range(epochs):
        chosen=_sample(train_ids,available,seed,epoch,prefix_sampling); arrays=assemble(stages,train_ids,chosen,contexts)
        _train_epoch(model,_loader(arrays,batch_size,seed+epoch,True),optimizer,loss_fn,scaler,config["gradient_clip_norm"]);scheduler.step()
    predictions={stage:predict(model,stages,stage,ids,contexts,batch_size) for stage,ids in valid_ids_by_stage.items() if ids}
    peak=int(torch.cuda.max_memory_allocated()); runtime=time.monotonic()-started; params=sum(p.numel() for p in model.parameters())
    del model,optimizer,scheduler,loss_fn,scaler;gc.collect();torch.cuda.empty_cache()
    return predictions,pos_weight,runtime,peak,params
