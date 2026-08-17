"""Phase 6H validated multi-task gradient combiners and diagnostics."""
from __future__ import annotations
import copy,time
import numpy as np
import torch
from scipy.optimize import minimize
from libauc.optimizers import SOAP
from src.hybrid.models import SharedHeadHybrid
from src.hybrid.optimization.phase6 import STAGES,_metrics,class_pos_weight,predict
from src.hybrid.optimization.phase6c import multistage_arrays
from src.hybrid.optimization.phase6e import shared_config
from src.hybrid.optimization.phase6f import make_stage_ap_losses,positive_safe_batches,recall_constrained_threshold,ap_blended_stage_loss
from src.hybrid.training.evaluation import binary_classification_metrics
from src.hybrid.training.trainer import _scheduler,seed_everything

METHODS=("gradnorm","cagrad");CAGRAD_ALPHAS=(.1,.2,.4,.6,.8);LR_MULTIPLIERS=(.5,.75,1.,1.25,1.5);WD_MULTIPLIERS=(.5,1.,2.);PATIENCES=(6,10,14);CLIPS=(.5,1.,2.);GRADNORM_ALPHAS=(.5,1.,1.5,2.)

def baseline_margins(stage_pr,anchors):
 d={s:float(stage_pr[s])-float(anchors[s]) for s in STAGES["uci"]};return {"deltas":d,"worst_stage_margin":min(d.values()),"mean_baseline_margin":float(np.mean(list(d.values())))}
def _flat(grads,params):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,grads)])
def _assign(vector,params):
 cursor=0
 for p in params:n=p.numel();p.grad=vector[cursor:cursor+n].view_as(p).clone();cursor+=n
def gradient_matrix(losses,params,create_graph=False):return torch.stack([_flat(torch.autograd.grad(loss,params,retain_graph=True,create_graph=create_graph,allow_unused=True),params) for loss in losses],dim=1)
def cagrad_combine(gradients,alpha,rescale=1):
 """Canonical CAGrad conflict-averse direction for task-gradient columns."""
 tasks=gradients.shape[1];g0=gradients.mean(1);g0_norm=torch.sqrt(torch.dot(g0,g0)+1e-12);A=(gradients.t()@gradients).detach().double().cpu().numpy();b=np.ones(tasks)/tasks;c=float(alpha*g0_norm)
 result=minimize(lambda x:float(x@A@b+c*np.sqrt(x@A@x+1e-12)),b,bounds=[(0.,1.)]*tasks,constraints={"type":"eq","fun":lambda x:x.sum()-1.},method="SLSQP")
 if not result.success:raise RuntimeError("CAGRAD_SOLVER_FAILED:"+result.message)
 w=torch.as_tensor(result.x,dtype=gradients.dtype,device=gradients.device);gw=gradients@w;lam=c/(float(torch.linalg.vector_norm(gw))+1e-12);combined=g0+lam*gw
 return combined/(1+alpha**2) if rescale==1 else combined

class GradNormState:
 def __init__(self,device,alpha,lr):
  self.weights=torch.nn.Parameter(torch.ones(3,device=device));self.optimizer=torch.optim.Adam([self.weights],lr=lr);self.alpha=float(alpha);self.initial=None
 def combine(self,losses,params):
  values=torch.stack(losses)
  if self.initial is None:self.initial=values.detach().clamp_min(1e-8)
  norms=[]
  # ||grad(w_i * L_i)|| = w_i ||grad(L_i)|| for positive scalar task
  # weights. Detaching the base norm keeps only the task-weight derivative.
  for i,loss in enumerate(losses):
   base_norm=torch.linalg.vector_norm(_flat(torch.autograd.grad(loss,params,retain_graph=True,create_graph=False,allow_unused=True),params)).detach()
   norms.append(self.weights[i]*base_norm)
  norms=torch.stack(norms);rates=(values.detach()/self.initial);rates=rates/rates.mean();target=norms.detach().mean()*rates.pow(self.alpha);objective=torch.abs(norms-target).sum();self.optimizer.zero_grad(set_to_none=True);weight_grad=torch.autograd.grad(objective,self.weights,retain_graph=True)[0]
  model_loss=(self.weights.detach()*values).sum()/self.weights.detach().sum();grads=_flat(torch.autograd.grad(model_loss,params,allow_unused=True),params);self.weights.grad=weight_grad;return grads
 def step(self):
  self.optimizer.step()
  with torch.no_grad():self.weights.clamp_(min=1e-6);self.weights.mul_(3./self.weights.sum())

def multitask_step(losses,model,optimizer,method,state,clip):
 params=[p for p in model.parameters() if p.requires_grad];optimizer.zero_grad(set_to_none=True)
 if method=="gradnorm":vector=state.combine(losses,params)
 elif method=="cagrad":vector=cagrad_combine(gradient_matrix(losses,params),state)
 else:raise ValueError(method)
 _assign(vector,params);norm=torch.nn.utils.clip_grad_norm_(params,clip)
 if not torch.isfinite(norm):raise RuntimeError("NONFINITE_MULTITASK_GRADIENT")
 optimizer.step()
 if method=="gradnorm":state.step()
 return float(norm)

def _target(stages,stage,ids):d=stages[stage];return np.asarray([d.target[d.index[r]] for r in ids],dtype=int)
def _tensors(stages,ids,contexts):return {s:tuple(torch.from_numpy(x) for x in a) for s,a in multistage_arrays(stages,ids,contexts).items()}
def _stage_bce(model,tensors,batch,loss_fn):
 out=[]
 for s in STAGES["uci"]:
  temporal,mask,lengths,context,y,stage_index=(x[batch].cuda(non_blocking=True) for x in tensors[s])
  with torch.autocast("cuda",dtype=torch.float16):out.append(loss_fn(model(temporal,mask,lengths,context,stage_index),y))
 return out
def _new_state(method,params):return GradNormState("cuda",params["gradnorm_alpha"],params["task_weight_lr"]) if method=="gradnorm" else params["cagrad_alpha"]

def fit_method_development_fold(stages,fit_ids,stop_ids,valid_by_stage,contexts,temporal_dim,context_dim,base,ap_base,method,params,seed):
 valid_ids=set().union(*map(set,valid_by_stage.values()))
 if set(fit_ids)&set(stop_ids) or (set(fit_ids)|set(stop_ids))&valid_ids:raise RuntimeError("Phase6H split leakage")
 seed_everything(seed);torch.cuda.reset_peak_memory_stats();started=time.monotonic();model=SharedHeadHybrid(shared_config("uci",temporal_dim,context_dim,base)).cuda();count=sum(p.numel() for p in model.parameters())
 if count!=494795:raise RuntimeError(f"Phase6H parameter count changed:{count}")
 optimizer=torch.optim.AdamW(model.parameters(),lr=base["learning_rate"],weight_decay=base["weight_decay"]);scheduler=_scheduler(optimizer,80,base.get("warmup_epochs",0),1e-5,base["learning_rate"]);target=_target(stages,"S0",fit_ids);bce=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_pos_weight(target,"full")],device="cuda"));tensors=_tensors(stages,fit_ids,contexts);state=_new_state(method,params);best,best_state,best_epoch,stale=-np.inf,None,0,0
 for epoch in range(80):
  order=np.random.default_rng(seed+epoch).permutation(len(fit_ids));model.train()
  for start in range(0,len(order),base["batch_size"]):batch=order[start:start+base["batch_size"]];multitask_step(_stage_bce(model,tensors,batch,bce),model,optimizer,method,state,params["gradient_clip"],)
  scheduler.step();stop_by={s:[r for r in stop_ids if r in stages[s].index] for s in STAGES["uci"]};m=_metrics(model,"uci",stages,stop_by,contexts,base["batch_size"]);macro=float(np.mean([m[s]["pr_auc"] for s in STAGES["uci"]]))
  if macro>best:best,best_epoch,best_state,stale=macro,epoch+1,copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()}),0
  else:stale+=1
  if stale>=params["patience"]:break
 model.load_state_dict(best_state);ap_params=dict(ap_base);ap_params["ap_lr"]*=params["lr_multiplier"];ap_params["ap_weight_decay"]*=params["wd_multiplier"];ap_params["gradient_clip"]=params["gradient_clip"];soap=SOAP(model.parameters(),lr=ap_params["ap_lr"],mode="adam",clip_value=ap_params["gradient_clip"],weight_decay=ap_params["ap_weight_decay"],device="cuda",verbose=False);ap=make_stage_ap_losses(len(fit_ids),"cuda",ap_params["gamma"]);state=_new_state(method,params);best,best_state,ap_epoch=-np.inf,None,0
 for epoch in range(ap_params["ap_epochs"]):
  model.train()
  for batch in positive_safe_batches(target,base["batch_size"],seed,epoch):
   index=torch.as_tensor(batch,dtype=torch.long,device="cuda");losses=[]
   for s in STAGES["uci"]:
    temporal,mask,lengths,context,y,stage_index=(x[batch].cuda(non_blocking=True) for x in tensors[s]);logits=model(temporal,mask,lengths,context,stage_index);losses.append(ap_blended_stage_loss(logits,y,index,ap[s],bce,ap_params["alpha"]))
   multitask_step(losses,model,soap,method,state,ap_params["gradient_clip"])
  stop_by={s:[r for r in stop_ids if r in stages[s].index] for s in STAGES["uci"]};m=_metrics(model,"uci",stages,stop_by,contexts,base["batch_size"]);macro=float(np.mean([m[s]["pr_auc"] for s in STAGES["uci"]]))
  if macro>best:best,ap_epoch,best_state=macro,epoch+1,copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
 model.load_state_dict(best_state);train_by={s:[r for r in fit_ids if r in stages[s].index] for s in STAGES["uci"]};stop_by={s:[r for r in stop_ids if r in stages[s].index] for s in STAGES["uci"]};train_m=_metrics(model,"uci",stages,train_by,contexts,base["batch_size"]);rows=[]
 for s in STAGES["uci"]:
  ss=predict(model,"uci",stages,s,stop_by[s],contexts,base["batch_size"]);vs=predict(model,"uci",stages,s,valid_by_stage[s],contexts,base["batch_size"]);selected,feasible,reason=recall_constrained_threshold(_target(stages,s,stop_by[s]),ss,s);vm=binary_classification_metrics(_target(stages,s,valid_by_stage[s]),vs,threshold=selected["threshold"]);rows.append({"stage":s,**vm,"selected_threshold":selected["threshold"],"stop_threshold_feasible":feasible,"threshold_reason":reason,"stop_metrics":selected,"train_pr_auc":train_m[s]["pr_auc"],"validation_pr_auc":vm["pr_auc"],"train_validation_gap":train_m[s]["pr_auc"]-vm["pr_auc"]})
 train_macro=float(np.mean([train_m[s]["pr_auc"] for s in STAGES["uci"]]));valid_macro=float(np.mean([r["pr_auc"] for r in rows]));return {"pretrain_best_epoch":best_epoch,"ap_best_epoch":ap_epoch,"parameter_count":count,"train_macro_pr_auc":train_macro,"validation_macro_pr_auc":valid_macro,"train_validation_gap":train_macro-valid_macro,"runtime_seconds":time.monotonic()-started,"peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated()),"rows":rows,"task_weights":state.weights.detach().cpu().tolist() if method=="gradnorm" else None}

def gradient_diagnostic_batch(model,losses):
 params=[p for p in model.parameters() if p.requires_grad];G=gradient_matrix(losses,params).detach();norms=torch.linalg.vector_norm(G,dim=0);row={f"norm_{s}":float(norms[i]) for i,s in enumerate(STAGES["uci"])}
 for i,a in enumerate(STAGES["uci"]):
  for j,b in enumerate(STAGES["uci"]):
   if i<j:row[f"cos_{a}_{b}"]=float(torch.dot(G[:,i],G[:,j])/(norms[i]*norms[j]+1e-12))
 row["norm_ratio"]=float(norms.max()/(norms.min()+1e-12));return row
