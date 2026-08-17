"""Phase7B inner-development execution adapters (no outer-test access)."""
from __future__ import annotations
import copy, os, time
from types import SimpleNamespace
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset
from src.hybrid.contracts import MaskedStandardScaler
from src.hybrid.data.uci import build_uci_combined,UCI_NUMERIC_CONTEXT,UCI_CATEGORICAL_CONTEXT
from src.hybrid.data.oulad import load_oulad_static_tables,build_compact_vle_daily,OULAD_NUMERIC_CONTEXT,OULAD_CATEGORICAL_CONTEXT
from src.hybrid.phase7.data import build_uci_phase7_view,build_oulad_phase7_view
from src.hybrid.phase7.model import UnifiedHybrid,UnifiedHybridConfig
from src.hybrid.training.data import ContextPreprocessor,sample_prefixes
from src.hybrid.training.evaluation import binary_classification_metrics
from src.hybrid.training.trainer import _scheduler,seed_everything
from src.hybrid.optimization.phase6b import stage_threshold_metrics
from src.hybrid.phase7.data import build_phase7_baseline_frame
from src.hybrid.models import Hybrid,HybridConfig,SharedHeadHybrid
from src.hybrid.optimization.phase4b import candidate_model_kwargs
from src.hybrid.optimization.phase6e import shared_config
from scripts.hybrid.run_hybrid_v1 import split_fit_stop,load_domain

ROOT=__import__('pathlib').Path(__file__).resolve().parents[3]
STAGES={'uci':('S0','S1','S2'),'oulad':('20pct','35pct','50pct','75pct')}
PROGRESS={'S0':0.,'S1':.5,'S2':1.,'20pct':.2,'35pct':.35,'50pct':.5,'75pct':.75}
_PHASE7_DOMAIN_CACHE={}
_OULAD_OUTCOME_CACHE=None

def _pad(v):
 t=max(x.temporal.shape[1] for x in v.values())
 for x in v.values():
  if x.temporal.shape[1]<t:
   object.__setattr__(x,'temporal',np.pad(x.temporal,((0,0),(0,t-x.temporal.shape[1]),(0,0))))
   object.__setattr__(x,'temporal_mask',np.pad(x.temporal_mask,((0,0),(0,t-x.temporal_mask.shape[1]))))
 return v

def _build_phase7_domain(domain):
 if domain=='uci':
  f,_=build_uci_combined(ROOT/'data/raw/student-mat.csv',ROOT/'data/raw/student-por.csv');f=f.rename(columns={'global_student_group':'group_id'})
  return _pad({s:build_uci_phase7_view(f.rename(columns={'group_id':'global_student_group'}),s) for s in STAGES[domain]}),f[['record_id','group_id','target']+UCI_NUMERIC_CONTEXT+UCI_CATEGORICAL_CONTEXT],UCI_NUMERIC_CONTEXT,UCI_CATEGORICAL_CONTEXT
 _,_,base=load_oulad_static_tables(ROOT/'data/raw');daily=build_compact_vle_daily(ROOT/'data/raw',ROOT/'artifacts/hybrid/phase1/runtime');views={};frames=[]
 for s in STAGES[domain]:
  e,v,_=build_oulad_phase7_view(base,daily,int(s[:-3])/100,str(ROOT/'data/raw'));views[s]=v;frames.append(e)
 c=pd.concat(frames,ignore_index=True).drop_duplicates('record_id')[['record_id','group_id','target']+OULAD_NUMERIC_CONTEXT+OULAD_CATEGORICAL_CONTEXT]
 return _pad(views),c,OULAD_NUMERIC_CONTEXT,OULAD_CATEGORICAL_CONTEXT

def phase7_domain(domain):
 if domain not in _PHASE7_DOMAIN_CACHE:_PHASE7_DOMAIN_CACHE[domain]=_build_phase7_domain(domain)
 return copy.deepcopy(_PHASE7_DOMAIN_CACHE[domain])

def oulad_outcomes():
 global _OULAD_OUTCOME_CACHE
 if _OULAD_OUTCOME_CACHE is None:
  _,_,base=load_oulad_static_tables(ROOT/'data/raw');_OULAD_OUTCOME_CACHE=base.assign(record_id=base.record_id.astype(str)).drop_duplicates('record_id').set_index('record_id').final_result.to_dict()
 return _OULAD_OUTCOME_CACHE

def historical_domain(domain):
 old,context,numeric,categorical,_,_=load_domain(domain);views={}
 for stage,x in old.items():
  n=len(x.record_id);views[stage]=SimpleNamespace(record_id=x.record_id.astype(str),group_id=x.group_id,target=x.target,temporal=x.temporal,temporal_mask=x.mask,lengths=x.lengths,aggregate=np.zeros((n,1),np.float32),aggregate_available=np.zeros(n,np.int8),progress=np.full(n,PROGRESS[stage],np.float32))
 return views,context,numeric,categorical

def _partitions(domain,fold,context):
 a=pd.read_parquet(ROOT/f'artifacts/hybrid/phase1/splits/{domain}_inner.parquet');a=a[a.outer_fold==0].copy();a.record_id=a.record_id.astype(str)
 valid=set(a.loc[a.inner_fold==fold,'record_id']);train=set(a.loc[a.inner_fold!=fold,'record_id']);frame=context[context.record_id.astype(str).isin(train)].drop_duplicates('record_id').reset_index(drop=True);fi,si=split_fit_stop(frame);fit=frame.iloc[fi].record_id.astype(str).tolist();stop=frame.iloc[si].record_id.astype(str).tolist()
 if set(fit)&set(stop) or (set(fit)|set(stop))&valid:raise RuntimeError('PHASE7_STOP_VALID_LEAKAGE')
 return fit,stop,valid

def _arrays(view,ids,static,agg):
 ix=np.asarray([view.record_id.tolist().index(r) for r in ids]);return (torch.tensor(static[[static[0].shape[0]*0+i for i in ix]],dtype=torch.float32),torch.tensor(view.temporal[ix],dtype=torch.float32),torch.tensor(view.temporal_mask[ix]),torch.tensor(view.lengths[ix]),torch.tensor(agg[ix],dtype=torch.float32),torch.tensor(view.aggregate_available[ix],dtype=torch.float32),torch.tensor(view.progress[ix],dtype=torch.float32),torch.tensor(view.target[ix],dtype=torch.float32))

def _scale(views,context,numeric,categorical,fit,domain):
 prep=ContextPreprocessor(numeric,categorical).fit(context[context.record_id.astype(str).isin(fit)]); raw=prep.transform(context);sm={r:raw[i] for i,r in enumerate(context.record_id.astype(str))}
 # aggregate scaling is fit on available FIT observations only.
 aggs=[]
 for v in views.values():
  lookup={r:i for i,r in enumerate(v.record_id.astype(str))};ix=[lookup[r] for r in fit if r in lookup and v.aggregate_available[lookup[r]]]
  if ix:aggs.append(v.aggregate[ix])
 mean=np.concatenate(aggs).mean(0) if aggs else np.zeros(next(iter(views.values())).aggregate.shape[1]);std=np.concatenate(aggs).std(0) if aggs else np.ones_like(mean);std=np.where(std<1e-6,1,std)
 for v in views.values():v.aggregate[:]=(v.aggregate-mean)/std
 if domain=='oulad':
  xs=[];ms=[]
  for v in views.values():
   lookup={r:i for i,r in enumerate(v.record_id.astype(str))};ix=[lookup[r] for r in fit if r in lookup]
   if ix:xs.append(v.temporal[ix]);ms.append(v.temporal_mask[ix])
  sc=MaskedStandardScaler().fit(np.concatenate(xs),np.concatenate(ms))
  for v in views.values():v.temporal[:]=sc.transform(v.temporal,v.temporal_mask)
 return sm,prep

def _model(domain,candidate,static_dim,td,ad,dropout):
 if candidate in {'A0','A1'}:
  if domain=='uci':
   p={'d_model':96,'cnn_channels':128,'cnn_blocks':2,'bilstm_hidden':128,'bilstm_layers':1,'context_hidden':96,'shared_head_hidden':128,'dropout':dropout,'uci_wide_context':False};return SharedHeadHybrid(shared_config('uci',td,static_dim+(ad if candidate=='A1' else 0),p)).cuda()
  p=__import__('json').loads((ROOT/'artifacts/hybrid/phase4/oulad_phase4a_selected_config.json').read_text())['params'];return Hybrid(HybridConfig(td,static_dim+(ad if candidate=='A1' else 0),**candidate_model_kwargs(p,'R0'))).cuda()
 flags={'A2':(False,False,False),'A3':(True,True,False),'A4':(True,True,True)}[candidate];return UnifiedHybrid(UnifiedHybridConfig(static_dim,td,ad,dropout=dropout,use_last_state=flags[0],use_progress=flags[1],use_interaction=flags[2])).cuda()

def run_redesigned(domain,candidate,fold,seed,config,checkpoint_path=None,model_transform=None,collect_predictions=False):
 """Common BCE-only standard 3-fold FIT/STOP/VALID job for A0--A4."""
 views,context,numeric,categorical=(historical_domain(domain) if candidate=='A0' else phase7_domain(domain));fit,stop,valid=_partitions(domain,fold,context);sm,prep=_scale(views,context,numeric,categorical,fit,domain);indices={s:{r:i for i,r in enumerate(v.record_id.astype(str))} for s,v in views.items()};static_dim=prep.output_dim;td=next(iter(views.values())).temporal.shape[2];ad=next(iter(views.values())).aggregate.shape[1];seed_everything(seed);model=_model(domain,candidate,static_dim,td,ad,config['dropout']);model=model_transform(model) if model_transform is not None else model;n=sum(p.numel() for p in model.parameters());torch.cuda.reset_peak_memory_stats();started=time.monotonic();opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay']);sch=_scheduler(opt,config['max_epochs'],config.get('warmup_epochs',0),1e-5,config['learning_rate']);y=np.asarray([next(v.target[indices[s][r]] for s,v in views.items() if r in indices[s]) for r in fit]);base_weight=(len(y)-y.sum())/max(1,y.sum());weight=base_weight*float(config.get('positive_class_weight_multiplier',1.0));loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight],device='cuda'));best=-1.;state=None;epoch_best=0;stale=0
 def forward(stage,ids):
  v=views[stage];ix=np.asarray([indices[stage][r] for r in ids]);s=torch.tensor(np.asarray([sm[r] for r in ids]),dtype=torch.float32,device='cuda');t=torch.tensor(v.temporal[ix],dtype=torch.float32,device='cuda');m=torch.tensor(v.temporal_mask[ix],device='cuda');l=torch.tensor(v.lengths[ix],device='cuda');a=torch.tensor(v.aggregate[ix],dtype=torch.float32,device='cuda');aa=torch.tensor(v.aggregate_available[ix],dtype=torch.float32,device='cuda');p=torch.tensor(v.progress[ix],dtype=torch.float32,device='cuda')
  if candidate in {'A0','A1'}:
   context_input=s if candidate=='A0' else torch.cat((s,a),-1)
   return (model(t,m,l,context_input,None if domain=='uci' else p),v.target[ix])
  return (model(s,t,m,l,a,aa,p),v.target[ix])
 @torch.no_grad()
 def predict(stage,ids):
  model.eval();out=[]
  for chunk in [ids[i:i+config['batch_size']] for i in range(0,len(ids),config['batch_size'])]:out.append(torch.sigmoid(forward(stage,chunk)[0]).float().cpu().numpy())
  return np.concatenate(out)
 for epoch in range(config['max_epochs']):
  model.train()
  # UCI deliberately has equal losses for its three aligned stage views;
  # OULAD retains historical one-prefix-per-record sampling.
  if domain=='uci':
   loader=DataLoader(TensorDataset(torch.arange(len(fit))),batch_size=config['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed+epoch))
   for (bi,) in loader:
    ids=[fit[i] for i in bi.tolist()];opt.zero_grad(set_to_none=True);losses=[]
    for s in STAGES[domain]:
     z,yy=forward(s,ids);losses.append(loss_fn(z,torch.tensor(yy,dtype=torch.float32,device='cuda')))
    loss=torch.stack(losses).mean()
    if not torch.isfinite(loss):raise RuntimeError('NONFINITE_LOSS')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm']);
    if not torch.isfinite(norm):raise RuntimeError('NONFINITE_GRADIENT')
    opt.step()
  else:
   avail={r:[s for s in views if r in indices[s]] for r in fit};chosen=sample_prefixes(fit,[avail[r] for r in fit],seed,epoch);by={s:[] for s in STAGES[domain]}
   for r,s in zip(fit,chosen):by[s].append(r)
   for s,ids in by.items():
    for i in range(0,len(ids),config['batch_size']):
     q=ids[i:i+config['batch_size']];opt.zero_grad(set_to_none=True);z,yy=forward(s,q);loss=loss_fn(z,torch.tensor(yy,dtype=torch.float32,device='cuda'))
     if not torch.isfinite(loss):raise RuntimeError('NONFINITE_LOSS')
     loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'])
     if not torch.isfinite(norm):raise RuntimeError('NONFINITE_GRADIENT')
     opt.step()
  sch.step();stop_ap=[]
  for s,v in views.items():
   ids=[r for r in stop if r in indices[s]];stop_ap.append(average_precision_score(v.target[[indices[s][r] for r in ids]],predict(s,ids)))
  score=float(np.mean(stop_ap))
  if score>best:best=score;epoch_best=epoch+1;state=copy.deepcopy({k:x.detach().cpu() for k,x in model.state_dict().items()});stale=0
  else:stale+=1
  if stale>=config['patience']:break
 model.load_state_dict(state)
 if checkpoint_path is not None:
  checkpoint_path=__import__('pathlib').Path(checkpoint_path);checkpoint_path.parent.mkdir(parents=True,exist_ok=True);temporary=checkpoint_path.with_suffix(checkpoint_path.suffix+'.tmp')
  torch.save({'model_state':state,'best_epoch':epoch_best,'domain':domain,'candidate':candidate,'fold':fold,'seed':seed,'training_config':config},temporary);os.replace(temporary,checkpoint_path)
 metrics={};diag=[];predictions={}
 for s,v in views.items():
  tr=[r for r in fit if r in indices[s]];sp=[r for r in stop if r in indices[s]];va=[r for r in valid if r in indices[s]];ts=predict(s,tr);ss=predict(s,sp);vs=predict(s,va);ti=[indices[s][r] for r in tr];si=[indices[s][r] for r in sp];vi=[indices[s][r] for r in va];threshold=stage_threshold_metrics(v.target[si],ss,v.target[vi],vs);m=threshold['stop_selected'];m.update({'train_pr_auc':float(average_precision_score(v.target[ti],ts)),'validation_pr_auc':m['pr_auc'],'train_validation_gap':float(average_precision_score(v.target[ti],ts)-m['pr_auc']),'selected_threshold':threshold['selected_threshold'],'threshold_source':'stop_only'});metrics[s]=m
  if domain=='oulad':
   outcomes=oulad_outcomes();pred=vs>=threshold['selected_threshold'];labels=np.asarray([outcomes[r] for r in va]);m['recall_fail']=float(pred[labels=='Fail'].mean()) if np.any(labels=='Fail') else None;m['recall_withdrawn']=float(pred[labels=='Withdrawn'].mean()) if np.any(labels=='Withdrawn') else None;m['risk_prevalence']=float(v.target[vi].mean())
  if collect_predictions:
   outcomes=oulad_outcomes() if domain=='oulad' else {}
   predictions[s]=[{'record_id':r,'target':int(v.target[indices[s][r]]),'score':float(score),'prediction':int(score>=threshold['selected_threshold']),'outcome':outcomes.get(r)} for r,score in zip(va,vs)]
 if candidate in {'A2','A3','A4'}:
  import torch.nn.functional as F
  pairs=(('h_static','h_cnn'),('h_static','h_bilstm'),('h_static','h_aggregate'),('h_cnn','h_bilstm'),('h_cnn','h_aggregate'),('h_bilstm','h_aggregate'))
  for stage,v in views.items():
   va=[r for r in valid if r in indices[stage]];chunks=[]
   for i in range(0,len(va),config['batch_size']):
    forward(stage,va[i:i+config['batch_size']]);chunks.append({k:x.float().cpu() for k,x in model.last_diagnostics.items()})
   d={k:torch.cat([x[k] for x in chunks]) for k in chunks[0]};row={'stage':stage}
   for key in ('h_static','h_cnn','h_bilstm','h_aggregate'):row[f'{key}_norm']=float(d[key].norm(dim=1).mean());row[f'{key}_variance']=float(d[key].var(dim=0,unbiased=False).mean())
   for left,right in pairs:row[f'cosine_{left[2:]}_{right[2:]}']=float(F.cosine_similarity(d[left],d[right],dim=1).mean())
   row['interaction_base_ratio']=float(d['interaction'].norm(dim=1).mean()/d['base'].norm(dim=1).mean().clamp_min(1e-8)) if candidate=='A4' else None;diag.append(row)
 result={'metrics':metrics,'parameter_count':n,'best_epoch':epoch_best,'runtime_seconds':time.monotonic()-started,'peak_gpu_memory_bytes':int(torch.cuda.max_memory_allocated()),'branch_diagnostics':diag,'outer_test_used':False}
 if collect_predictions:result['predictions']=predictions
 return result

def run_parity_baseline(domain,family,fold,seed):
 """Fixed, train-only Phase7 parity baseline; no parameter search."""
 from sklearn.compose import ColumnTransformer
 from sklearn.impute import SimpleImputer
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import OneHotEncoder,StandardScaler
 from sklearn.linear_model import LogisticRegression
 from sklearn.ensemble import RandomForestClassifier
 views,context,numeric,categorical=phase7_domain(domain);fit,stop,valid=_partitions(domain,fold,context);started=time.monotonic();rows={}
 for stage,v in views.items():
  aligned=context.assign(record_id=context.record_id.astype(str)).set_index('record_id').loc[v.record_id.astype(str)].reset_index();raw=build_phase7_baseline_frame(aligned,v);ids=raw.record_id.astype(str);tr=raw[ids.isin(fit)];st=raw[ids.isin(stop)];va=raw[ids.isin(valid)];features=[c for c in raw if c not in {'record_id','group_id','target'}];cats=[c for c in categorical if c in features];num=[c for c in features if c not in cats];pre=ColumnTransformer([('n',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),num),('c',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore')),cats)])
  if family=='logistic_regression':model=LogisticRegression(C=1.0,class_weight='balanced',max_iter=1000,random_state=seed)
  elif family=='random_forest':model=RandomForestClassifier(n_estimators=300,class_weight='balanced',min_samples_leaf=2,n_jobs=-1,random_state=seed)
  elif family=='xgboost':
   from xgboost import XGBClassifier;model=XGBClassifier(n_estimators=300,max_depth=5,learning_rate=.05,subsample=.8,colsample_bytree=.8,eval_metric='logloss',random_state=seed,n_jobs=1)
  else:
   from catboost import CatBoostClassifier;model=CatBoostClassifier(iterations=300,depth=6,learning_rate=.05,verbose=False,random_seed=seed)
  pipe=make_pipeline(pre,model);pipe.fit(tr[features],tr.target);train_score=pipe.predict_proba(tr[features])[:,1];stop_score=pipe.predict_proba(st[features])[:,1];score=pipe.predict_proba(va[features])[:,1];threshold=stage_threshold_metrics(st.target,stop_score,va.target,score);rows[stage]=threshold['stop_selected'];rows[stage].update({'train_pr_auc':float(average_precision_score(tr.target,train_score)),'validation_pr_auc':rows[stage]['pr_auc'],'train_validation_gap':float(average_precision_score(tr.target,train_score)-rows[stage]['pr_auc']),'selected_threshold':threshold['selected_threshold'],'threshold_source':'stop_only','runtime_seconds':time.monotonic()-started})
 return {'metrics':rows,'parameter_count':None,'runtime_seconds':time.monotonic()-started,'outer_test_used':False}
