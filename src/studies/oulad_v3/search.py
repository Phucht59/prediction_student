from __future__ import annotations

import json, time
from dataclasses import dataclass
from typing import Any
import numpy as np
import optuna
from .data import OULADV3Data, manifest_indices
from .training import fit_candidate
from src.studies.oulad_v2.metrics import choose_thresholds

@dataclass
class SearchResult:
    candidate_id:str; outer_fold:int; temporal_config:dict|None; aggregate_config:dict; thresholds:dict
    refit_epochs:int; inner_selected_epochs:list[int]; parameter_count:int; trial_rows:list[dict]
    learning_curves:list[dict]; runtime_seconds:float

def _common(config):
    return {**config,"max_epochs":40,"patience":6,"gradient_clip":1.0}

def _config(trial,candidate,parent_temporal,parent_aggregate,pooling_temporal):
    if candidate=="V3-P0":
        temporal=dict(parent_temporal); parent_dropout=float(temporal["dropout"])
        temporal.update({"pooling":trial.suggest_categorical("pooling",["last_mean_max","masked_attention"]),
                         "pooling_projection":trial.suggest_categorical("pooling_projection",[32,64]),
                         "dropout":float(np.clip(parent_dropout+trial.suggest_categorical("dropout_adjustment",[-.05,0,.05]),.10,.40))})
        return _common(temporal),dict(parent_aggregate)
    if candidate=="V3-D0":
        temporal=dict(pooling_temporal); parent_dropout=float(temporal["dropout"])
        temporal.update({"learning_rate":trial.suggest_categorical("learning_rate",[2e-4,5e-4,1e-3]),
                         "weight_decay":trial.suggest_categorical("weight_decay",[1e-6,1e-5,1e-4]),
                         "dropout":float(np.clip(parent_dropout+trial.suggest_categorical("dropout_adjustment",[-.05,0,.05]),.10,.40)),
                         "positive_weight":trial.suggest_categorical("positive_weight",["none","sqrt_balanced","fully_balanced"]),
                         "scheduler":trial.suggest_categorical("scheduler",["fixed_lr","deterministic_cosine"])})
        return _common(temporal),dict(parent_aggregate)
    if candidate=="V3-A1":
        aggregate={"aggregate_hidden_1":trial.suggest_categorical("aggregate_hidden_1",[64,128]),
                   "aggregate_hidden_2":trial.suggest_categorical("aggregate_hidden_2",[0,64]),
                   "dropout":trial.suggest_categorical("dropout",[.15,.25,.35]),
                   "learning_rate":trial.suggest_categorical("learning_rate",[2e-4,5e-4,1e-3]),
                   "weight_decay":trial.suggest_categorical("weight_decay",[1e-6,1e-5,1e-4]),
                   "batch_size":trial.suggest_categorical("batch_size",[128,256]),
                   "positive_weight":trial.suggest_categorical("positive_weight",["none","sqrt_balanced","fully_balanced"]),
                   "scheduler":trial.suggest_categorical("scheduler",["fixed_lr","deterministic_cosine"]),"static_hidden":32}
        return None,_common(aggregate)
    raise KeyError(candidate)

def run_search(data:OULADV3Data,candidate_id:str,outer_fold:int,inner_manifest,*,trials:int,device:str,
               seed:int,parent_temporal:dict,parent_aggregate:dict,pooling_temporal:dict|None=None)->SearchResult:
    started=time.perf_counter(); trial_rows=[]; curves=[]
    def objective(trial):
        temporal,aggregate=_config(trial,candidate_id,parent_temporal,parent_aggregate,pooling_temporal)
        probabilities=[]; targets=[]; epochs=[]; counts=[]; fit_runtime=0.; trial_started=time.perf_counter()
        try:
            for inner_fold in sorted(inner_manifest.inner_fold.unique()):
                train,validation=manifest_indices(data.v2,inner_manifest,int(inner_fold))
                result=fit_candidate(data,candidate_id,train,validation,temporal_config=temporal,aggregate_config=aggregate,
                                     seed=seed+int(inner_fold),device_name=device)
                probabilities.append(result.probabilities); targets.append(data.y[validation]); epochs.append(result.selected_epoch)
                counts.append(result.parameter_count); fit_runtime+=result.runtime_seconds
                curves.extend({"candidate_id":candidate_id,"outer_fold":outer_fold,"trial_id":trial.number,"inner_fold":int(inner_fold),**row} for row in result.history)
            thresholds=choose_thresholds(np.concatenate(targets),np.concatenate(probabilities))
            trial.set_user_attr("temporal_config",json.dumps(temporal,sort_keys=True)); trial.set_user_attr("aggregate_config",json.dumps(aggregate,sort_keys=True))
            trial.set_user_attr("thresholds",json.dumps(thresholds,sort_keys=True)); trial.set_user_attr("epochs",json.dumps(epochs)); trial.set_user_attr("count",max(counts)); trial.set_user_attr("fit_runtime",fit_runtime)
            return float(thresholds["inner_macro_f1"])
        except RuntimeError as error:
            trial.set_user_attr("failure",str(error))
            if "out of memory" in str(error).lower(): raise optuna.TrialPruned("CUDA OOM") from error
            raise
        finally: trial.set_user_attr("runtime",time.perf_counter()-trial_started)
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective,n_trials=trials,catch=(RuntimeError,),show_progress_bar=False)
    for t in study.trials:
        trial_rows.append({"candidate_id":candidate_id,"outer_fold":outer_fold,"trial_id":t.number,"state":t.state.name,"value":t.value,
                           "temporal_config":t.user_attrs.get("temporal_config"),"aggregate_config":t.user_attrs.get("aggregate_config"),
                           "thresholds":t.user_attrs.get("thresholds"),"selected_epochs":t.user_attrs.get("epochs"),"parameter_count":t.user_attrs.get("count"),
                           "fit_runtime_seconds":t.user_attrs.get("fit_runtime",0),"wall_runtime_seconds":t.user_attrs.get("runtime",0),"failure_reason":t.user_attrs.get("failure")})
    if not any(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials): raise RuntimeError(f"No complete trials: {candidate_id}/{outer_fold}")
    best=study.best_trial; epochs=json.loads(best.user_attrs["epochs"])
    return SearchResult(candidate_id,outer_fold,json.loads(best.user_attrs["temporal_config"]),json.loads(best.user_attrs["aggregate_config"]),
                        json.loads(best.user_attrs["thresholds"]),max(1,int(round(np.median(epochs)))),epochs,int(best.user_attrs["count"]),trial_rows,curves,time.perf_counter()-started)
