"""Run controlled student-mat benchmarks under Scientific Protocol V2.

The runner deliberately never reads the legacy observed holdout.  It consumes
the shared 316-record manifest, writes immutable run directories, and emits
outer-validation predictions only.
"""
from __future__ import annotations

import argparse, hashlib, json, subprocess, time, uuid, sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT_DIR
from src.data_pipeline import SOURCE_ROW_NUMBER_COLUMN, process_target_and_stratify
from src.evaluation.protocol import (DEFAULT_FOLD_MANIFEST_PATH, assert_no_legacy_records,
    classification_metrics, file_checksum, load_fold_manifest, outer_folds_from_manifest,
    source_record_identity, validate_scenario_features, hard_label_probabilities, validate_probability_matrix)
from src.model_selection import fit_fold_predict_proba
from src.postgres_data_source import load_dataset_version_from_postgres
from src.evaluation.metrics import classification_metrics as canonical_metrics

SEEDS = [42, 52, 62, 72, 82]
BENCHMARK_ROOT = ROOT_DIR / "artifacts" / "benchmark_v2"
REPORT_ROOT = ROOT_DIR / "reports" / "benchmark_v2"
LEGACY_CONFIG = json.loads((ROOT_DIR / "artifacts/model_selection/nested-full-20260710/selected_config.json").read_text(encoding="utf-8"))["best_params"]


def sha(payload): return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
def git_commit(): return subprocess.run(["git","rev-parse","HEAD"], cwd=ROOT_DIR, capture_output=True, text=True, check=True).stdout.strip()
def git_tree_is_clean(): return not subprocess.run(["git", "status", "--porcelain"], cwd=ROOT_DIR, capture_output=True, text=True, check=True).stdout.strip()
def label_from_g3(values): return np.select([values <= 9, values <= 14], [0,1], default=2).astype(int)
def probs_from_pred(pred):
    return hard_label_probabilities(pred)
def ordinal_probs(x, y, xt, c):
    scaler=StandardScaler().fit(x); x,xt=scaler.transform(x),scaler.transform(xt)
    p=[]
    for cutoff in (0,1):
        clf=LogisticRegression(C=c,max_iter=500,random_state=42).fit(x,(y>cutoff).astype(int))
        p.append(clf.predict_proba(xt)[:,1])
    greater0,greater1=np.maximum(p[0],p[1]),np.minimum(p[0],p[1])
    return np.column_stack([1-greater0, greater0-greater1, greater1])
def metric(y,p,pred,raw=None):
    d=canonical_metrics(y,pred,p)
    if raw:
        truth,estimate=raw
        d.update({"rmse":float(mean_squared_error(truth,estimate)**.5),"mae_g3":float(mean_absolute_error(truth,estimate)),"r2":float(r2_score(truth,estimate))})
    return d

def predict_sklearn(name,x,y,raw,xt,config,seed):
    if name=="majority":
        pred=np.full(len(xt),np.bincount(y,minlength=3).argmax()); return probs_from_pred(pred),pred,None,None
    if name in {"g2_rule","g1_rule"}:
        pred=label_from_g3(xt[:, -1]); return probs_from_pred(pred),pred,None,None
    if name=="logistic":
        sc=StandardScaler().fit(x); clf=LogisticRegression(C=config["C"],max_iter=500,random_state=seed).fit(sc.transform(x),y); p=clf.predict_proba(sc.transform(xt))
    elif name=="ordinal":
        p=ordinal_probs(x,y,xt,config["C"]); clf=None
    elif name=="ridge":
        sc=StandardScaler().fit(x); clf=Ridge(alpha=config["alpha"]).fit(sc.transform(x),raw); g3=np.clip(clf.predict(sc.transform(xt)),0,20); pred=label_from_g3(g3); return probs_from_pred(pred),pred,g3,clf
    elif name=="hgb":
        clf=HistGradientBoostingClassifier(max_leaf_nodes=config["leaves"],l2_regularization=config["l2"],learning_rate=.05,max_iter=100,random_state=seed).fit(x,y); p=clf.predict_proba(xt)
    elif name=="mlp":
        sc=StandardScaler().fit(x); clf=MLPClassifier(hidden_layer_sizes=config["hidden"],alpha=config["alpha"],learning_rate_init=config["lr"],batch_size=config["batch"],max_iter=config["epochs"],early_stopping=False,random_state=seed).fit(sc.transform(x),y); p=clf.predict_proba(sc.transform(xt)); return p,p.argmax(1),None,(sc,clf)
    else: raise ValueError(name)
    return p,p.argmax(1),None,clf

def inner_select(name,x,y,raw,configs,seed=42):
    folds=StratifiedKFold(3,shuffle=True,random_state=42)
    best,bestscore=None,-np.inf
    for config in configs:
        scores=[]
        for tr,va in folds.split(x,y):
            p,pred,_,_=predict_sklearn(name,x[tr],y[tr],raw[tr],x[va],config,seed)
            scores.append(metric(y[va],p,pred)["macro_f1"])
        score=float(np.mean(scores))
        if score>bestscore: best,bestscore=config,score
    return best,bestscore

def cnn_config_trial(trial):
    return {"learning_rate":trial.suggest_float("learning_rate",5e-4,8e-3,log=True),"weight_decay":trial.suggest_float("weight_decay",1e-6,1e-3,log=True),"batch_size":trial.suggest_categorical("batch_size",[16,32]),"cnn_channels":trial.suggest_categorical("cnn_channels",[8,16,32]),"cnn_kernel_size":trial.suggest_categorical("cnn_kernel_size",[1,2]),"lstm_hidden_dim":trial.suggest_categorical("lstm_hidden_dim",[8,16,32]),"dropout":trial.suggest_float("dropout",.1,.5),"sequence_dropout":trial.suggest_float("sequence_dropout",.05,.4),"loss":"weighted_ce","class_weight_mode":trial.suggest_categorical("class_weight_mode",["none","balanced"]),"oversample_method":"none","smote_ratio":1.,"resampling_k_neighbors":2,"max_epochs":20,"patience":5,"scheduler_patience":3}

def inner_select_cnn(frame,features,spec,fold,trial_budget):
    outer_train=frame.iloc[fold[0]].copy(); y=outer_train.G3.astype(int).to_numpy(); inner=StratifiedKFold(3,shuffle=True,random_state=42)
    def objective(trial):
        cfg=cnn_config_trial(trial); scores=[]
        for i,(tr,va) in enumerate(inner.split(outer_train,y)):
            r=fit_fold_predict_proba(train_fold=outer_train.iloc[tr],validation_fold=outer_train.iloc[va],spec=spec,params=cfg,seed=42+i,fold_index=i)
            scores.append(metric(r.true_labels,r.probabilities,r.predictions)["macro_f1"])
        return float(np.mean(scores))
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42),pruner=optuna.pruners.NopPruner())
    study.optimize(objective,n_trials=trial_budget)
    return {**study.best_params,"loss":"weighted_ce","oversample_method":"none","smote_ratio":1.,"resampling_k_neighbors":2,"max_epochs":20,"patience":5,"scheduler_patience":3},study

def run(args):
    manifest=load_fold_manifest(args.fold_manifest); validate_scenario_features(["G1","G2"],"late_stage"); validate_scenario_features(["G1"],"early_warning")
    raw,meta=load_dataset_version_from_postgres("student-mat",1); raw_g3=raw.G3.astype(float).to_numpy(); frame=process_target_and_stratify(raw.copy(),"G3","student","3class").drop(columns=["_strat_target"])
    wanted={r["source_row_number"] for r in manifest["development_records"]}; frame=frame[frame[SOURCE_ROW_NUMBER_COLUMN].isin(wanted)].sort_values(SOURCE_ROW_NUMBER_COLUMN).reset_index(drop=True); frame["_raw_g3"]=raw_g3[frame[SOURCE_ROW_NUMBER_COLUMN].astype(int).to_numpy()]
    assert_no_legacy_records([source_record_identity(1,v) for v in frame[SOURCE_ROW_NUMBER_COLUMN]])
    folds=outer_folds_from_manifest(frame,manifest); run_id=args.run_id or f"benchmark-v2-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"; root=BENCHMARK_ROOT/run_id
    if root.exists(): raise FileExistsError(root)
    if not git_tree_is_clean():
        raise RuntimeError("Benchmark source tree must be clean before a run is created.")
    (root/"predictions").mkdir(parents=True); (root/"checkpoints").mkdir(); (root/"configs").mkdir()
    allrows=[]; configs={}
    plans=[("late_stage","majority",["G1","G2"],[{}],False),("late_stage","g2_rule",["G2"],[{}],False),("late_stage","logistic_g2",["G2"],[{"C":.1},{"C":1},{"C":10}],False),("late_stage","logistic_g1_g2",["G1","G2"],[{"C":.1},{"C":1},{"C":10}],False),("late_stage","ordinal_logistic",["G1","G2"],[{"C":.1},{"C":1},{"C":10}],False),("late_stage","ridge_regression",["G1","G2"],[{"alpha":.1},{"alpha":1},{"alpha":10}],False),("late_stage","hgb_g1_g2",["G1","G2"],[{"leaves":3,"l2":0.},{"leaves":7,"l2":.1},{"leaves":15,"l2":1.}],False),("late_stage","small_mlp",["G1","G2"],[{"hidden":(8,),"alpha":1e-4,"lr":.001,"batch":16,"epochs":100},{"hidden":(16,),"alpha":1e-3,"lr":.003,"batch":16,"epochs":100}],True),("early_warning","majority",["G1"],[{}],False),("early_warning","g1_rule",["G1"],[{}],False),("early_warning","logistic_g1",["G1"],[{"C":.1},{"C":1},{"C":10}],False),("early_warning","ordinal_logistic",["G1"],[{"C":.1},{"C":1},{"C":10}],False),("early_warning","ridge_regression",["G1"],[{"alpha":.1},{"alpha":1},{"alpha":10}],False),("early_warning","hgb_g1",["G1"],[{"leaves":3,"l2":0.},{"leaves":7,"l2":.1},{"leaves":15,"l2":1.}],False),("early_warning","small_mlp",["G1"],[{"hidden":(8,),"alpha":1e-4,"lr":.001,"batch":16,"epochs":100},{"hidden":(16,),"alpha":1e-3,"lr":.003,"batch":16,"epochs":100}],True)]
    if args.smoke: plans=[plans[0],plans[2],plans[6],plans[7]]; folds=folds[:1]
    cnn_models=[("cnn_only",{**LEGACY_CONFIG,"architecture_variant":"cnn_only"},False),("bilstm_only",{**LEGACY_CONFIG,"architecture_variant":"bilstm_only"},False),("cnn_bilstm_legacy_config_v2_refit",{**LEGACY_CONFIG,"architecture_variant":"cnn_bilstm"},False),("cnn_bilstm_v2_tuned",None,True)]
    seeds_per_plan = lambda multiseed: len(SEEDS if multiseed and not args.smoke else [42])
    expected_jobs = len(folds) * (sum(seeds_per_plan(multiseed) for _, _, _, _, multiseed in plans) + len(cnn_models) * len([42] if args.smoke else SEEDS))
    expected_predictions = sum(len(validation) for _, validation in folds) * (sum(seeds_per_plan(multiseed) for _, _, _, _, multiseed in plans) + len(cnn_models) * len([42] if args.smoke else SEEDS))
    feature_contract_checksum = sha({scenario: sorted(features) for scenario, _, features, _, _ in plans})
    started_at = datetime.now(timezone.utc).isoformat()
    manifest_out={"benchmark_run_id":run_id,"created_at":started_at,"started_at":started_at,"source_commit":git_commit(),"source_tree_clean":True,"dataset_checksum":meta["content_hash"],"fold_manifest_checksum":manifest["manifest_checksum"],"feature_contract_checksum":feature_contract_checksum,"protocol_version":"scientific_protocol_v2","probability_contract_version":"deterministic_one_hot_v1_strict_1e-6","model_list":[name for _, name, _, _, _ in plans]+[name for name, _, _ in cnn_models],"scenario_list":sorted({scenario for scenario, _, _, _, _ in plans}),"seed_list":SEEDS,"expected_jobs":expected_jobs,"expected_predictions":expected_predictions,"outer_folds":5,"inner_folds":3,"optuna_trials":1 if args.smoke else 30,"status":"running","command":" ".join(__import__('sys').argv),"legacy_heldout_observed_used":False,"legacy_79_blocked":True,"pre_assessment":"NOT_EVALUABLE_UNDER_STRICT_FEATURE_AVAILABILITY_CONTRACT","invalid_predecessor_run":"benchmark-v2-full-20260713b","predecessor_rejection_reason":"probability_contract_violation"}; json.dump(manifest_out,open(root/"benchmark_manifest.json","w"),indent=2)
    for scenario,display,features,candidates,multiseed in plans:
        x=frame[features].to_numpy(float); y=frame.G3.astype(int).to_numpy(); rawy=frame._raw_g3.to_numpy(float); base={"logistic_g2":"logistic","logistic_g1_g2":"logistic","logistic_g1":"logistic","ordinal_logistic":"ordinal","ridge_regression":"ridge","hgb_g1_g2":"hgb","hgb_g1":"hgb","small_mlp":"mlp"}.get(display,display)
        for fi,(tr,va) in enumerate(folds):
            cfg,score=inner_select(base,x[tr],y[tr],rawy[tr],candidates); configs[f"{scenario}/{display}/fold{fi}"]={"config":cfg,"inner_macro_f1":score,"candidates":candidates}
            for seed in (SEEDS if multiseed and not args.smoke else [42]):
                p,pred,g3,trained=predict_sklearn(base,x[tr],y[tr],rawy[tr],x[va],cfg,seed)
                if multiseed: joblib.dump(trained,root/"checkpoints"/f"{scenario}_{display}_fold{fi}_seed{seed}.joblib")
                for idx,(_,row) in enumerate(frame.iloc[va].iterrows()):
                    allrows.append({"run_id":run_id,"model_name":display,"scenario":scenario,"feature_set_id":"+".join(features),"dataset_version_id":1,"record_id":source_record_identity(1,row[SOURCE_ROW_NUMBER_COLUMN]),"outer_fold":fi,"training_seed":seed,"true_label":int(row.G3),"predicted_label":int(pred[idx]),"probability_low":float(p[idx,0]),"probability_medium":float(p[idx,1]),"probability_high":float(p[idx,2]),"predicted_g3":None if g3 is None else float(g3[idx]),"fold_manifest_checksum":manifest["manifest_checksum"],"config_checksum":sha(cfg),"source_commit":git_commit()})
    # CNN models are isolated to late-stage only and use the refit-safe existing trainer.
    spec=type("Spec",(),{"target_col":"G3","kind":"student"})()
    cnn_frame=frame[[SOURCE_ROW_NUMBER_COLUMN,"G1","G2","G3","_raw_g3"]].copy()
    for name,fixed,tuned in cnn_models:
        for fi,fold in enumerate(folds):
            cfg,study=(inner_select_cnn(cnn_frame,["G1","G2"],spec,fold,1 if args.smoke else 30) if tuned else (fixed,None)); configs[f"late_stage/{name}/fold{fi}"]={"config":cfg,"trial_count":0 if study is None else len(study.trials),"trial_history":[] if study is None else [{"number":t.number,"value":t.value,"params":t.params} for t in study.trials]}
            for seed in ([42] if args.smoke else SEEDS):
                r=fit_fold_predict_proba(train_fold=cnn_frame.iloc[fold[0]],validation_fold=cnn_frame.iloc[fold[1]],spec=spec,params=cfg,seed=seed,fold_index=fi)
                ck=root/"checkpoints"/f"late_stage_{name}_fold{fi}_seed{seed}.pt"; torch.save(r.refit_state_dict,ck)
                for idx,(_,row) in enumerate(frame.iloc[fold[1]].iterrows()):
                    allrows.append({"run_id":run_id,"model_name":name,"scenario":"late_stage","feature_set_id":"G1+G2","dataset_version_id":1,"record_id":source_record_identity(1,row[SOURCE_ROW_NUMBER_COLUMN]),"outer_fold":fi,"training_seed":seed,"true_label":int(row.G3),"predicted_label":int(r.predictions[idx]),"probability_low":float(r.probabilities[idx,0]),"probability_medium":float(r.probabilities[idx,1]),"probability_high":float(r.probabilities[idx,2]),"predicted_g3":None,"fold_manifest_checksum":manifest["manifest_checksum"],"config_checksum":sha(cfg),"source_commit":git_commit()})
    pred=pd.DataFrame(allrows); validate_probability_matrix(pred[["probability_low","probability_medium","probability_high"]].to_numpy(), pred["predicted_label"].to_numpy()); pred.to_csv(root/"predictions"/"outer_validation_predictions.csv",index=False); json.dump(configs,open(root/"configs"/"selected_configs.json","w"),indent=2,default=str)
    fold_metrics=[]
    for keys,g in pred.groupby(["model_name","scenario","training_seed","outer_fold"]): fold_metrics.append({"model_name":keys[0],"scenario":keys[1],"training_seed":keys[2],"outer_fold":keys[3],**metric(g.true_label.to_numpy(),g[["probability_low","probability_medium","probability_high"]].to_numpy(),g.predicted_label.to_numpy(),None if g.predicted_g3.isna().all() else (frame.set_index([SOURCE_ROW_NUMBER_COLUMN]).loc[[int(v.rsplit(':',1)[1]) for v in g.record_id],"_raw_g3"].to_numpy(),g.predicted_g3.to_numpy()))})
    pd.DataFrame(fold_metrics).to_csv(root/"fold_metrics.csv",index=False)
    if not git_tree_is_clean() or git_commit() != manifest_out["source_commit"]:
        raise RuntimeError("Source revision changed during benchmark execution; run cannot be marked completed.")
    manifest_out.update({"status":"completed","completed_at":datetime.now(timezone.utc).isoformat()}); json.dump(manifest_out,open(root/"benchmark_manifest.json","w"),indent=2)
    checks={str(p.relative_to(root)):file_checksum(p) for p in root.rglob("*") if p.is_file()}; json.dump(checks,open(root/"checksums.json","w"),indent=2); print(root)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--fold-manifest",type=Path,default=DEFAULT_FOLD_MANIFEST_PATH); p.add_argument("--run-id"); p.add_argument("--smoke",action="store_true"); run(p.parse_args())
