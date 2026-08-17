"""Frozen estimators and Optuna parameter samplers."""
from __future__ import annotations
import os
import inspect
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))

def mlp_supports_sample_weight() -> bool:
    return "sample_weight" in inspect.signature(MLPClassifier.fit).parameters

def suggest(trial, family):
    c = trial.suggest_categorical
    f = trial.suggest_float; i = trial.suggest_int
    if family == "logistic_regression": return {"C":f("C",1e-3,1e2,log=True),"penalty":c("penalty",["l1","l2"]),"class_weight":c("class_weight",[None,"balanced"])}
    if family == "svm": return {"C":f("C",1e-2,1e2,log=True),"gamma":f("gamma",1e-5,1,log=True),"class_weight":c("class_weight",[None,"balanced"])}
    if family == "random_forest": return {"n_estimators":i("n_estimators",300,1000,step=100),"max_depth":c("max_depth",[None,5,10,15,20,30]),"min_samples_leaf":i("min_samples_leaf",1,20),"max_features":c("max_features",["sqrt","log2",0.3,0.5,0.8]),"class_weight":c("class_weight",[None,"balanced","balanced_subsample"])}
    if family == "xgboost": return {"n_estimators":i("n_estimators",200,1200),"max_depth":i("max_depth",3,10),"learning_rate":f("learning_rate",.01,.2,log=True),"min_child_weight":f("min_child_weight",.5,20,log=True),"subsample":f("subsample",.6,1),"colsample_bytree":f("colsample_bytree",.6,1),"reg_alpha":f("reg_alpha",1e-8,1,log=True),"reg_lambda":f("reg_lambda",1e-3,20,log=True),"gamma":f("gamma",0,5),"imbalance_mode":c("imbalance_mode",["none","balanced"])}
    if family == "catboost": return {"iterations":i("iterations",300,1200),"depth":i("depth",4,10),"learning_rate":f("learning_rate",.01,.2,log=True),"l2_leaf_reg":f("l2_leaf_reg",1,30,log=True),"random_strength":f("random_strength",0,2),"bagging_temperature":f("bagging_temperature",0,2),"border_count":c("border_count",[64,128]),"imbalance_mode":c("imbalance_mode",["none","balanced"])}
    if family == "mlp": return {"hidden_layer_sizes":tuple(c("hidden_layer_sizes",[(64,),(128,),(64,32),(128,64),(256,128)])),"activation":c("activation",["relu","tanh"]),"alpha":f("alpha",1e-6,1e-2,log=True),"learning_rate_init":f("learning_rate_init",1e-4,5e-3,log=True),"batch_size":c("batch_size",[32,64,128,256]),"imbalance_mode":c("imbalance_mode",["none","balanced"])}
    raise ValueError(family)

def make_estimator(family, params, seed, y_train):
    p=dict(params); mode=p.pop("imbalance_mode", "none")
    if family=="logistic_regression": return LogisticRegression(solver="liblinear",max_iter=2000,random_state=seed,**p)
    if family=="svm":
        e=SVC(kernel="rbf",probability=False,cache_size=4096,shrinking=True,random_state=seed,**p); e._hybrid_family="svm"; return e
    if family=="random_forest": return RandomForestClassifier(bootstrap=True,random_state=seed,n_jobs=WORKERS,**p)
    if family=="xgboost":
        if mode=="balanced": p["scale_pos_weight"]=(len(y_train)-sum(y_train))/max(1,sum(y_train))
        return XGBClassifier(objective="binary:logistic",eval_metric="logloss",tree_method="hist",device="cpu",random_state=seed,n_jobs=WORKERS,**p)
    if family=="catboost":
        if mode=="balanced": p["auto_class_weights"]="Balanced"
        return CatBoostClassifier(loss_function="Logloss",verbose=False,allow_writing_files=False,task_type="CPU",random_seed=seed,thread_count=WORKERS,**p)
    if family=="mlp":
        if not mlp_supports_sample_weight(): raise RuntimeError("MLPClassifier.fit does not support required sample_weight")
        return MLPClassifier(solver="adam",max_iter=500,early_stopping=False,tol=1e-4,random_state=seed,**p)
    raise ValueError(family)

def fit_estimator(estimator, family, X, y, cat_features=None, balanced=False):
    if family=="catboost": return estimator.fit(X,y,cat_features=cat_features)
    if family=="mlp" and balanced: return estimator.fit(X,y,sample_weight=compute_sample_weight("balanced",y))
    return estimator.fit(X,y)
