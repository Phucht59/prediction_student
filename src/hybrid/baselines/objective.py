"""Pooled inner-OOF Optuna objective; no outer-test interface exists here."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from .models import suggest, make_estimator, fit_estimator
from .preprocessing import make_preprocessor, feature_columns, catboost_frame
from .registry import get_ranking_score

def run_inner_oof_trial(trial, family, cv_data: pd.DataFrame, seed: int):
    params=suggest(trial,family); fold_rows=[]; start=time.monotonic()
    for fold in sorted(cv_data.inner_fold.unique()):
        train=cv_data[cv_data.inner_fold!=fold]; valid=cv_data[cv_data.inner_fold==fold]
        if set(train.group_id.astype(str)) & set(valid.group_id.astype(str)): raise AssertionError("inner group leakage")
        y_train=train.target.to_numpy(); y_valid=valid.target.to_numpy()
        if len(np.unique(y_valid)) < 2: raise RuntimeError("single-class validation; study blocked")
        predictor_columns, _, _ = feature_columns(train)
        if "inner_fold" in predictor_columns: raise RuntimeError("inner_fold predictor leakage")
        estimator=make_estimator(family,params,seed,y_train)
        if family=="catboost":
            X_train,cats=catboost_frame(train); X_valid,_=catboost_frame(valid)
            fit_estimator(estimator,family,X_train,y_train,cats,params.get("imbalance_mode")=="balanced")
        else:
            prep=make_preprocessor(train); X_train=prep.fit_transform(train); X_valid=prep.transform(valid)
            fit_estimator(estimator,family,X_train,y_train,None,params.get("imbalance_mode")=="balanced")
        score=get_ranking_score(estimator,X_valid)
        if not np.isfinite(score).all(): raise RuntimeError("NaN ranking score")
        fold_rows.append(pd.DataFrame({"record_id":valid.record_id,"group_id":valid.group_id,"inner_fold":fold,"target":y_valid,"ranking_score":score}))
        trial.report(float(average_precision_score(y_valid,score)), fold + 1)
        if fold >= 1 and trial.should_prune():
            import optuna
            raise optuna.TrialPruned()
    oof=pd.concat(fold_rows,ignore_index=True)
    trial.set_user_attr("oof_pr_auc",float(average_precision_score(oof.target,oof.ranking_score)))
    trial.set_user_attr("oof_roc_auc",float(roc_auc_score(oof.target,oof.ranking_score)))
    trial.set_user_attr("fit_time_seconds",time.monotonic()-start)
    return float(average_precision_score(oof.target,oof.ranking_score))
