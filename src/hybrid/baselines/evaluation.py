"""Inner-CV construction explicitly excludes every outer-test record."""
from __future__ import annotations
import pandas as pd
import time
from sklearn.metrics import average_precision_score, roc_auc_score
from .models import make_estimator, fit_estimator
from .preprocessing import make_preprocessor, catboost_frame, feature_columns
from .registry import get_ranking_score

def build_inner_cv_data(data, outer, inner, outer_fold):
    test_ids=set(outer.loc[outer.outer_fold==outer_fold,"record_id"].astype(str))
    assignments=inner.loc[inner.outer_fold==outer_fold].copy()
    assignments["record_id"]=assignments.record_id.astype(str)
    # Group identity is retained from the tabular view; assignments only supply
    # the frozen inner-fold label, avoiding duplicate group columns on merge.
    result=data.merge(assignments[["record_id", "inner_fold"]],on="record_id",how="inner",validate="one_to_one")
    if set(result.record_id.astype(str)) & test_ids: raise AssertionError("outer-test record entered Phase 2 study data")
    return result

def reproduce_fixed_params_inner_oof(family, params, cv_data, seed):
    """Canonical non-Optuna rerun of a selected trial on frozen inner folds."""
    rows=[]; dimensions={}; started=time.monotonic(); raw_count=None
    for fold in sorted(cv_data.inner_fold.unique()):
        train=cv_data[cv_data.inner_fold!=fold]; valid=cv_data[cv_data.inner_fold==fold]
        if set(train.group_id.astype(str)) & set(valid.group_id.astype(str)): raise RuntimeError('inner group leakage')
        columns,_,_=feature_columns(train)
        if 'inner_fold' in columns: raise RuntimeError('control metadata entered predictors')
        raw_count=len(columns); estimator=make_estimator(family,params,seed,train.target.to_numpy())
        if family=='catboost':
            X_train,cats=catboost_frame(train); X_valid,_=catboost_frame(valid); dimensions[str(fold)]=len(columns); fit_estimator(estimator,family,X_train,train.target.to_numpy(),cats,params.get('imbalance_mode')=='balanced')
        else:
            prep=make_preprocessor(train); X_train=prep.fit_transform(train); X_valid=prep.transform(valid); dimensions[str(fold)]=int(X_train.shape[1]); fit_estimator(estimator,family,X_train,train.target.to_numpy(),None,params.get('imbalance_mode')=='balanced')
        score=get_ranking_score(estimator,X_valid)
        rows.append(pd.DataFrame({'record_id':valid.record_id,'group_id':valid.group_id,'inner_fold':fold,'target':valid.target,'ranking_score':score}))
    oof=pd.concat(rows,ignore_index=True)
    per_pr={str(f):float(average_precision_score(g.target,g.ranking_score)) for f,g in oof.groupby('inner_fold')}
    per_roc={str(f):float(roc_auc_score(g.target,g.ranking_score)) for f,g in oof.groupby('inner_fold')}
    return oof, {'pooled_pr_auc':float(average_precision_score(oof.target,oof.ranking_score)), 'pooled_roc_auc':float(roc_auc_score(oof.target,oof.ranking_score)), 'per_inner_fold_pr_auc':per_pr, 'per_inner_fold_roc_auc':per_roc, 'raw_predictor_count':raw_count, 'transformed_feature_count_by_inner_fold':dimensions, 'reproduction_fit_time_seconds':time.monotonic()-started}
