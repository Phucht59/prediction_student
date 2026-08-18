"""Validation-only selection for thresholds over frozen Hybrid probabilities."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score

def ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); out=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(p>=lo)&(p<hi if hi<1 else p<=hi)
        if m.any(): out += float(m.mean())*abs(float(y[m].mean())-float(p[m].mean()))
    return out
def metrics(df, low, high, umax, smax):
    p=df.risk_probability.to_numpy(float); y=df.outcome.to_numpy(int)
    uncertain=df.hybrid_uncertainty.to_numpy(float)>umax
    disagree=df.seed_disagreement.notna().to_numpy() & (df.seed_disagreement.fillna(0).to_numpy(float)>smax)
    bands=np.where(uncertain|disagree,"BORDERLINE",np.where(p<low,"LOW",np.where(p<high,"BORDERLINE","HIGH")))
    pred=(bands=="HIGH").astype(int); positives=int(y.sum())
    return {"pr_auc":float(average_precision_score(y,p)) if len(np.unique(y))>1 else None,"roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,"brier":float(brier_score_loss(y,p)),"ece":ece(y,p),"high_precision":float(precision_score(y,pred,zero_division=0)),"high_recall":float(recall_score(y,pred,zero_division=0)) if positives else None,"high_f1":float(f1_score(y,pred,zero_division=0)),"high_f2":float(fbeta_score(y,pred,beta=2,zero_division=0)),"low_coverage":float((bands=="LOW").mean()),"borderline_coverage":float((bands=="BORDERLINE").mean()),"high_coverage":float((bands=="HIGH").mean()),"alerts_per_1000":float(pred.mean()*1000),"uncertainty_routed_count":int(uncertain.sum()),"seed_disagreement_routed_count":int(disagree.sum()),"learner_stage_count":int(len(df)),"positive_count":positives,"confusion_high_vs_rest":{"tn":int(((pred==0)&(y==0)).sum()),"fp":int(((pred==1)&(y==0)).sum()),"fn":int(((pred==0)&(y==1)).sum()),"tp":int(((pred==1)&(y==1)).sum())}}
def select(dev, config):
    grid=config["risk_authority"]["stratification"]; c=grid["selection_constraints"]
    candidates=[]
    for low in grid["low_threshold_grid"]:
      for high in grid["high_threshold_grid"]:
       if high-low < grid["minimum_threshold_gap"]: continue
       for umax in config["safety_router"]["maximum_hybrid_uncertainty_grid"]:
        for smax in config["safety_router"]["maximum_seed_disagreement_grid"]:
         m=metrics(dev,low,high,umax,smax); valid=(m["high_precision"]>=c["minimum_high_precision"] and (m["high_recall"] or 0)>=c["minimum_high_recall"] and c["minimum_high_coverage"]<=m["high_coverage"]<=c["maximum_high_coverage"])
         candidates.append((valid,m["high_f2"],m["high_recall"] or 0,m["high_precision"],-m["high_coverage"],(-low,high,umax,smax),{"low":low,"high":high,"umax":umax,"smax":smax,"metrics":m}))
    if not candidates: raise RuntimeError("no threshold candidates")
    valid=[x for x in candidates if x[0]]; chosen=max(valid or candidates,key=lambda x:x[:6]); chosen[6]["gate_status"]="PASS" if valid else "FAIL"; chosen[6]["selection_constraints"]=c; return chosen[6]
def run(root:Path):
    cfg=yaml.safe_load((root/"configs/recommend_hybrid/explainable_v2.yaml").read_text(encoding="utf-8")); src=pd.read_parquet(root/"artifacts/recommend_hybrid/explainable_v2/data/learner_stage_features.parquet"); raw=pd.read_parquet(root/"artifacts/recommend_hybrid/causal/input/landmark_rows.parquet").drop_duplicates(["student_id","course_id","stage"]); raw["outcome"]=raw["target"].astype(int); src=src.merge(raw[["student_id","course_id","stage","outcome"]].rename(columns={"student_id":"student_key","course_id":"course_key"}),on=["student_key","course_key","stage"],validate="one_to_one"); out=root/"artifacts/recommend_hybrid/explainable_v2/risk_policy"; out.mkdir(parents=True,exist_ok=True); report=[]
    for fold in range(3):
        test=src[src.outer_fold==fold]; dev=src[src.outer_fold!=fold]; students=sorted(dev.student_key.astype(str).unique()); val_students=set(students[::5]); val=dev[dev.student_key.astype(str).isin(val_students)]; train=dev[~dev.student_key.astype(str).isin(val_students)]; chosen=select(val,cfg); chosen["outer_fold"]=fold; chosen["train_student_count"]=int(train.student_key.nunique()); chosen["validation_student_count"]=int(val.student_key.nunique()); chosen["test_student_count"]=int(test.student_key.nunique()); chosen["positive_class_definition"]="AT_RISK = Fail or Withdrawn; canonical target=1"; chosen["validation_metrics"]=chosen.pop("metrics"); chosen["test_metrics"]=metrics(test,chosen["low"],chosen["high"],chosen["umax"],chosen["smax"]); chosen["per_stage_metrics"]={s:metrics(test[test.stage==s],chosen["low"],chosen["high"],chosen["umax"],chosen["smax"]) for s in sorted(test.stage.unique())}; chosen["selected_low_threshold"]=chosen.pop("low"); chosen["selected_high_threshold"]=chosen.pop("high"); chosen["selected_maximum_uncertainty"]=chosen.pop("umax"); chosen["selected_maximum_seed_disagreement"]=chosen.pop("smax"); chosen["feature_table_sha256"]=hashlib.sha256((root/"artifacts/recommend_hybrid/explainable_v2/data/learner_stage_features.parquet").read_bytes()).hexdigest(); chosen["runtime_authorized"]=False; (out/f"outer_{fold}.json").write_text(json.dumps(chosen,indent=2,default=str)+"\n",encoding="utf-8"); report.append(chosen)
    (out/"RISK_POLICY_MANIFEST.json").write_text(json.dumps({"status":"COMPLETE","runtime_authorized":False,"outer_folds":3,"positive_class_definition":"AT_RISK = Fail or Withdrawn"},indent=2)+"\n",encoding="utf-8"); return report
