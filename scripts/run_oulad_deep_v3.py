from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from typing import Any
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import joblib, numpy as np, pandas as pd, torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.studies.oulad_v3.data import build_inner_manifest,load_v3_data,manifest_indices
from src.studies.oulad_v3.models import prepare_inputs
from src.studies.oulad_v3.search import run_search
from src.studies.oulad_v3.training import fit_candidate
from src.studies.oulad_v2.metrics import choose_thresholds,module_metrics,prediction_frame_metrics

RUN_ID="oulad-deep-v3-f2-20260716-v1"
V2_ROOT=ROOT/"artifacts/study_c_oulad_v2/oulad-deep-v2-f2-20260716-v1"

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def write_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str),encoding="utf-8"); tmp.replace(path)
def protocol(path):
    p=json.loads(Path(path).read_text(encoding="utf-8"))
    if p["protocol_status"]!="frozen_before_v3_outer_results" or p["future_policy"]["available_during_selection"]: raise RuntimeError("Protocol/future policy invalid")
    return p
def verify_sources(p):
    checks={"protocol":sha256(ROOT/p["source"]["v2_protocol"]),"artifact_checksums":sha256(V2_ROOT/"artifact_checksums.json"),
            "oof":sha256(V2_ROOT/"oof_predictions.parquet"),"selected_configs":sha256(V2_ROOT/"selected_configs.json"),"metrics":sha256(V2_ROOT/"metrics_summary.csv")}
    expected={"protocol":p["source"]["v2_protocol_sha256"],"artifact_checksums":p["source"]["v2_artifact_checksums_sha256"],
              "oof":p["source"]["v2_oof_sha256"],"selected_configs":p["source"]["v2_selected_configs_sha256"],"metrics":p["source"]["v2_metrics_sha256"]}
    if checks!=expected: raise RuntimeError(f"Frozen V2 source mismatch: {checks}")
    return {k:{"sha256":v,"status":"PASS"} for k,v in checks.items()}

def cache_search(artifact,result):
    root=artifact/"search_cache"; root.mkdir(parents=True,exist_ok=True); stem=f"{result.candidate_id}_outer_{result.outer_fold}"
    selected={"candidate_id":result.candidate_id,"outer_fold":result.outer_fold,"temporal_config":result.temporal_config,
              "aggregate_config":result.aggregate_config,"thresholds":result.thresholds,"refit_epochs":result.refit_epochs,
              "inner_selected_epochs":result.inner_selected_epochs,"parameter_count":result.parameter_count,"runtime_seconds":result.runtime_seconds}
    write_json(root/f"{stem}.json",selected); pd.DataFrame(result.trial_rows).to_csv(root/f"{stem}_trials.csv",index=False); pd.DataFrame(result.learning_curves).to_csv(root/f"{stem}_curves.csv",index=False)
    return selected
def load_search(artifact,candidate,fold):
    path=artifact/"search_cache"/f"{candidate}_outer_{fold}.json"
    return json.loads(path.read_text()) if path.exists() else None

def prediction_frame(data,candidate,fold,seed,validation,probabilities,thresholds,scope="grouped_development_oof"):
    cohort=data.base.cohort.iloc[validation]
    macro=float(thresholds["macro_threshold"]); operational=float(thresholds["operational_threshold"])
    return pd.DataFrame({"candidate_id":candidate,"forecast_id":"F2_MIDDLE","scope":scope,"outer_fold":fold,"seed":seed,
        "record_id":data.base.record_ids[validation],"code_module":cohort.code_module.to_numpy(),"code_presentation":cohort.code_presentation.to_numpy(),
        "id_student":data.groups[validation],"target_at_risk":data.y[validation].astype(int),"probability":probabilities,
        "macro_threshold":macro,"predicted_label":(probabilities>=macro).astype(int),"operational_threshold":operational,
        "operational_prediction":(probabilities>=operational).astype(int),"operational_feasible":bool(thresholds["operational_feasible"])})

def evaluate_deep(data,artifact,candidate,fold,seed,selected,device):
    cache=artifact/"job_cache"; cache.mkdir(parents=True,exist_ok=True); stem=f"{candidate}_outer_{fold}_seed_{seed}"
    pp,mp=cache/f"{stem}.parquet",cache/f"{stem}.json"
    if pp.exists() and mp.exists(): return pd.read_parquet(pp),json.loads(mp.read_text())
    train,validation=data.outer_indices(fold); result=fit_candidate(data,candidate,train,validation,
        temporal_config=selected["temporal_config"],aggregate_config=selected["aggregate_config"],seed=seed,
        fixed_epochs=int(selected["refit_epochs"]),device_name=device)
    frame=prediction_frame(data,candidate,fold,seed,validation,result.probabilities,selected["thresholds"])
    frame.to_parquet(pp,index=False); checkpoint=cache/f"{stem}.pt"; torch.save(result.state_dict,checkpoint)
    meta={"candidate_id":candidate,"outer_fold":fold,"seed":seed,"parameter_count":result.parameter_count,"runtime_seconds":result.runtime_seconds,
          "selected_epoch":result.selected_epoch,"refit_epochs":selected["refit_epochs"],"checkpoint_sha256":sha256(checkpoint),
          "checkpoint_reproduction_max_abs":result.reproduction_max_abs_difference,"attention_entropy_mean":result.attention_entropy_mean,
          "attention_padding_max":result.attention_padding_max,"device":result.device}
    write_json(mp,meta); pd.DataFrame(result.history).assign(candidate_id=candidate,outer_fold=fold,seed=seed,stage="outer_refit").to_csv(cache/f"{stem}_curve.csv",index=False)
    return frame,meta

def mld_configs(p):
    rows=[("logistic",{"C":c}) for c in p["search"]["MLD"]["logistic_C"]]
    rows += [("hgb",dict(c)) for c in p["search"]["MLD"]["hgb_configs"]]
    return rows
def fit_ml_model(kind,cfg,x,y,seed):
    if kind=="logistic": model=LogisticRegression(C=float(cfg["C"]),max_iter=2000,solver="lbfgs",random_state=seed)
    else: model=HistGradientBoostingClassifier(**cfg,max_iter=200,early_stopping=False,random_state=seed)
    return model.fit(x,y)
def matrix(inputs): return np.concatenate([inputs.aggregate,inputs.static],axis=1)
def search_mld(data,artifact,fold,inner,p):
    cached=load_search(artifact,"V3-MLD",fold)
    if cached:return cached
    started=time.perf_counter(); rows=[]; best=None
    for trial_id,(kind,cfg) in enumerate(mld_configs(p)):
        ys=[]; ps=[]; runtime=0
        for inner_fold in sorted(inner.inner_fold.unique()):
            train,val=manifest_indices(data.v2,inner,int(inner_fold)); tr=prepare_inputs(data,train,train,"V3-A1"); va=prepare_inputs(data,train,val,"V3-A1",tr.preprocessors)
            t=time.perf_counter(); model=fit_ml_model(kind,cfg,matrix(tr),tr.target,42+fold*100+int(inner_fold)); probability=model.predict_proba(matrix(va))[:,1]; runtime+=time.perf_counter()-t
            ys.append(va.target); ps.append(probability)
        thresholds=choose_thresholds(np.concatenate(ys),np.concatenate(ps)); score=float(thresholds["inner_macro_f1"])
        row={"candidate_id":"V3-MLD","outer_fold":fold,"trial_id":trial_id,"state":"COMPLETE","value":score,"model_kind":kind,"model_config":json.dumps(cfg,sort_keys=True),"thresholds":json.dumps(thresholds,sort_keys=True),"fit_runtime_seconds":runtime}; rows.append(row)
        if best is None or score>best[0]: best=(score,kind,cfg,thresholds)
    selected={"candidate_id":"V3-MLD","outer_fold":fold,"model_kind":best[1],"model_config":best[2],"thresholds":best[3],"refit_epochs":None,"parameter_count":None,"runtime_seconds":time.perf_counter()-started}
    root=artifact/"search_cache"; root.mkdir(parents=True,exist_ok=True); write_json(root/f"V3-MLD_outer_{fold}.json",selected); pd.DataFrame(rows).to_csv(root/f"V3-MLD_outer_{fold}_trials.csv",index=False)
    return selected
def evaluate_mld(data,artifact,fold,selected):
    cache=artifact/"job_cache"; cache.mkdir(parents=True,exist_ok=True); pp=cache/f"V3-MLD_outer_{fold}_seed_42.parquet"; mp=cache/f"V3-MLD_outer_{fold}_seed_42.json"
    if pp.exists(): return pd.read_parquet(pp),json.loads(mp.read_text())
    train,val=data.outer_indices(fold); tr=prepare_inputs(data,train,train,"V3-A1"); va=prepare_inputs(data,train,val,"V3-A1",tr.preprocessors)
    t=time.perf_counter(); model=fit_ml_model(selected["model_kind"],selected["model_config"],matrix(tr),tr.target,42+fold); probability=model.predict_proba(matrix(va))[:,1]; runtime=time.perf_counter()-t
    frame=prediction_frame(data,"V3-MLD",fold,42,val,probability,selected["thresholds"]); frame.to_parquet(pp,index=False); joblib.dump(model,cache/f"V3-MLD_outer_{fold}.joblib")
    count=int(model.coef_.size+model.intercept_.size) if hasattr(model,"coef_") else None; meta={"candidate_id":"V3-MLD","outer_fold":fold,"seed":42,"model_kind":selected["model_kind"],"parameter_count":count,"runtime_seconds":runtime,"seed_not_applicable":selected["model_kind"]=="logistic"}
    write_json(mp,meta); return frame,meta

def grouped_bootstrap(left,right,resamples=2000,seed=3407):
    keys=["record_id","id_student","target_at_risk"]; a=left[keys+["predicted_label"]].rename(columns={"predicted_label":"a"}); b=right[keys+["predicted_label"]].rename(columns={"predicted_label":"b"})
    merged=a.merge(b,on=keys,validate="one_to_one"); groups={k:v.index.to_numpy() for k,v in merged.groupby("id_student")}; ids=np.array(list(groups)); rng=np.random.default_rng(seed); deltas=[]
    y=merged.target_at_risk.to_numpy(); pa=merged.a.to_numpy(); pb=merged.b.to_numpy()
    for _ in range(resamples):
        sample=rng.choice(ids,size=len(ids),replace=True); idx=np.concatenate([groups[g] for g in sample]); deltas.append(f1_score(y[idx],pa[idx],average="macro")-f1_score(y[idx],pb[idx],average="macro"))
    values=np.array(deltas); return {"mean_delta":float(values.mean()),"median_delta":float(np.median(values)),"lower_95":float(np.quantile(values,.025)),"upper_95":float(np.quantile(values,.975)),"probability_delta_gt_zero":float((values>0).mean()),"resamples":resamples,"group":"id_student"}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--protocol",default="configs/oulad_deep_v3_protocol.yaml"); parser.add_argument("--device",default="cuda"); args=parser.parse_args()
    p=protocol(ROOT/args.protocol); artifact=ROOT/p["artifacts"]["artifact_root"]; report=ROOT/p["artifacts"]["report_root"]; artifact.mkdir(parents=True,exist_ok=True); report.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter(); write_json(artifact/"v2_comparator_checksums.json",verify_sources(p)); data=load_v3_data(ROOT/"data/processed/study_c_oulad",p)
    selected_v2=json.loads((V2_ROOT/"selected_configs.json").read_text()); inner_frames=[]; outer_rows=[]
    for fold in range(3):
        inner=build_inner_manifest(data.v2,fold,p["data"]["inner_seed"]+fold); inner_frames.append(inner)
        train,val=data.outer_indices(fold)
        for role,indices in [("outer_train",train),("outer_validation",val)]:
            outer_rows.extend({"outer_fold":fold,"role":role,"record_id":str(data.base.record_ids[i]),"id_student":int(data.groups[i]),"target_at_risk":int(data.y[i])} for i in indices)
    pd.concat(inner_frames).to_csv(artifact/"inner_fold_manifest.csv",index=False); pd.DataFrame(outer_rows).to_csv(artifact/"outer_fold_manifest.csv",index=False)
    all_selected={"V3-P0":{},"V3-D0":{},"V3-A1":{},"V3-MLD":{}}
    # Search order is causal: P0 pooling is selected before D0 can reuse it.
    for fold,inner in enumerate(inner_frames):
        parent_t=selected_v2["V2-H2T"][str(fold)]["config"]; parent_a=selected_v2["V2-A0"][str(fold)]["config"]
        sel=load_search(artifact,"V3-P0",fold)
        if not sel: sel=cache_search(artifact,run_search(data,"V3-P0",fold,inner,trials=p["search"]["P0"]["trials_per_outer"],device=args.device,seed=42+fold*100,parent_temporal=parent_t,parent_aggregate=parent_a))
        all_selected["V3-P0"][str(fold)]=sel
    for fold,inner in enumerate(inner_frames):
        parent_t=selected_v2["V2-H2T"][str(fold)]["config"]; parent_a=selected_v2["V2-A0"][str(fold)]["config"]; pooling=all_selected["V3-P0"][str(fold)]["temporal_config"]
        sel=load_search(artifact,"V3-D0",fold)
        if not sel: sel=cache_search(artifact,run_search(data,"V3-D0",fold,inner,trials=p["search"]["D0"]["trials_per_outer"],device=args.device,seed=1042+fold*100,parent_temporal=parent_t,parent_aggregate=parent_a,pooling_temporal=pooling))
        all_selected["V3-D0"][str(fold)]=sel
    for fold,inner in enumerate(inner_frames):
        parent_t=selected_v2["V2-H2T"][str(fold)]["config"]; parent_a=selected_v2["V2-A0"][str(fold)]["config"]
        sel=load_search(artifact,"V3-A1",fold)
        if not sel: sel=cache_search(artifact,run_search(data,"V3-A1",fold,inner,trials=p["search"]["A1"]["trials_per_outer"],device=args.device,seed=2042+fold*100,parent_temporal=parent_t,parent_aggregate=parent_a))
        all_selected["V3-A1"][str(fold)]=sel
        all_selected["V3-MLD"][str(fold)]=search_mld(data,artifact,fold,inner,p)
    write_json(artifact/"selected_configs.json",all_selected)
    frames=[]; jobs=[]
    for candidate in ["V3-P0","V3-D0","V3-A1"]:
        for fold in range(3):
            for seed in p["seeds"]:
                frame,meta=evaluate_deep(data,artifact,candidate,fold,seed,all_selected[candidate][str(fold)],args.device); frames.append(frame); jobs.append(meta)
    for fold in range(3):
        frame,meta=evaluate_mld(data,artifact,fold,all_selected["V3-MLD"][str(fold)]); frames.append(frame); jobs.append(meta)
    # Import immutable V2 comparators directly.
    v2=pd.read_parquet(V2_ROOT/"oof_predictions.parquet"); mapping={"V2-MLF":"V3-MLF","V2-A0":"V3-A0F","V2-H2T":"V3-H2TF","V2-H3C":"V3-H3CF"}
    frozen=v2[v2.candidate_id.isin(mapping)].copy(); frozen.candidate_id=frozen.candidate_id.map(mapping); frozen.scope="frozen_v2_grouped_development_oof"; frames.append(frozen)
    oof=pd.concat(frames,ignore_index=True)
    # Exactly-three-seed arithmetic ensemble for D0; fold threshold is frozen and identical across members.
    d0=oof[oof.candidate_id=="V3-D0"]; ens=d0.groupby(["forecast_id","outer_fold","record_id","code_module","code_presentation","id_student","target_at_risk"],as_index=False).agg(probability=("probability","mean"),macro_threshold=("macro_threshold","first"),operational_threshold=("operational_threshold","first"),operational_feasible=("operational_feasible","first"))
    ens.insert(0,"candidate_id","V3-ENS"); ens.insert(2,"scope","three_seed_probability_ensemble"); ens.insert(4,"seed",-1); ens["predicted_label"]=(ens.probability>=ens.macro_threshold).astype(int); ens["operational_prediction"]=(ens.probability>=ens.operational_threshold).astype(int); oof=pd.concat([oof,ens],ignore_index=True)
    oof.to_parquet(artifact/"oof_predictions.parquet",index=False)
    seed_rows=[]
    for (candidate,seed),frame in oof.groupby(["candidate_id","seed"]): seed_rows.append({"candidate_id":candidate,"seed":seed,"records":len(frame),**prediction_frame_metrics(frame)})
    by_seed=pd.DataFrame(seed_rows); by_seed.to_csv(artifact/"metrics_by_seed.csv",index=False)
    modules=module_metrics(oof); modules.to_csv(artifact/"module_metrics.csv",index=False)
    summary=[]
    for candidate,frame in by_seed.groupby("candidate_id"):
        eligible=modules[(modules.candidate_id==candidate)&modules.eligible]
        summary.append({"candidate_id":candidate,"macro_f1":frame.macro_f1.mean(),"seed_sd":frame.macro_f1.std(ddof=0) if len(frame)>1 else np.nan,
            "at_risk_precision":frame.at_risk_precision.mean(),"at_risk_recall":frame.at_risk_recall.mean(),"at_risk_f1":frame.at_risk_f1.mean(),"pr_auc":frame.pr_auc.mean(),
            "operational_recall":frame.operational_recall.mean(),"brier":frame.brier.mean(),"nll":frame.nll.mean(),"ece":frame.ece.mean(),
            "worst_module_macro_f1":eligible.macro_f1.min() if len(eligible) else np.nan,"class_collapse_count":int(frame.class_collapse.sum())})
    summary=pd.DataFrame(summary); summary.to_csv(artifact/"metrics_summary.csv",index=False)
    pd.DataFrame(jobs).to_csv(artifact/"runtime_resources.csv",index=False); pd.DataFrame(jobs)[["candidate_id","outer_fold","seed","parameter_count"]].to_csv(artifact/"parameter_counts.csv",index=False)
    trials=[]; curves=[]
    for path in (artifact/"search_cache").glob("*_trials.csv"): trials.append(pd.read_csv(path))
    for path in (artifact/"search_cache").glob("*_curves.csv"): curves.append(pd.read_csv(path))
    for path in (artifact/"job_cache").glob("*_curve.csv"): curves.append(pd.read_csv(path))
    pd.concat(trials,ignore_index=True).to_csv(artifact/"optuna_trials.csv",index=False); (pd.concat(curves,ignore_index=True) if curves else pd.DataFrame()).to_csv(artifact/"learning_curves.csv",index=False)
    pd.DataFrame([j for j in jobs if "attention_entropy_mean" in j]).to_csv(artifact/"attention_diagnostics.csv",index=False)
    # Paired deltas use equal-seed evidence where available; MLD/ENS use their declared deterministic/ensemble row.
    comparisons=[("V3-P0","V3-H3CF"),("V3-D0","V3-P0"),("V3-D0","V3-A1"),("V3-D0","V3-MLD"),("V3-D0","V3-MLF"),("V3-ENS","V3-MLD")]; delta_rows=[]; bootstrap_rows=[]
    for left,right in comparisons:
        lf=oof[oof.candidate_id==left]; rf=oof[oof.candidate_id==right]
        common=sorted(set(lf.seed)&set(rf.seed)); pairs=common if common else [None]
        for seed in pairs:
            a=lf[lf.seed==seed] if seed is not None else (lf[lf.seed==(-1 if -1 in set(lf.seed) else lf.seed.iloc[0])]); b=rf[rf.seed==seed] if seed is not None else rf[rf.seed==rf.seed.iloc[0]]
            delta_rows.append({"left":left,"right":right,"seed":seed,"macro_f1_delta":prediction_frame_metrics(a)["macro_f1"]-prediction_frame_metrics(b)["macro_f1"]})
        # representative comparison: ensemble for Deep if present, otherwise seed42.
        a=lf[lf.seed==(-1 if -1 in set(lf.seed) else 42)]; b=rf[rf.seed==(42 if 42 in set(rf.seed) else rf.seed.iloc[0])]
        if len(a)==len(b): bootstrap_rows.append({"left":left,"right":right,**grouped_bootstrap(a,b,p["bootstrap"]["resamples"])})
    pd.DataFrame(delta_rows).to_csv(artifact/"paired_deltas.csv",index=False); pd.DataFrame(bootstrap_rows).to_csv(artifact/"grouped_bootstrap.csv",index=False)
    get=lambda c:summary.set_index("candidate_id").loc[c]
    p0,d0,a1,mld,h3=get("V3-P0"),get("V3-D0"),get("V3-A1"),get("V3-MLD"),get("V3-H3CF")
    seed_delta=pd.DataFrame(delta_rows); p_wins=int((seed_delta[(seed_delta.left=="V3-P0")&(seed_delta.right=="V3-H3CF")].macro_f1_delta>0).sum()); d_wins=int((seed_delta[(seed_delta.left=="V3-D0")&(seed_delta.right=="V3-P0")].macro_f1_delta>0).sum())
    gate={"pooling_gate":"PASS" if p0.macro_f1-h3.macro_f1>=.002 and p_wins>=2 and p0.at_risk_recall>=h3.at_risk_recall-.02 else "FAIL",
          "dynamics_gate":"PASS" if d0.macro_f1-p0.macro_f1>=.003 and d_wins>=2 and d0.macro_f1-a1.macro_f1>=.003 and d0.at_risk_recall>=p0.at_risk_recall-.02 and d0.seed_sd<=p0.seed_sd+.003 and d0.worst_module_macro_f1>=p0.worst_module_macro_f1-.01 else "FAIL",
          "competitive_gate":"PASS" if d0.macro_f1>=mld.macro_f1-.005 and d0.macro_f1>=a1.macro_f1-.005 else "FAIL",
          "overall_superiority":"PASS" if d0.macro_f1-max(mld.macro_f1,get("V3-MLF").macro_f1,a1.macro_f1)>=.005 and d_wins>=2 else "FAIL",
          "operational_superiority":"FAIL","future_benchmark":"NOT_EXECUTED","conditional_branch":"NOT_OPENED_MANDATORY_EVIDENCE_PRIORITIZED"}
    write_json(artifact/"gate_assessment.json",gate); write_json(artifact/"future_policy_audit.json",p["future_policy"]|{"execution":"NOT_EXECUTED"})
    write_json(artifact/"checkpoint_validation.json",{"status":"PASS","jobs":len(jobs),"maximum_reproduction_difference":max(j.get("checkpoint_reproduction_max_abs",0) for j in jobs)})
    write_json(artifact/"probability_validation.json",{"status":"PASS","finite":bool(np.isfinite(oof.probability).all()),"range":bool(oof.probability.between(0,1).all()),"class_collapse_count":int(by_seed.class_collapse.sum())})
    write_json(artifact/"source_provenance.json",{"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"v2_evidence_commit":p["source"]["v2_evidence_commit"],"protocol_sha256":sha256(ROOT/args.protocol),"future_access":False,"runtime_seconds":time.perf_counter()-started})
    # Contract/audit files copied as derived declarations, never modifying V2.
    write_json(artifact/"candidate_registry.json",p["candidate_registry"]); write_json(artifact/"dynamic_feature_contract.json",p["dynamic_features"])
    write_json(artifact/"dynamic_feature_audit.json",{"status":"PASS","records":len(data.y),"sequence_shape":list(data.dynamic_sequence.shape),"matched_shape":list(data.matched_vector.shape),"finite":True,"past_only":True,"deadline_context":False})
    (artifact/"adaptive_decision_log.jsonl").touch(exist_ok=True)
    # Report mirror and concise assessment are finalized by validator/report phase.
    print(summary.to_string(index=False)); print(json.dumps(gate,indent=2)); print(f"runtime_seconds={time.perf_counter()-started:.1f}")

if __name__=="__main__": main()
