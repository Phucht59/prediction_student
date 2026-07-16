from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
from typing import Any
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.studies.oulad_v2.data import load_v2_data, manifest_indices
from src.studies.oulad_v2.metrics import choose_thresholds
from src.studies.oulad_v2.training import fit_candidate as fit_v2
from src.studies.oulad_v3.data import load_v3_data
from src.studies.oulad_v3.training import fit_candidate as fit_v3
from src.studies.oulad_v3_closure.fairness import DECLARED_SEEDS, ensemble_outer_predictions, grouped_bootstrap_pair, metrics_with_modules, sha256, validate_seed_coverage

V3_ROOT=ROOT/"artifacts/study_c_oulad_v3/oulad-deep-v3-f2-20260716-v1"
V2_ROOT=ROOT/"artifacts/study_c_oulad_v2/oulad-deep-v2-f2-20260716-v1"
SOURCE_MAP={"V3-A0F-ENS":"V3-A0F","V3-H2TF-ENS":"V3-H2TF","V3-H3CF-ENS":"V3-H3CF","V3-P0-ENS":"V3-P0","V3-D0-ENS":"V3-D0","V3-A1-ENS":"V3-A1"}

def write_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str),encoding="utf-8"); tmp.replace(path)
def load_protocol(path):
    p=json.loads(Path(path).read_text())
    if p["protocol_status"]!="frozen_before_fair_recalculation_and_database_write": raise RuntimeError("Closure protocol not frozen")
    return p
def verify_v3(p):
    checks={"protocol":sha256(ROOT/p["source"]["v3_protocol_path"]),"artifact_manifest":sha256(V3_ROOT/"artifact_checksums.json"),
            "oof":sha256(V3_ROOT/"oof_predictions.parquet"),"selected_configs":sha256(V3_ROOT/"selected_configs.json"),"metrics":sha256(V3_ROOT/"metrics_summary.csv")}
    expected={"protocol":p["source"]["v3_protocol_sha256"],"artifact_manifest":p["source"]["v3_artifact_manifest_sha256"],
              "oof":p["source"]["v3_oof_sha256"],"selected_configs":p["source"]["v3_selected_configs_sha256"],"metrics":p["source"]["v3_metrics_sha256"]}
    if checks!=expected: raise RuntimeError(f"Frozen V3 hash mismatch: {checks}")
    return {key:{"sha256":value,"status":"PASS"} for key,value in checks.items()}

def replay_candidate(candidate,fold,seed,inner_manifest,data_v2,data_v3,selected_v2,selected_v3,device):
    probabilities=[]; rows=[]; runtime=0.; checkpoint_max=0.
    for inner_fold in sorted(inner_manifest.inner_fold.unique()):
        train,validation=manifest_indices(data_v2,inner_manifest,int(inner_fold))
        if candidate=="V3-A0F-ENS":
            selection=selected_v2["V2-A0"][str(fold)]; result=fit_v2(data_v2,"V2-A0",train,validation,temporal_config=None,aggregate_config=selection["config"],seed=seed,fixed_epochs=int(selection["refit_epochs"]),device_name=device)
        elif candidate=="V3-H2TF-ENS":
            selection=selected_v2["V2-H2T"][str(fold)]; result=fit_v2(data_v2,"V2-H2T",train,validation,temporal_config=selection["config"],aggregate_config=None,seed=seed,fixed_epochs=int(selection["refit_epochs"]),device_name=device)
        elif candidate=="V3-H3CF-ENS":
            h2=selected_v2["V2-H2T"][str(fold)]; a0=selected_v2["V2-A0"][str(fold)]; result=fit_v2(data_v2,"V2-H3C",train,validation,temporal_config=h2["config"],aggregate_config=a0["config"],seed=seed,fixed_epochs=int(h2["refit_epochs"]),device_name=device)
        else:
            base=candidate.replace("-ENS",""); selection=selected_v3[base][str(fold)]; result=fit_v3(data_v3,base,train,validation,temporal_config=selection["temporal_config"],aggregate_config=selection["aggregate_config"],seed=seed,fixed_epochs=int(selection["refit_epochs"]),device_name=device)
        runtime+=result.runtime_seconds; checkpoint_max=max(checkpoint_max,result.reproduction_max_abs_difference)
        probabilities.extend(result.probabilities.tolist())
        for index,probability in zip(validation,result.probabilities): rows.append({"candidate_id":candidate,"outer_fold":fold,"inner_fold":int(inner_fold),"seed":seed,"record_id":str(data_v2.base.record_ids[index]),"target_at_risk":int(data_v2.y[index]),"probability":float(probability)})
    return pd.DataFrame(rows),{"candidate_id":candidate,"outer_fold":fold,"seed":seed,"fits":2,"runtime_seconds":runtime,"checkpoint_reproduction_max_abs":checkpoint_max}

def replay_or_load(artifact,candidate,fold,seed,*args):
    cache=artifact/"threshold_replay_cache"; cache.mkdir(parents=True,exist_ok=True); path=cache/f"{candidate}_outer_{fold}_seed_{seed}.parquet"; meta=cache/f"{candidate}_outer_{fold}_seed_{seed}.json"
    if path.exists() and meta.exists(): return pd.read_parquet(path),json.loads(meta.read_text())
    frame,metadata=replay_candidate(candidate,fold,seed,*args); frame.to_parquet(path,index=False); write_json(meta,metadata); return frame,metadata

def reconstruct_thresholds(artifact,p,data_v2,data_v3,selected_v2,selected_v3,device):
    inner_all=pd.read_csv(V3_ROOT/"inner_fold_manifest.csv",dtype={"record_id":str}); threshold_rows=[]; coverage=[]; replays=[]
    for candidate in SOURCE_MAP:
        for fold in range(3):
            manifest=inner_all.loc[inner_all.outer_fold==fold].copy(); seed_frames=[]
            for seed in DECLARED_SEEDS:
                frame,metadata=replay_or_load(artifact,candidate,fold,seed,manifest,data_v2,data_v3,selected_v2,selected_v3,device); seed_frames.append(frame); replays.append(metadata)
            combined=pd.concat(seed_frames,ignore_index=True)
            counts=combined.groupby(["record_id","inner_fold"]).agg(seeds=("seed","nunique"),labels=("target_at_risk","nunique"),probabilities=("probability","size")).reset_index()
            if not ((counts.seeds==3)&(counts.labels==1)&(counts.probabilities==3)).all(): raise RuntimeError(f"Inner ensemble parity failed: {candidate}/{fold}")
            ensemble=combined.groupby(["record_id","inner_fold","target_at_risk"],as_index=False).probability.mean()
            thresholds=choose_thresholds(ensemble.target_at_risk.to_numpy(),ensemble.probability.to_numpy(),p["threshold_reconstruction"]["operational_precision_constraint"])
            threshold_rows.append({"candidate_id":candidate,"outer_fold":fold,"prediction_contract":"pooled_inner_oof_three_seed_probability_ensemble",**thresholds})
            coverage.append({"candidate_id":candidate,"outer_fold":fold,"inner_records":len(ensemble),"seed_rows":len(combined),"declared_seed_count":3,"record_alignment":"PASS","label_alignment":"PASS"})
    return pd.DataFrame(threshold_rows),pd.DataFrame(coverage),replays

def metric_row(candidate,contract,frame): return {"candidate_id":candidate,"prediction_contract":contract,"records":len(frame),**metrics_with_modules(frame)}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--protocol",default="configs/oulad_v3_fair_db_closure_protocol.yaml"); parser.add_argument("--device",default="cuda"); args=parser.parse_args()
    p=load_protocol(ROOT/args.protocol); artifact=ROOT/p["artifacts"]["artifact_root"]; report=ROOT/p["artifacts"]["report_root"]; artifact.mkdir(parents=True,exist_ok=True); report.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    write_json(artifact/"v3_artifact_checksums.json",verify_v3(p)); shutil.copy2(ROOT/args.protocol,artifact/"resolved_protocol.yaml")
    v2_protocol=json.loads((ROOT/"configs/oulad_deep_v2_protocol.yaml").read_text()); v3_protocol=json.loads((ROOT/"configs/oulad_deep_v3_protocol.yaml").read_text())
    data_v2=load_v2_data(ROOT/"data/processed/study_c_oulad",v2_protocol); data_v3=load_v3_data(ROOT/"data/processed/study_c_oulad",v3_protocol)
    selected_v2=json.loads((V2_ROOT/"selected_configs.json").read_text()); selected_v3=json.loads((V3_ROOT/"selected_configs.json").read_text())
    thresholds,coverage,replays=reconstruct_thresholds(artifact,p,data_v2,data_v3,selected_v2,selected_v3,args.device)
    thresholds.to_csv(artifact/"ensemble_thresholds.csv",index=False); coverage.to_csv(artifact/"ensemble_prediction_coverage.csv",index=False)
    source=pd.read_parquet(V3_ROOT/"oof_predictions.parquet"); parity=[]; ensemble_frames=[]
    for ensemble_candidate,base in SOURCE_MAP.items():
        parity.append(validate_seed_coverage(source,base)); ensemble_frames.append(ensemble_outer_predictions(source,base,ensemble_candidate,thresholds))
    deterministic=[]
    for candidate in ["V3-MLF","V3-MLD"]:
        frame=source.loc[source.candidate_id==candidate].copy(); frame["prediction_contract"]="deterministic"; frame["prediction_variant"]="deterministic_registered_prediction"; deterministic.append(frame)
    fair=pd.concat(ensemble_frames+deterministic,ignore_index=True); fair.to_parquet(artifact/"ensemble_oof_predictions.parquet",index=False)
    # Contract A and B are kept distinct from the ensemble table.
    single_rows=[]
    for base in SOURCE_MAP.values():
        for seed in DECLARED_SEEDS:
            frame=source[(source.candidate_id==base)&(source.seed==seed)]; single_rows.append({"seed":seed,**metric_row(base,"single_seed",frame)})
    single=pd.DataFrame(single_rows); single.to_csv(artifact/"single_seed_metrics.csv",index=False)
    mean_rows=[]
    numeric=[column for column in single.columns if column not in {"candidate_id","prediction_contract","seed"} and pd.api.types.is_numeric_dtype(single[column])]
    for candidate,frame in single.groupby("candidate_id"):
        row={"candidate_id":candidate,"prediction_contract":"mean_of_seed_metrics","declared_seeds":"42,2026,3407"}
        for column in numeric: row[column]=frame[column].mean(); row[f"{column}_sd"]=frame[column].std(ddof=0)
        mean_rows.append(row)
    pd.DataFrame(mean_rows).to_csv(artifact/"mean_seed_metrics.csv",index=False)
    ensemble_metrics=pd.DataFrame([metric_row(candidate,"probability_ensemble" if candidate.endswith("-ENS") else "deterministic",frame) for candidate,frame in fair.groupby("candidate_id")]); ensemble_metrics.to_csv(artifact/"ensemble_metrics.csv",index=False)
    comparisons=p["comparisons"]["primary"]+p["comparisons"]["secondary"]; summary=ensemble_metrics.set_index("candidate_id"); deltas=[]
    for left,right in comparisons:
        deltas.append({"left_candidate":left,"right_candidate":right,"prediction_contract":"fair_ensemble_closure","macro_f1_delta":summary.loc[left,"macro_f1"]-summary.loc[right,"macro_f1"],"pr_auc_delta":summary.loc[left,"pr_auc"]-summary.loc[right,"pr_auc"],"at_risk_recall_delta":summary.loc[left,"at_risk_recall"]-summary.loc[right,"at_risk_recall"],"operational_recall_delta":summary.loc[left,"operational_recall"]-summary.loc[right,"operational_recall"]})
    pd.DataFrame(deltas).to_csv(artifact/"paired_deltas_fair.csv",index=False); pd.DataFrame(deltas).to_csv(artifact/"fair_comparison_summary.csv",index=False)
    bootstrap=[]
    for index,(left,right) in enumerate(comparisons):
        bootstrap.extend(grouped_bootstrap_pair(fair[fair.candidate_id==left],fair[fair.candidate_id==right],resamples=int(p["bootstrap"]["resamples"]),seed=int(p["bootstrap"]["seed"])+index,superiority_margin=float(p["bootstrap"]["superiority_margin"])))
    pd.DataFrame(bootstrap).to_csv(artifact/"grouped_bootstrap_fair.csv",index=False)
    strongest=max(["V3-A0F-ENS","V3-A1-ENS","V3-MLD","V3-MLF"],key=lambda c:summary.loc[c,"macro_f1"]); d0=float(summary.loc["V3-D0-ENS","macro_f1"]); delta=d0-float(summary.loc[strongest,"macro_f1"])
    primary_boot=pd.DataFrame(bootstrap); boot=primary_boot[(primary_boot.left_candidate=="V3-D0-ENS")&(primary_boot.right_candidate==strongest)&(primary_boot.metric=="macro_f1")].iloc[0]
    if delta>=.005 and boot.lower_95>=0: verdict="OVERALL_SUPERIORITY"
    elif delta>0 and boot.lower_95>0: verdict="POSITIVE_EXPLORATORY_LEAD"
    elif abs(delta)<.005: verdict="PRACTICAL_TIE"
    else: verdict="NOT_SUPPORTED"
    verdict_payload={"verdict":verdict,"old_v3_verdict":"PRACTICAL_TIE","strongest_fair_comparator":strongest,"d0_ensemble_macro_f1":d0,"comparator_macro_f1":float(summary.loc[strongest,"macro_f1"]),"delta":delta,"superiority_margin":.005,"future_benchmark":"NOT_EXECUTED"}; write_json(artifact/"verdict.json",verdict_payload)
    write_json(artifact/"superseded_v3_comparisons.json",{"status":"historical_v3_mixed_contract_result","source":"artifacts/study_c_oulad_v3/oulad-deep-v3-f2-20260716-v1/grouped_bootstrap.csv","reason":"some old rows compared probability ensemble against single-seed or mean-of-metric evidence","eligible_for_closure_verdict":False})
    write_json(artifact/"candidate_registry.json",p["candidate_registry"]); write_json(artifact/"prediction_contract_registry.json",p["prediction_contracts"])
    write_json(artifact/"fairness_audit.json",{"status":"PASS","parity":parity,"inner_threshold_coverage":"PASS","outer_labels_used_for_threshold":False,"future_access":False,"replay_label":"threshold-reconstruction replay","replay_jobs":len(replays),"failed_replays":0,"runtime_seconds":time.perf_counter()-started})
    write_json(artifact/"source_provenance.json",{"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"v3_evidence_commit":p["source"]["v3_evidence_commit"],"protocol_sha256":sha256(ROOT/args.protocol),"v3_oof_sha256":sha256(V3_ROOT/"oof_predictions.parquet"),"future_access":False,"threshold_retraining":False,"threshold_reconstruction_replays":len(replays),"runtime_seconds":time.perf_counter()-started})
    print(ensemble_metrics[["candidate_id","macro_f1","at_risk_precision","at_risk_recall","pr_auc","operational_recall"]].to_string(index=False)); print(json.dumps(verdict_payload,indent=2))
if __name__=="__main__": main()
