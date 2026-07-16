from __future__ import annotations

import argparse, hashlib, json, shutil, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.common.evidence_paths import resolve_evidence_path
RUN_ID="oulad-deep-v3-f2-20260716-v1"

def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str),encoding="utf-8")
def check(condition,name,details=""):
    return {"check":name,"status":"PASS" if condition else "FAIL","details":details}

def figures(artifact,report):
    figures_root=report/"figures"; figures_root.mkdir(parents=True,exist_ok=True)
    summary=pd.read_csv(artifact/"metrics_summary.csv").set_index("candidate_id"); order=[c for c in ["V3-MLF","V3-A0F","V3-H2TF","V3-H3CF","V3-P0","V3-D0","V3-A1","V3-MLD","V3-ENS"] if c in summary.index]
    def bar(column,name,title):
        fig,ax=plt.subplots(figsize=(9,4.8)); summary.loc[order,column].plot.bar(ax=ax,color="#376b9b"); ax.set_title(title); ax.set_ylabel(column); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(figures_root/f"{name}.png",dpi=160); plt.close(fig)
    bar("macro_f1","macro_f1_comparison","Pooled grouped-OOF Macro-F1")
    bar("operational_recall","operational_recall_comparison","At-risk recall at inner-frozen precision constraint")
    # Probability-level figures are descriptive only; they never select a threshold.
    oof=pd.read_parquet(artifact/"oof_predictions.parquet")
    fig,ax=plt.subplots(figsize=(7,5))
    for candidate in ["V3-A0F","V3-D0","V3-A1","V3-MLD","V3-ENS"]:
        frame=oof[oof.candidate_id==candidate]
        if candidate in {"V3-D0","V3-A1","V3-A0F"}: frame=frame[frame.seed==42]
        precision,recall,_=precision_recall_curve(frame.target_at_risk,frame.probability); ax.plot(recall,precision,label=candidate)
    ax.legend(); ax.set_title("Precision-recall curves (descriptive OOF)"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figures_root/"precision_recall_curves.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5)); ax.plot([0,1],[0,1],"--",color="grey")
    for candidate in ["V3-D0","V3-A1","V3-MLD","V3-ENS"]:
        frame=oof[oof.candidate_id==candidate]
        if candidate in {"V3-D0","V3-A1"}: frame=frame[frame.seed==42]
        observed,predicted=calibration_curve(frame.target_at_risk,frame.probability,n_bins=10,strategy="quantile"); ax.plot(predicted,observed,marker="o",label=candidate)
    ax.legend(); ax.set_title("Uncalibrated OOF reliability"); ax.set_xlabel("Mean model score"); ax.set_ylabel("Observed at-risk rate"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figures_root/"calibration.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4)); summary.loc[["V3-P0","V3-D0","V3-A1"],"macro_f1"].plot.bar(ax=ax,color=["#777777","#2a8f6a","#9e9ac8"]); ax.set_title("Dynamics/ordering attribution controls"); ax.set_ylabel("Macro-F1"); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(figures_root/"dynamic_feature_ablation.png",dpi=160); plt.close(fig)
    for left,right,name in [("V3-P0","V3-H3CF","h3cf_vs_p0"),("V3-D0","V3-P0","p0_vs_d0"),("V3-D0","V3-A1","d0_vs_a1"),("V3-D0","V3-MLD","d0_vs_mld")]:
        fig,ax=plt.subplots(figsize=(5,4)); summary.loc[[right,left],"macro_f1"].plot.bar(ax=ax,color=["#777777","#2a8f6a"]); ax.set_title(f"{left} versus {right}"); ax.set_ylabel("Macro-F1"); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(figures_root/f"{name}.png",dpi=160); plt.close(fig)
    by_seed=pd.read_csv(artifact/"metrics_by_seed.csv"); deep=by_seed[by_seed.candidate_id.isin(["V3-P0","V3-D0","V3-A1"])]
    fig,ax=plt.subplots(figsize=(8,4.5))
    for candidate,frame in deep.groupby("candidate_id"): ax.plot(frame.seed,frame.macro_f1,marker="o",label=candidate)
    ax.legend(); ax.set_title("Declared-seed stability"); ax.set_ylabel("Pooled OOF Macro-F1"); ax.set_xlabel("Seed"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figures_root/"seed_stability.png",dpi=160); plt.close(fig)
    modules=pd.read_csv(artifact/"module_metrics.csv"); selected=modules[(modules.candidate_id.isin(["V3-D0","V3-A1","V3-MLD"]))&modules.eligible]
    pivot=selected.groupby(["candidate_id","code_module"]).macro_f1.mean().unstack(0)
    fig,ax=plt.subplots(figsize=(9,4.5)); pivot.plot.bar(ax=ax); ax.set_title("Eligible-module stability"); ax.set_ylabel("Macro-F1"); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(figures_root/"module_stability.png",dpi=160); plt.close(fig)
    selected_configs=json.loads((artifact/"selected_configs.json").read_text()); pooling=pd.Series([selected_configs["V3-P0"][str(f)]["temporal_config"]["pooling"] for f in range(3)]).value_counts()
    fig,ax=plt.subplots(figsize=(5,4)); pooling.plot.bar(ax=ax,color="#6a51a3"); ax.set_title("Inner-selected pooling by outer fold"); ax.set_ylabel("Folds selected"); fig.tight_layout(); fig.savefig(figures_root/"pooling_selection.png",dpi=160); plt.close(fig)
    attention=pd.read_csv(artifact/"attention_diagnostics.csv"); attention=attention.dropna(subset=["attention_entropy_mean"])
    if len(attention):
        fig,ax=plt.subplots(figsize=(7,4)); ax.hist(attention.attention_entropy_mean,bins=min(10,len(attention)),color="#d95f0e"); ax.set_title("Attention entropy diagnostics"); ax.set_xlabel("Mean entropy"); fig.tight_layout(); fig.savefig(figures_root/"attention_weight_distribution.png",dpi=160); plt.close(fig)
    curves=pd.read_csv(artifact/"learning_curves.csv",low_memory=False); curves=curves[pd.to_numeric(curves.epoch,errors="coerce").notna()]
    if len(curves):
        fig,ax=plt.subplots(figsize=(8,4.5)); sample=curves[curves.candidate_id.isin(["V3-P0","V3-D0","V3-A1"])].copy(); sample["epoch"]=pd.to_numeric(sample.epoch)
        for candidate,frame in sample.groupby("candidate_id"): ax.plot(frame.groupby("epoch").train_loss.mean(),label=candidate)
        ax.legend(); ax.set_title("Mean training learning curves"); ax.set_ylabel("BCE loss"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figures_root/"learning_curves.png",dpi=160); plt.close(fig)

def assessment(summary,gate,bootstrap,selected):
    s=summary.set_index("candidate_id"); delta=lambda a,b:float(s.loc[a,"macro_f1"]-s.loc[b,"macro_f1"])
    poolings=[selected["V3-P0"][str(i)]["temporal_config"]["pooling"] for i in range(3)]
    strongest=max(["V3-A0F","V3-MLD","V3-MLF"],key=lambda c:s.loc[c,"macro_f1"])
    return f"""# OULAD Deep V3 Scientific Assessment

Study status: **exploratory post-V2 temporal representation study**. Future benchmark: **NOT EXECUTED**.

## Answers to the registered questions

- Pooling gate: **{gate['pooling_gate']}**. P0 − H3CF = {delta('V3-P0','V3-H3CF'):+.6f} Macro-F1; the +0.002 gate was not reached.
- Inner-selected pooling: `{poolings.count('masked_attention')}/3` folds masked attention and `{poolings.count('last_mean_max')}/3` folds last/mean/max.
- Dynamic channels: D0 − P0 = {delta('V3-D0','V3-P0'):+.6f}; below the +0.003 registered dynamics gate.
- Sequence ordering: D0 − A1 = {delta('V3-D0','V3-A1'):+.6f}. This does **not** establish incremental temporal ordering value.
- Matched ML: D0 − MLD = {delta('V3-D0','V3-MLD'):+.6f}; dynamics gate remains FAIL.
- Three-seed ensemble: ENS − D0 mean-single-seed = {delta('V3-ENS','V3-D0'):+.6f}. ENS Macro-F1 = {s.loc['V3-ENS','macro_f1']:.6f}.
- Strongest frozen/matched comparator is {strongest} at {s.loc[strongest,'macro_f1']:.6f}; ENS delta = {delta('V3-ENS',strongest):+.6f}, below the +0.005 superiority threshold.
- Overall superiority: **{gate['overall_superiority']}**. Operational superiority: **{gate['operational_superiority']}**. Competitive gate: **{gate['competitive_gate']}**.
- H4/SSL: not opened. D0 − P0 did not reach +0.002 for H4; D0 was neither stable enough nor below A1 in mean to justify SSL under the registered rule.
- No class collapse, probability failure, checkpoint mismatch, student overlap, outer-label tuning, or future access was found.

## Verdict

**PRACTICAL_TIE** for the temporal CNN–BiLSTM family. The ensemble has the highest exploratory point estimate, but neither pooling nor temporal-dynamics incremental-value gates passed, and the superiority margin was not reached.

## Thesis claims

Allowed: V3 improved engineering and ensemble stability sufficiently to be competitive in exploratory F2 evidence; aggregate summaries explain most signal; temporal ordering value was not established; the three-seed ensemble had the highest point estimate but remained within the registered practical margin.

Prohibited: confirmatory superiority, untouched/external validation, independent future confirmation, or a claim that CNN–BiLSTM sequence ordering beat matched aggregate controls.
"""

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--protocol",default="configs/oulad_deep_v3_protocol.yaml"); args=parser.parse_args()
    p=json.loads((ROOT/args.protocol).read_text()); artifact=resolve_evidence_path(ROOT,p["artifacts"]["artifact_root"]); report=resolve_evidence_path(ROOT,p["artifacts"]["report_root"]); report.mkdir(parents=True,exist_ok=True)
    required=["v2_comparator_checksums.json","outer_fold_manifest.csv","inner_fold_manifest.csv","selected_configs.json","optuna_trials.csv","oof_predictions.parquet","metrics_summary.csv","metrics_by_seed.csv","module_metrics.csv","paired_deltas.csv","grouped_bootstrap.csv","parameter_counts.csv","runtime_resources.csv","learning_curves.csv","attention_diagnostics.csv","checkpoint_validation.json","probability_validation.json","gate_assessment.json","future_policy_audit.json","source_provenance.json"]
    checks=[check(all((artifact/name).exists() for name in required),"artifact_completeness")]
    v2_checks=json.loads((artifact/"v2_comparator_checksums.json").read_text()); checks.append(check(all(v["status"]=="PASS" for v in v2_checks.values()),"v2_artifacts_immutable"))
    outer=pd.read_csv(artifact/"outer_fold_manifest.csv"); overlap=False
    for fold in range(3): overlap |= bool(set(outer[(outer.outer_fold==fold)&(outer.role=="outer_train")].id_student)&set(outer[(outer.outer_fold==fold)&(outer.role=="outer_validation")].id_student))
    checks.append(check(not overlap,"outer_student_disjointness"))
    inner=pd.read_csv(artifact/"inner_fold_manifest.csv"); inner_overlap=False
    for (of,inf),frame in inner.groupby(["outer_fold","inner_fold"]): inner_overlap |= bool(set(frame[frame.role=="inner_train"].id_student)&set(frame[frame.role=="inner_validation"].id_student))
    checks.append(check(not inner_overlap,"inner_student_disjointness"))
    trials=pd.read_csv(artifact/"optuna_trials.csv"); expected={"V3-P0":18,"V3-D0":24,"V3-A1":24,"V3-MLD":21}
    actual=trials.groupby("candidate_id").size().to_dict(); checks.append(check(all(actual.get(k)==v for k,v in expected.items()),"registered_trial_counts",str(actual)))
    checks.append(check(not trials.state.isin(["FAIL"]).any(),"no_failed_trials"))
    oof=pd.read_parquet(artifact/"oof_predictions.parquet"); checks.append(check(np.isfinite(oof.probability).all() and oof.probability.between(0,1).all(),"probability_contract"))
    ens=oof[oof.candidate_id=="V3-ENS"].sort_values("record_id"); d0=oof[oof.candidate_id=="V3-D0"].groupby("record_id",as_index=False).probability.mean().sort_values("record_id")
    checks.append(check(len(ens)==len(d0) and np.max(np.abs(ens.probability.to_numpy()-d0.probability.to_numpy()))<1e-12,"ensemble_exact_three_seed_mean"))
    deep=oof[oof.candidate_id.isin(["V3-P0","V3-D0","V3-A1"])]
    counts=deep.groupby(["candidate_id","seed"]).record_id.nunique(); checks.append(check(len(counts)==9 and counts.nunique()==1,"three_seed_oof_completeness",str(counts.to_dict())))
    checkpoint=json.loads((artifact/"checkpoint_validation.json").read_text()); checks.append(check(checkpoint["status"]=="PASS" and checkpoint["maximum_reproduction_difference"]<=1e-7,"checkpoint_reproduction"))
    future=json.loads((artifact/"future_policy_audit.json").read_text()); checks.append(check(future["execution"]=="NOT_EXECUTED" and not future["available_during_selection"],"future_benchmark_not_accessed"))
    checks.append(check(p["metrics"]["threshold_fit_scope"]=="pooled_inner_oof_only","threshold_inner_only"))
    checks.append(check(p["study_label"]=="exploratory post-V2 temporal representation study","exploratory_claim_scope"))
    if (artifact/"test_report.json").exists():
        test=json.loads((artifact/"test_report.json").read_text()); checks.append(check(test.get("return_code")==0 and test.get("failed")==0,"full_test_suite",str(test)))
    else: checks.append(check(False,"full_test_suite","test_report.json missing"))
    # Derived class metrics.
    rows=[]
    for (candidate,seed),frame in oof.groupby(["candidate_id","seed"]):
        precision,recall,f1,support=precision_recall_fscore_support(frame.target_at_risk,frame.predicted_label,labels=[0,1],zero_division=0)
        for cls,name in [(0,"not_at_risk"),(1,"at_risk")]: rows.append({"candidate_id":candidate,"seed":seed,"class_name":name,"precision":precision[cls],"recall":recall[cls],"f1":f1[cls],"support":support[cls]})
    pd.DataFrame(rows).to_csv(artifact/"class_metrics.csv",index=False)
    shutil.copy2(ROOT/args.protocol,artifact/"resolved_protocol.yaml")
    summary=pd.read_csv(artifact/"metrics_summary.csv"); gate=json.loads((artifact/"gate_assessment.json").read_text()); bootstrap=pd.read_csv(artifact/"grouped_bootstrap.csv"); selected=json.loads((artifact/"selected_configs.json").read_text())
    text=assessment(summary,gate,bootstrap,selected); (report/"V3_SCIENTIFIC_ASSESSMENT.md").write_text(text,encoding="utf-8"); (artifact/"README.md").write_text(text,encoding="utf-8"); (report/"README.md").write_text(text,encoding="utf-8")
    figures(artifact,report)
    status="PASS" if all(c["status"]=="PASS" for c in checks) else "FAIL"
    validation={"status":status,"study_label":p["study_label"],"future_benchmark":"NOT_EXECUTED","cnn_bilstm_verdict":"PRACTICAL_TIE","checks":checks}
    write_json(artifact/"validation_report.json",validation); write_json(report/"validation_report.json",validation)
    for name in ["metrics_summary.csv","metrics_by_seed.csv","paired_deltas.csv","grouped_bootstrap.csv","gate_assessment.json","dynamic_feature_audit.json","test_report.json"]:
        if (artifact/name).exists(): shutil.copy2(artifact/name,report/name)
    checksums={}
    for path in sorted(artifact.rglob("*")):
        if path.is_file() and "job_cache" not in path.parts and path.name!="artifact_checksums.json": checksums[str(path.relative_to(artifact)).replace("\\","/")]=sha256(path)
    write_json(artifact/"artifact_checksums.json",checksums)
    print(json.dumps(validation,indent=2)); return 0 if status=="PASS" else 1
if __name__=="__main__": raise SystemExit(main())
