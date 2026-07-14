"""Run the frozen six-cell CNN-BiLSTM Neural Sanity Ablation V2.2."""
from __future__ import annotations
import argparse, json, subprocess, sys, uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT_DIR
from src.data_pipeline import SOURCE_ROW_NUMBER_COLUMN, process_target_and_stratify
from src.evaluation.metrics import classification_metrics
from src.evaluation.neural_sanity_v2_2 import EXPERIMENTS, SEEDS, build_expected_job_contract, checksum, variant_config
from src.evaluation.protocol import (DEFAULT_FOLD_MANIFEST_PATH, assert_no_legacy_records, file_checksum,
    load_fold_manifest, outer_folds_from_manifest, source_record_identity, validate_probability_matrix, validate_scenario_features)
from src.model_selection import fit_fold_predict_proba
from src.postgres_data_source import load_dataset_version_from_postgres

ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "neural_sanity_v2_2"
REPORT_ROOT = ROOT_DIR / "reports" / "neural_sanity_v2_2"
SOURCE_RUN = ROOT_DIR / "artifacts" / "benchmark_v2" / "benchmark-v2-full-20260713c"

def git(*args): return subprocess.check_output(["git", *args], cwd=ROOT_DIR, text=True).strip()
def git_clean(): return not git("status", "--porcelain", "--untracked-files=no")
def write_json(path, payload): path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-id"); p.add_argument("--smoke",action="store_true"); p.add_argument("--fold-manifest",type=Path,default=DEFAULT_FOLD_MANIFEST_PATH); a=p.parse_args()
    if not git_clean(): raise RuntimeError("Source tree must be clean before creating a V2.2 run.")
    manifest=load_fold_manifest(a.fold_manifest); validate_scenario_features(["G1","G2"],"late_stage")
    run_id=a.run_id or f"neural-sanity-v2-2-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    root=ARTIFACT_ROOT/run_id
    if root.exists(): raise FileExistsError(root)
    for directory in (root,root/"checkpoints",root/"logs"): directory.mkdir(parents=True,exist_ok=True)
    source_configs_all=json.loads((SOURCE_RUN/"configs"/"selected_configs.json").read_text(encoding="utf-8"))
    source_configs={fold:source_configs_all[f"late_stage/cnn_bilstm_v2_tuned/fold{fold}"]["config"] for fold in range(5)}
    validation_counts={fold:sum(1 for row in manifest["assignments"] if int(row["outer_fold"])==fold and row["outer_role"]=="validation") for fold in range(5)}
    if a.smoke: validation_counts={0:validation_counts[0]}; source_configs={0:source_configs[0]}
    contract=build_expected_job_contract(run_id,source_configs,validation_counts,manifest["manifest_checksum"])
    experiment_contract={"contract_version":"neural_sanity_v2_2","experiments":EXPERIMENTS,"seeds":list(SEEDS),"scenario":"late_stage","feature_set":["G1","G2"],"source_selected_config_path":str(SOURCE_RUN/"configs"/"selected_configs.json"),"source_selected_config_checksum":file_checksum(SOURCE_RUN/"configs"/"selected_configs.json"),"source_run_id":"benchmark-v2-full-20260713c","source_run_commit":"b4339c35a0197baf81ba9be871f70ff8d3030c81"}
    write_json(root/"expected_job_contract.json",contract); write_json(root/"experiment_contract.json",experiment_contract); write_json(root/"selected_source_configs.json",source_configs)
    started=datetime.now(timezone.utc).isoformat()
    run_manifest={"run_id":run_id,"status":"running","created_at":started,"source_commit":git("rev-parse","HEAD"),"source_tree_clean":True,"dataset_checksum":"e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80","dataset_version_id":1,"fold_manifest_checksum":manifest["manifest_checksum"],"source_benchmark_run":"benchmark-v2-full-20260713c","source_benchmark_commit":"b4339c35a0197baf81ba9be871f70ff8d3030c81","expected_jobs":len(contract["jobs"]),"expected_predictions":sum(x["expected_record_count"] for x in contract["jobs"]),"smoke":bool(a.smoke),"legacy_79_blocked":True,"command":" ".join(sys.argv)}
    write_json(root/"run_manifest.json",run_manifest)

    raw,meta=load_dataset_version_from_postgres("student-mat",1)
    frame=process_target_and_stratify(raw.copy(),"G3","student","3class").drop(columns=["_strat_target"])
    wanted={r["source_row_number"] for r in manifest["development_records"]}; frame=frame[frame[SOURCE_ROW_NUMBER_COLUMN].isin(wanted)].sort_values(SOURCE_ROW_NUMBER_COLUMN).reset_index(drop=True)
    assert_no_legacy_records([source_record_identity(1,v) for v in frame[SOURCE_ROW_NUMBER_COLUMN]])
    folds=outer_folds_from_manifest(frame,manifest); folds=folds[:1] if a.smoke else folds
    spec=type("Spec",(),{"target_col":"G3","kind":"student"})(); cnn_frame=frame[[SOURCE_ROW_NUMBER_COLUMN,"G1","G2","G3"]].copy()
    rows=[]; metric_rows=[]; training=[]; shapes=[]; completed=[]
    for experiment_id in EXPERIMENTS:
        for fold_index,(train_index,validation_index) in enumerate(folds):
            config=variant_config(source_configs[fold_index],experiment_id)
            for seed in SEEDS:
                result=fit_fold_predict_proba(train_fold=cnn_frame.iloc[train_index].copy(),validation_fold=cnn_frame.iloc[validation_index].copy(),spec=spec,params=config,seed=seed,fold_index=fold_index,drop_last_train=config["drop_last_train"])
                job={"experiment_id":experiment_id,"scenario":"late_stage","model_name":"cnn_bilstm","outer_fold":fold_index,"training_seed":seed}
                torch.save(result.refit_state_dict,root/"checkpoints"/f"{experiment_id}_fold{fold_index}_seed{seed}.pt")
                probability=result.probabilities; validate_probability_matrix(probability,result.predictions)
                diagnostics={**job,**result.training_diagnostics,"config_checksum":checksum(config),"feature_contract_checksum":contract["feature_contract"]["semantic_checksum"],"fold_manifest_checksum":manifest["manifest_checksum"]}
                training.append(diagnostics); shapes.append({**job,**result.shape_diagnostics,"config_checksum":checksum(config)})
                metric_rows.append({**job,**classification_metrics(result.true_labels,result.predictions,probability)})
                for idx,(_,source_row) in enumerate(frame.iloc[validation_index].iterrows()):
                    rows.append({"run_id":run_id,**job,"feature_set_id":"G1+G2","dataset_version_id":1,"record_id":source_record_identity(1,source_row[SOURCE_ROW_NUMBER_COLUMN]),"true_label":int(result.true_labels[idx]),"predicted_label":int(result.predictions[idx]),"probability_low":float(probability[idx,0]),"probability_medium":float(probability[idx,1]),"probability_high":float(probability[idx,2]),"config_checksum":checksum(config),"feature_contract_checksum":contract["feature_contract"]["semantic_checksum"],"fold_manifest_checksum":manifest["manifest_checksum"],"source_commit":run_manifest["source_commit"]})
                completed.append({**job,"checkpoint":str((root/"checkpoints"/f"{experiment_id}_fold{fold_index}_seed{seed}.pt").relative_to(root)),"status":"completed"})
                (root/"completed_jobs.jsonl").open("a",encoding="utf-8").write(json.dumps(completed[-1])+"\n")
    pred=pd.DataFrame(rows); pred.to_csv(root/"outer_validation_predictions.csv",index=False); pd.DataFrame(metric_rows).to_csv(root/"fold_seed_metrics.csv",index=False); pd.DataFrame(training).to_csv(root/"training_diagnostics.csv",index=False); pd.DataFrame(shapes).to_csv(root/"shape_diagnostics.csv",index=False)
    if not git_clean() or git("rev-parse","HEAD")!=run_manifest["source_commit"]: raise RuntimeError("Source revision changed during V2.2 run.")
    run_manifest.update({"status":"completed","completed_at":datetime.now(timezone.utc).isoformat(),"dataset_checksum_observed":meta["content_hash"]}); write_json(root/"run_manifest.json",run_manifest)
    checks={str(x.relative_to(root)):file_checksum(x) for x in root.rglob("*") if x.is_file()}; write_json(root/"checksums.json",checks); print(root)
if __name__=="__main__": main()
