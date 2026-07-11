"""Post-hoc imbalance ablation on the frozen development pool only.

This deliberately does not read, score, or write the locked-test split.
"""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATASETS, ROOT_DIR
from src.data_pipeline import process_target_and_stratify
from src.model_selection import fit_fold_predict_proba, make_folds, metric_summary_from_predictions
from src.postgres_data_source import load_dataset_version_from_postgres, reconstruct_splits_from_run

OUT = ROOT_DIR / "artifacts/supplementary/imbalance_adasyn_analysis"
RUN_ID = "5a0b5041-5216-4a48-9e46-b0c16ab14866"
CONFIG = ROOT_DIR / "artifacts/final/final-5a0b5041-5216-4a48-9e46-b0c16ab14866/selected_config.json"

def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    spec = DATASETS["student-mat"]
    raw, _ = load_dataset_version_from_postgres("student-mat", 1)
    frame = process_target_and_stratify(raw.copy(), spec.target_col, spec.kind, "3class").dropna(subset=["_strat_target"]).drop(columns=["_strat_target"])
    train, _ = reconstruct_splits_from_run(frame, RUN_ID)
    base = json.loads(CONFIG.read_text(encoding="utf-8"))["best_params"]
    strategies = {"none": ("none", "none"), "class_weight": ("none", "balanced"), "smote": ("smote", "none"), "adasyn": ("adasyn", "none"), "smote_plus_class_weight": ("smote", "balanced")}
    rows=[]; summaries={}
    for name, (sampling, weight) in strategies.items():
        params={**base, "oversample_method": sampling, "class_weight_mode": weight}
        folds=make_folds(train, spec.target_col, n_splits=5, seed=42)
        p=np.zeros((len(train),3)); pred=np.zeros(len(train), dtype=int); ids=np.zeros(len(train), dtype=int)
        for fold,(tr,va) in enumerate(folds):
            result=fit_fold_predict_proba(train_fold=train.iloc[tr].copy(), validation_fold=train.iloc[va].copy(), spec=spec, params=params, seed=42, fold_index=fold, ablation_mode="sequence_only")
            p[va]=result.probabilities; pred[va]=result.predictions; ids[va]=fold
            rows.append({"strategy":name,"outer_fold":fold,"n_train":len(tr),"n_scoring":len(va),"macro_f1":metric_summary_from_predictions(train.iloc[va][spec.target_col].to_numpy(int),result.probabilities,result.predictions,np.zeros(len(va),dtype=int))["f1_macro"]})
        summaries[name]=metric_summary_from_predictions(train[spec.target_col].to_numpy(int),p,pred,ids)
    with (OUT/"fold_metrics.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    protocol={"label":"supplementary post-hoc analysis","run_id":RUN_ID,"data":"frozen train pool only","outer_folds":5,"seed":42,"architecture":"frozen CNN-BiLSTM","hyperparameters":"frozen selected config except imbalance strategy","strategies":list(strategies),"locked_test_used":False,"resampling_scope":"fold-training partition only; early-stop and scoring folds are not resampled","model_selection":False}
    (OUT/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    (OUT/"summary.json").write_text(json.dumps(summaries,indent=2),encoding="utf-8")
    (OUT/"README.md").write_text("# Supplementary imbalance ablation\n\nĐây là thí nghiệm hậu nghiệm nhằm đối chiếu yêu cầu đề cương, không tham gia lựa chọn mô hình final. It uses only the frozen development pool; no locked-test records are read or scored. ADASYN is valid here because the model input is numeric G1/G2 only.\n",encoding="utf-8")
    checks={p.name:checksum(p) for p in OUT.iterdir() if p.is_file()}
    (OUT/"artifact_checksums.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
if __name__=="__main__": main()
