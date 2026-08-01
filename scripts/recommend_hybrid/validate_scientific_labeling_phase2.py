"""Validate Phase 2 candidate, split, Snorkel, and silver-label artefacts."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.recommend_hybrid.weak_supervision.label_model import fit_dataset, vote_matrix
from src.recommend_hybrid.weak_supervision.lf_registry import registry

ARTIFACT = ROOT / "artifacts/recommend_hybrid/scientific_labeling"; PASS = "RECOMMEND_SCIENTIFIC_LABELING_PHASE2_PASS"
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    candidates=pd.read_parquet(ARTIFACT / "candidates.parquet"); silver=pd.read_parquet(ARTIFACT / "silver_labels.parquet")
    if candidates.duplicated(["query_id","action_id"]).any(): raise ValueError("duplicate candidates")
    forbidden={"G3","target","final_result","date_unregistration","withdrawal_outcome","raw_g3"}
    if forbidden & set(candidates.columns): raise ValueError("forbidden candidate field")
    if set(candidates.dataset)!={"student_mat","student_por","oulad"}: raise ValueError("dataset coverage")
    if not {"S0","S1","S2"}.issubset(set(candidates[candidates.dataset!="oulad"].stage)): raise ValueError("UCI stages")
    if not {"EARLY_20","EARLY_35","MIDDLE_50","LATE_75"}.issubset(set(candidates[candidates.dataset=="oulad"].stage)): raise ValueError("OULAD stages")
    if (candidates[candidates.dataset=="oulad"].requested_cutoff < 20).any(): raise ValueError("pre-20 candidate")
    if (candidates.stage=="FINAL_EVALUATION").any(): raise ValueError("FINAL intervention")
    if candidates.groupby("student_key").split.nunique().gt(1).any(): raise ValueError("split overlap")
    if not ((silver[["silver_prob_0","silver_prob_1","silver_prob_2"]].sum(axis=1)-1).abs()<1e-6).all(): raise ValueError("invalid soft probabilities")
    if not set(silver.silver_status).issubset({"RETAINED","ABSTAIN"}): raise ValueError("silver status")
    manifest=json.loads((ARTIFACT/"label_model_manifest.json").read_text(encoding="utf-8"))
    if manifest["library"]!="snorkel.labeling.model.LabelModel" or len(manifest["models"])!=3: raise ValueError("Snorkel model manifest")
    stability={}; ablations={};
    for dataset in ("student_mat","student_por","oulad"):
        train=candidates[(candidates.dataset==dataset)&(candidates.split=="train")].head(5000)
        matrices=[]
        for seed in (42,1201,2026,3407,7319): matrices.append(fit_dataset(train,seed=seed,epochs=100).get_weights().tolist())
        stability[dataset]={"seeds":[42,1201,2026,3407,7319],"bootstrap_repeats":1,"weight_replay_hashes":[hashlib.sha256(json.dumps(value).encode()).hexdigest() for value in matrices]}
        matrix=vote_matrix(train); model=fit_dataset(train,seed=2026,epochs=100); base=model.predict_proba(matrix).argmax(axis=1); changes={}
        for family in {lf.lf_family for lf in registry()}:
            masked=matrix.copy()
            for index, lf in enumerate(registry()):
                if lf.lf_family==family: masked[:,index]=-1
            changes[family]=float((base!=model.predict_proba(masked).argmax(axis=1)).mean())
        ablations[dataset]=changes
    (ARTIFACT/"ablation_metrics.json").write_text(json.dumps(ablations,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (ARTIFACT/"split_manifest.json").write_text(json.dumps({"schema_version":"phase2_split_v1","counts":candidates.drop_duplicates("query_id").split.value_counts().to_dict(),"student_overlap_violations":0,"cross_stage_split_violations":0,"dataset_scoped_uci_identity_limitation":True},indent=2,sort_keys=True)+"\n",encoding="utf-8")
    checks={name:sha(ARTIFACT/name) for name in ("candidates.parquet","silver_labels.parquet","label_model_manifest.json","quality_metrics.json")}
    (ARTIFACT/"checksums.sha256").write_text("".join(f"{value}  {key}\n" for key,value in sorted(checks.items())),encoding="utf-8")
    gate={"gate":PASS,"candidate_rows":len(candidates),"silver_rows":len(silver),"snorkel":"0.9.9","split_overlap_violations":0,"checksums":checks,"stability":stability,"largest_family_ablation_change":max(max(value.values()) for value in ablations.values())}
    (ROOT/"reports/recommend_hybrid/scientific_labeling/PHASE2_GATE.json").write_text(json.dumps(gate,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(PASS)
if __name__=="__main__": main()
