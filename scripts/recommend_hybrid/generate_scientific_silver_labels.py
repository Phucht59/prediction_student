"""Replay canonical train-only Snorkel fitting and generate soft silver labels."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]; ARTIFACT = ROOT / "artifacts/recommend_hybrid/scientific_labeling"
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.recommend_hybrid.weak_supervision.label_model import fit_dataset
from src.recommend_hybrid.weak_supervision.metrics import quality_metrics
from src.recommend_hybrid.weak_supervision.silver_labels import apply_silver_policy
if __name__ == "__main__":
    config=yaml.safe_load((ROOT / "configs/recommend_hybrid/labeling_functions.yaml").read_text(encoding="utf-8")); frame=pd.read_parquet(ARTIFACT / "candidates.parquet"); all_rows=[]; metrics={}
    policy=config["locked_policy"]
    for dataset in ("student_mat","student_por","oulad"):
        data=frame[frame.dataset==dataset].copy(); train=data[data.split=="train"]
        model=fit_dataset(train,seed=config["canonical_seed"],epochs=config["label_model"]["n_epochs"])
        output=apply_silver_policy(data,model,confidence_threshold=policy["confidence_threshold"],minimum_families=policy["minimum_independent_families"])
        all_rows.append(output); metrics[dataset]=quality_metrics(output)
    silver=pd.concat(all_rows,ignore_index=True); path=ARTIFACT / "silver_labels.parquet"; silver.to_parquet(path,index=False)
    sample=silver.drop(columns=["evidence_values"],errors="ignore").head(30).to_dict("records")
    (ARTIFACT / "samples").mkdir(exist_ok=True); (ARTIFACT / "samples/redacted_silver_label_sample.jsonl").write_text("".join(json.dumps(row,default=str,sort_keys=True)+"\n" for row in sample),encoding="utf-8")
    manifest={"schema_version":"phase2_silver_labels_v1","rows":len(silver),"dataset_metrics":metrics,"policy":policy,"fit_scope":"train_only","test_used_for_fit_or_tuning":False,"silver_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    (ARTIFACT / "quality_metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n",encoding="utf-8"); (ARTIFACT / "silver_label_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"SCIENTIFIC_SILVER_LABELS_GENERATED rows={len(silver)}")
