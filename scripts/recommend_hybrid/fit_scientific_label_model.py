"""Fit the three dataset-specific Snorkel Label Models on train only."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.recommend_hybrid.weak_supervision.label_model import fit_dataset, model_payload
ARTIFACT = ROOT / "artifacts/recommend_hybrid/scientific_labeling"
if __name__ == "__main__":
    config = yaml.safe_load((ROOT / "configs/recommend_hybrid/labeling_functions.yaml").read_text(encoding="utf-8"))
    frame = pd.read_parquet(ARTIFACT / "candidates.parquet")
    payloads = []
    for dataset in ("student_mat", "student_por", "oulad"):
        train = frame[(frame.dataset == dataset) & (frame.split == "train")]
        model = fit_dataset(train, seed=config["canonical_seed"], epochs=config["label_model"]["n_epochs"])
        payload = model_payload(model, dataset=dataset, seed=config["canonical_seed"], train_rows=len(train))
        payloads.append(payload)
        (ARTIFACT / "label_model_weights").mkdir(exist_ok=True)
        (ARTIFACT / "label_model_weights" / f"{dataset}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACT / "label_model_manifest.json").write_text(json.dumps({"schema_version":"phase2_label_models_v1","library":"snorkel.labeling.model.LabelModel","models":payloads}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SCIENTIFIC_LABEL_MODELS_FIT datasets=3 train_only=true")
