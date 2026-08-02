from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.recommend_hybrid.scientific_model.features import prepare
from src.recommend_hybrid.scientific_model.model import EvidenceGatedRecommender
from src.recommend_hybrid.scientific_model.trainer import train
ART=ROOT/"artifacts/recommend_hybrid/scientific_model"; ART.mkdir(parents=True,exist_ok=True)
f=pd.read_parquet(ROOT/"artifacts/recommend_hybrid/scientific_labeling/silver_labels.parquet"); f=f[f.silver_status.eq("RETAINED")].copy(); trainf=f[f.split.eq("train")].groupby("dataset",group_keys=False).apply(lambda g:g.sample(n=min(len(g),10000),random_state=2026)).reset_index(drop=True); x,a,d,s,schema=prepare(trainf); y=trainf[["silver_prob_0","silver_prob_1","silver_prob_2"]].to_numpy("float32"); w=(trainf.silver_confidence*(1-trainf.lf_conflict*.4)).to_numpy("float32"); seeds=[42,1201,2026,3407,7319]; ck=[]
for seed in seeds:
 m=train(EvidenceGatedRecommender(schema["feature_dim"],len(schema["action_ids"]),len(schema["datasets"]),len(schema["stages"])),x,a,d,s,y,w,seed); p=ART/"checkpoints"/f"seed_{seed}.pt";p.parent.mkdir(exist_ok=True);torch.save(m.state_dict(),p);ck.append(str(p.relative_to(ROOT)))
(ART/"feature_schema.json").write_text(json.dumps(schema,indent=2)+"\n");(ART/"architecture_manifest.json").write_text(json.dumps({"architecture":"HYBRID_CNN_BILSTM_EVIDENCE_GATED_RECOMMENDER","prediction_backbone":"Frozen Hybrid CNN-BiLSTM context","parameters":sum(p.numel() for p in m.parameters()),"seeds":seeds,"checkpoints":ck},indent=2)+"\n");(ART/"selected_config.yaml").write_text("hidden_dim: 128\nepochs: 8\nsearch_used_test: false\n")
print("SCIENTIFIC_MODEL_TRAINED seeds=5")
