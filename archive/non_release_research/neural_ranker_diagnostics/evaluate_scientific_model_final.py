from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));ART=ROOT/"artifacts/recommend_hybrid/scientific_model"
from src.recommend_hybrid.scientific_model.features import prepare
from src.recommend_hybrid.scientific_model.model import EvidenceGatedRecommender
from src.recommend_hybrid.scientific_model.constraints import constrain
from src.recommend_hybrid.scientific_model.evaluation import evaluate
f=pd.read_parquet(ROOT/"artifacts/recommend_hybrid/scientific_labeling/silver_labels.parquet"); f=f[(f.split=="test")&(f.silver_status=="RETAINED")].copy();x,a,d,s,schema=prepare(f);probs=[]
for seed in [42,1201,2026,3407,7319]:
 m=EvidenceGatedRecommender(schema["feature_dim"],len(schema["action_ids"]),len(schema["datasets"]),len(schema["stages"]));m.load_state_dict(torch.load(ART/"checkpoints"/f"seed_{seed}.pt",map_location="cpu",weights_only=True));m.eval()
 with torch.no_grad(): probs.append(torch.softmax(m(torch.tensor(x),torch.tensor(a),torch.tensor(d),torch.tensor(s)),1).numpy())
p=np.mean(probs,0); constrained,_,_,_=constrain(p,f);metrics=evaluate(f,constrained); by={k:evaluate(f.loc[f.dataset.eq(k)],constrained[f.dataset.eq(k).to_numpy()]) for k in f.dataset.unique()}
ART.joinpath("test_predictions.parquet").parent.mkdir(exist_ok=True);f.assign(**{"prob_0":constrained[:,0],"prob_1":constrained[:,1],"prob_2":constrained[:,2]}).to_parquet(ART/"test_predictions.parquet",index=False);(ART/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n");(ART/"metrics_by_dataset.json").write_text(json.dumps(by,indent=2)+"\n");print("SCIENTIFIC_MODEL_EVALUATED")
