import sys
import json
from pathlib import Path
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.improved_hybrid.config import DATASETS, HybridModelConfig, ensure_dirs, RESULTS_DIR, MODEL_DIR
from src.improved_hybrid.data import read_raw_data, add_targets, HybridDataProcessor, create_dataloaders
from src.improved_hybrid.features import apply_feature_engineering
from src.improved_hybrid.models import HybridCNNBiLSTMAttentionOrdinal
from src.improved_hybrid.recommendations import generate_recommendations
from sklearn.model_selection import train_test_split

def main():
    ensure_dirs()
    
    all_recs = []
    
    for ds_name, spec in DATASETS.items():
        print(f"Generating recommendations for {ds_name}...")
        raw = read_raw_data(spec)
        df = add_targets(raw, spec)
        df, seq_cols, num_cols, cat_cols = apply_feature_engineering(df, spec)
        
        train_val_df, test_df = train_test_split(df, test_size=0.20, stratify=df['target_class'], random_state=42)
        
        processor = HybridDataProcessor(seq_cols, num_cols, cat_cols)
        processor.fit(train_val_df)
        
        X_seq_te, X_num_te, X_cat_te, y_cls_te, y_ord_te = processor.transform(test_df)
        
        model_path = MODEL_DIR / f"final_{ds_name}.pt"
        if not model_path.exists():
            print(f"Model {model_path} not found. Skip.")
            continue
            
        model_config = HybridModelConfig()
        model = HybridCNNBiLSTMAttentionOrdinal(
            seq_input_dim=len(seq_cols),
            num_numeric_context=len(num_cols),
            categorical_cardinalities=processor.cat_cardinalities,
            num_classes=spec.n_classes,
            **model_config.__dict__
        )
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        with torch.no_grad():
            logits, _ = model(torch.tensor(X_seq_te, dtype=torch.float32), 
                              torch.tensor(X_num_te, dtype=torch.float32), 
                              torch.tensor(X_cat_te, dtype=torch.long))
            probs = torch.softmax(logits, dim=1).numpy()
            preds = torch.argmax(logits, dim=1).numpy()
            
        ds_recs = []
        for i in range(len(test_df)):
            student_id = f"{ds_name}_S{i:03d}"
            pred_class = int(preds[i])
            conf = float(max(probs[i]))
            feat_dict = test_df.iloc[i].to_dict()
            rec = generate_recommendations(student_id, pred_class, conf, feat_dict, spec)
            ds_recs.append(rec)
            
        all_recs.extend(ds_recs)
        
        with open(RESULTS_DIR / f"final_predictions_and_recommendations_{ds_name}.json", "w", encoding="utf-8") as f:
            json.dump(ds_recs, f, indent=2, ensure_ascii=False)
            
    print("Done generating recommendations.")

if __name__ == "__main__":
    main()
