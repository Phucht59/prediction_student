import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from src.improved_hybrid.config import DATASETS, HybridModelConfig, TrainingConfig, ensure_dirs, RESULTS_DIR, MODEL_DIR, FIGURES_DIR
from src.improved_hybrid.data import read_raw_data, add_targets, HybridDataProcessor, apply_resampling, create_dataloaders
from src.improved_hybrid.features import apply_feature_engineering
from src.improved_hybrid.models import HybridCNNBiLSTMAttentionOrdinal
from src.improved_hybrid.train import compute_class_weights, train_hybrid_model, evaluate_model
from src.improved_hybrid.evaluate import save_confusion_matrix, save_training_curve, write_json
from src.improved_hybrid.recommendations import generate_recommendations
from src.improved_hybrid.reporting import create_markdown_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["student-mat", "student-por", "xapi", "all"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    ensure_dirs()

    for ds_name in datasets:
        print(f"=== Running Hybrid Pipeline for {ds_name} ===")
        spec = DATASETS[ds_name]
        
        # 1. Load Data
        raw = read_raw_data(spec)
        df = add_targets(raw, spec)
        
        # 2. Engineer Features
        df, seq_cols, num_cols, cat_cols = apply_feature_engineering(df, spec)
        print(f"Features: Seq={len(seq_cols)}, Num={len(num_cols)}, Cat={len(cat_cols)}")
        
        # 3. Train/Val/Test Split (64/16/20) for a single run
        # First isolate 20% test
        train_val_df, test_df = train_test_split(df, test_size=0.20, stratify=df['target_class'], random_state=args.seed)
        # Then split train/val
        train_df, val_df = train_test_split(train_val_df, test_size=0.20, stratify=train_val_df['target_class'], random_state=args.seed)
        
        # 4. Process & Scale Data (Fit ONLY on Train)
        processor = HybridDataProcessor(seq_cols, num_cols, cat_cols)
        processor.fit(train_df)
        
        X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr = processor.transform(train_df)
        X_seq_va, X_num_va, X_cat_va, y_cls_va, y_ord_va = processor.transform(val_df)
        X_seq_te, X_num_te, X_cat_te, y_cls_te, y_ord_te = processor.transform(test_df)
        
        # 5. Apply Resampling ONLY on train data
        resample_strategy = "adasyn" if spec.kind == "xapi" else "smote" # default paper strategy
        if ds_name == "student-por":
            resample_strategy = "none"
            
        print(f"Original Train shape: {X_seq_tr.shape[0]}")
        X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr = apply_resampling(
            X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr, strategy=resample_strategy, seed=args.seed
        )
        print(f"Resampled Train shape: {X_seq_tr.shape[0]}")
        
        # 6. Dataloaders
        train_config = TrainingConfig(max_epochs=args.epochs)
        train_loader = create_dataloaders(X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr, train_config.batch_size, shuffle=True)
        val_loader = create_dataloaders(X_seq_va, X_num_va, X_cat_va, y_cls_va, y_ord_va, train_config.batch_size, shuffle=False)
        test_loader = create_dataloaders(X_seq_te, X_num_te, X_cat_te, y_cls_te, y_ord_te, train_config.batch_size, shuffle=False)
        
        # 7. Model
        model_config = HybridModelConfig()
        model = HybridCNNBiLSTMAttentionOrdinal(
            seq_input_dim=len(seq_cols),
            num_numeric_context=len(num_cols),
            categorical_cardinalities=processor.cat_cardinalities,
            num_classes=spec.n_classes,
            **model_config.__dict__
        )
        
        # 8. Train
        class_weights = compute_class_weights(y_cls_tr, spec.n_classes)
        print("Training model...")
        model, history = train_hybrid_model(model, train_loader, val_loader, train_config, class_weights)
        
        # 9. Evaluate
        val_metrics, _, _, _ = evaluate_model(model, val_loader)
        test_metrics, test_targets, test_preds, test_probs = evaluate_model(model, test_loader)
        
        print(f"Validation F1: {val_metrics['f1_macro']:.4f}")
        print(f"Test F1: {test_metrics['f1_macro']:.4f}")
        
        # 10. Save Outputs
        save_training_curve(history, f"{spec.display_name} Training", FIGURES_DIR / f"training_curve_{ds_name}.png")
        save_confusion_matrix(test_targets, test_preds, spec.class_names, f"{spec.display_name} Confusion Matrix", FIGURES_DIR / f"confusion_matrix_{ds_name}.png")
        torch.save(model.state_dict(), MODEL_DIR / f"final_{ds_name}.pt")
        
        # Save results
        report_data = {
            "val_metrics": val_metrics,
            "final_test": test_metrics
        }
        write_json(RESULTS_DIR / f"hybrid_single_run_{ds_name}.json", report_data)
        create_markdown_report(report_data, spec, RESULTS_DIR / f"hybrid_report_{ds_name}.md")

        # 11. Recommendations (Demo for first 5 test students)
        print("\n--- Recommendations Sample ---")
        for i in range(min(5, len(test_df))):
            student_id = f"S{i:03d}"
            pred_class = int(test_preds[i])
            conf = max(test_probs[i])
            feat_dict = test_df.iloc[i].to_dict()
            recs = generate_recommendations(student_id, pred_class, conf, feat_dict, spec)
            print(json.dumps(recs, indent=2, ensure_ascii=True))

if __name__ == "__main__":
    main()
