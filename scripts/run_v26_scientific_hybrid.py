import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import pandas as pd
import numpy as np
import torch
import copy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.v26_scientific_hybrid.config import (
    DATASETS, FIXED_SEEDS, V26ModelConfig, V26TrainingConfig, 
    ensure_dirs, RESULTS_DIR, FIGURES_DIR, MODEL_DIR
)
from src.v26_scientific_hybrid.data import read_raw_data, add_targets, V26DataProcessor, apply_resampling, create_dataloaders
from src.v26_scientific_hybrid.features import apply_feature_engineering
from src.v26_scientific_hybrid.models import HybridCNNBiLSTMAttentionOrdinalV2
from src.v26_scientific_hybrid.train import compute_class_weights, train_v26_model, evaluate_model
from src.v26_scientific_hybrid.ensemble import evaluate_ensemble
from src.v26_scientific_hybrid.reporting import write_markdown_report, write_json

def run_ablation(df, processor, spec, train_config, model_config, n_classes, target_mode):
    print(f"\n--- Running Ablation for {spec.name} ({target_mode}) ---")
    train_val_df, test_df = train_test_split(df, test_size=0.20, stratify=df['target_class'], random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.20, stratify=train_val_df['target_class'], random_state=42)
    
    X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr = processor.transform(train_df)
    X_seq_va, X_num_va, X_cat_va, y_cls_va, y_ord_va = processor.transform(val_df)
    
    resample_strategy = "adasyn" if spec.kind == "xapi" else "smote"
    if spec.name == "student-por":
        resample_strategy = "none"
        
    X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr = apply_resampling(X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr, strategy=resample_strategy, seed=42)
    
    train_loader = create_dataloaders(X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr, train_config.batch_size, shuffle=True)
    val_loader = create_dataloaders(X_seq_va, X_num_va, X_cat_va, y_cls_va, y_ord_va, train_config.batch_size, shuffle=False)
    
    def run_var(name, u_att, u_ctx, u_ord, u_foc):
        print(f"  > Running {name}...")
        model = HybridCNNBiLSTMAttentionOrdinalV2(
            seq_input_dim=len(processor.seq_cols), num_numeric_context=len(processor.num_cols),
            categorical_cardinalities=processor.cat_cardinalities, num_classes=n_classes,
            use_attention=u_att, use_context=u_ctx, **model_config.__dict__
        )
        cw = compute_class_weights(y_cls_tr, n_classes) if u_foc else None
        model, _ = train_v26_model(model, train_loader, val_loader, train_config, cw, use_ordinal=u_ord, use_focal=u_foc)
        mets, _, _, _ = evaluate_model(model, val_loader)
        return mets['f1_macro']

    return {
        "Dataset": spec.name,
        "E1 (Seq)": run_var("E1", False, False, False, False),
        "E2 (+Att)": run_var("E2", True, False, False, False),
        "E3 (+Ctx)": run_var("E3", True, True, False, False),
        "E4 (+Ord)": run_var("E4", True, True, True, False),
        "E5 (+Foc)": run_var("E5", True, True, True, True),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["student-mat", "student-por", "xapi", "all"])
    parser.add_argument("--target-mode", type=str, required=True, choices=["3class", "5class"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    ensure_dirs()
    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    all_ablation_res = []
    
    for ds_name in datasets:
        spec = DATASETS[ds_name]
        
        # xAPI is natively 3-class, so if 5class is asked, we skip or fallback to 3-class naturally, but the wrapper enforces 3.
        if spec.kind == 'xapi' and args.target_mode == '5class':
            print(f"Skipping 5class for {ds_name} (only 3class natively supported).")
            continue
            
        print(f"=== V26 Pipeline for {ds_name} ({args.target_mode}) ===")
        raw = read_raw_data(spec)
        df, n_classes = add_targets(raw, spec, args.target_mode)
        df, seq_cols, num_cols, cat_cols = apply_feature_engineering(df, spec.kind)
        
        processor = V26DataProcessor(seq_cols, num_cols, cat_cols)
        processor.fit(df)
        
        train_config = V26TrainingConfig(max_epochs=args.epochs, patience=args.patience)
        model_config = V26ModelConfig()
        
        # 1. Ablation on Validation
        ablation_dict = run_ablation(df, processor, spec, train_config, model_config, n_classes, args.target_mode)
        all_ablation_res.append(ablation_dict)
        
        # 2. Strict Full Training + Ensembling on 20% test
        train_val_df, test_df = train_test_split(df, test_size=0.20, stratify=df['target_class'], random_state=42)
        X_seq_te, X_num_te, X_cat_te, y_cls_te, y_ord_te = processor.transform(test_df)
        test_loader = create_dataloaders(X_seq_te, X_num_te, X_cat_te, y_cls_te, y_ord_te, train_config.batch_size, shuffle=False)
        
        trained_models = []
        resample_strategy = "adasyn" if spec.kind == "xapi" else "smote"
        if spec.name == "student-por": resample_strategy = "none"
        
        print("\n--- Training E6 (Seed Ensemble) on full Train+Val ---")
        for i, seed in enumerate(FIXED_SEEDS):
            print(f"  > Training Seed {seed} ({i+1}/{len(FIXED_SEEDS)})")
            # Create a fold or just train on entire train_val? Paper says stratified k-fold or train/val.
            # Usually for ensemble, we just use same data with different seeds, or K-fold. 
            # We will just train on full train_val_df directly to keep it simple, letting seed affect initialization and resampling.
            X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr = processor.transform(train_val_df)
            X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr = apply_resampling(X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr, strategy=resample_strategy, seed=seed)
            train_loader = create_dataloaders(X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr, train_config.batch_size, shuffle=True)
            
            model = HybridCNNBiLSTMAttentionOrdinalV2(
                seq_input_dim=len(seq_cols), num_numeric_context=len(num_cols),
                categorical_cardinalities=processor.cat_cardinalities, num_classes=n_classes,
                **model_config.__dict__
            )
            cw = compute_class_weights(y_cls_tr, n_classes)
            
            # Since we don't have val set here to early stop, we use validation split internally, or train fixed epochs.
            # To be strict but avoid overfitting, let's use 10% of train_val as internal val for early stopping.
            tv_tr, tv_va = train_test_split(train_val_df, test_size=0.1, stratify=train_val_df['target_class'], random_state=seed)
            X_seq_va, X_num_va, X_cat_va, y_cls_va, y_ord_va = processor.transform(tv_va)
            val_loader = create_dataloaders(X_seq_va, X_num_va, X_cat_va, y_cls_va, y_ord_va, train_config.batch_size, shuffle=False)
            
            model, _ = train_v26_model(model, train_loader, val_loader, train_config, cw, use_ordinal=True, use_focal=True)
            trained_models.append(model)
            torch.save(model.state_dict(), MODEL_DIR / f"v26_{ds_name}_{args.target_mode}_seed{seed}.pt")
            
        print("\n--- Evaluating Ensemble on Final Locked Test ---")
        final_mets, final_targets, final_preds, final_probs = evaluate_ensemble(trained_models, test_loader)
        
        print(f"Final E6 (Ensemble) F1: {final_mets['f1_macro']:.4f}")
        
        # Save plots
        plt.figure(figsize=(8,6))
        cm = confusion_matrix(final_targets, final_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix ({ds_name} - {args.target_mode})")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(FIGURES_DIR / f"confusion_matrix_{ds_name}_{args.target_mode}.png")
        plt.close()
        
        write_markdown_report(pd.DataFrame([ablation_dict]), final_mets, args.target_mode, RESULTS_DIR / f"v26_report_{ds_name}_{args.target_mode}.md")
        
    ablation_df = pd.DataFrame(all_ablation_res)
    ablation_df.to_csv(RESULTS_DIR / f"v26_ablation_{args.target_mode}.csv", index=False)
    print("\nAll done!")

if __name__ == "__main__":
    main()
