import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import pandas as pd
import torch
import json
from sklearn.model_selection import train_test_split
from src.improved_hybrid.config import DATASETS, HybridModelConfig, TrainingConfig, ensure_dirs, RESULTS_DIR
from src.improved_hybrid.data import read_raw_data, add_targets, HybridDataProcessor, apply_resampling, create_dataloaders
from src.improved_hybrid.features import apply_feature_engineering
from src.improved_hybrid.models import HybridCNNBiLSTMAttentionOrdinal
from src.improved_hybrid.train import compute_class_weights, train_hybrid_model, evaluate_model

def run_experiment(name, ds_name, df, processor, spec, train_config, model_config, resample_strategy, seed, use_attention=True, use_context=True, ordinal_lambda=0.1, use_focal=True, use_class_weights=True):
    print(f"\n--- Running Experiment: {name} ({ds_name}) ---")
    
    train_val_df, test_df = train_test_split(df, test_size=0.20, stratify=df['target_class'], random_state=seed)
    train_df, val_df = train_test_split(train_val_df, test_size=0.20, stratify=train_val_df['target_class'], random_state=seed)
    
    X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr = processor.transform(train_df)
    X_seq_va, X_num_va, X_cat_va, y_cls_va, y_ord_va = processor.transform(val_df)
    
    if resample_strategy != "none":
        X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr = apply_resampling(
            X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr, strategy=resample_strategy, seed=seed
        )
        
    train_loader = create_dataloaders(X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr, train_config.batch_size, shuffle=True)
    val_loader = create_dataloaders(X_seq_va, X_num_va, X_cat_va, y_cls_va, y_ord_va, train_config.batch_size, shuffle=False)
    
    model = HybridCNNBiLSTMAttentionOrdinal(
        seq_input_dim=processor.seq_cols.__len__(),
        num_numeric_context=processor.num_cols.__len__(),
        categorical_cardinalities=processor.cat_cardinalities,
        num_classes=spec.n_classes,
        use_attention=use_attention,
        use_context=use_context,
        **model_config.__dict__
    )
    
    # Custom config for train
    custom_train_config = TrainingConfig(
        max_epochs=train_config.max_epochs,
        ordinal_lambda=ordinal_lambda,
        focal_gamma=2.0 if use_focal else 0.0,
        batch_size=train_config.batch_size
    )
    
    class_weights = compute_class_weights(y_cls_tr, spec.n_classes) if use_class_weights else None
    
    model, _ = train_hybrid_model(model, train_loader, val_loader, custom_train_config, class_weights)
    val_metrics, _, _, _ = evaluate_model(model, val_loader)
    
    print(f"{name} Validation F1: {val_metrics['f1_macro']:.4f}")
    return val_metrics['f1_macro']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["student-mat", "student-por", "xapi", "all"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    ensure_dirs()

    results = []

    for ds_name in datasets:
        spec = DATASETS[ds_name]
        raw = read_raw_data(spec)
        df = add_targets(raw, spec)
        df, seq_cols, num_cols, cat_cols = apply_feature_engineering(df, spec)
        
        train_val_df, test_df = train_test_split(df, test_size=0.20, stratify=df['target_class'], random_state=args.seed)
        
        processor = HybridDataProcessor(seq_cols, num_cols, cat_cols)
        processor.fit(train_val_df) # Use train_val for full fitting for ablation (simplified)
        
        train_config = TrainingConfig(max_epochs=args.epochs)
        model_config = HybridModelConfig()
        
        resample_strategy = "adasyn" if spec.kind == "xapi" else "smote"
        if ds_name == "student-por":
            resample_strategy = "none"

        # E3: cnn_bilstm_attention (no context)
        e3_f1 = run_experiment("E3: CNN-BiLSTM-Attention Only", ds_name, df, processor, spec, train_config, model_config, resample_strategy, args.seed, use_context=False, use_attention=True)
        
        # E4: hybrid_sequence_context (attention + context, no ordinal)
        e4_f1 = run_experiment("E4: Hybrid Seq+Context", ds_name, df, processor, spec, train_config, model_config, resample_strategy, args.seed, use_context=True, use_attention=True, ordinal_lambda=0.0)
        
        # E5: hybrid_ordinal
        e5_f1 = run_experiment("E5: Hybrid + Ordinal", ds_name, df, processor, spec, train_config, model_config, resample_strategy, args.seed, use_context=True, use_attention=True, ordinal_lambda=0.1)
        
        # E6: hybrid_focal_classweight
        e6_f1 = run_experiment("E6: Hybrid + Focal/ClassWeight", ds_name, df, processor, spec, train_config, model_config, resample_strategy, args.seed, use_context=True, use_attention=True, ordinal_lambda=0.1, use_focal=True, use_class_weights=True)
        
        results.append({
            "Dataset": ds_name,
            "E3_CNN_BiLSTM_Attention": e3_f1,
            "E4_Hybrid_Context": e4_f1,
            "E5_Hybrid_Ordinal": e5_f1,
            "E6_Hybrid_Focal": e6_f1
        })
        
    results_df = pd.DataFrame(results)
    print("\n=== Ablation Results ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(RESULTS_DIR / "hybrid_ablation_results.csv", index=False)

if __name__ == "__main__":
    main()
