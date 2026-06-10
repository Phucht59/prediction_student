import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.improved_hybrid.config import DATASETS, HybridModelConfig, TrainingConfig, ensure_dirs, RESULTS_DIR
from src.improved_hybrid.data import read_raw_data, add_targets, HybridDataProcessor, apply_resampling, create_dataloaders
from src.improved_hybrid.features import apply_feature_engineering
from src.improved_hybrid.models import HybridCNNBiLSTMAttentionOrdinal
from src.improved_hybrid.train import compute_class_weights, train_hybrid_model, evaluate_model
from src.improved_hybrid.evaluate import write_json
from src.improved_hybrid.reporting import create_markdown_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["student-mat", "student-por", "xapi", "all"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    ensure_dirs()

    for ds_name in datasets:
        print(f"=== Running Repeated CV ({args.repeats}x{args.folds}) for {ds_name} ===")
        spec = DATASETS[ds_name]
        
        raw = read_raw_data(spec)
        df = add_targets(raw, spec)
        df, seq_cols, num_cols, cat_cols = apply_feature_engineering(df, spec)
        
        # 20% holdout test
        train_val_df, test_df = train_test_split(df, test_size=0.20, stratify=df['target_class'], random_state=42)
        
        cv_f1_scores = []
        cv_accuracy = []
        cv_rmse = []
        
        # Repeated CV on the 80% train_val_df
        for repeat in range(args.repeats):
            seed = 42 + repeat
            skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_df, train_val_df['target_class'])):
                print(f"Repeat {repeat+1}/{args.repeats} - Fold {fold+1}/{args.folds}")
                
                train_df = train_val_df.iloc[train_idx]
                val_df = train_val_df.iloc[val_idx]
                
                processor = HybridDataProcessor(seq_cols, num_cols, cat_cols)
                processor.fit(train_df)
                
                X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr = processor.transform(train_df)
                X_seq_va, X_num_va, X_cat_va, y_cls_va, y_ord_va = processor.transform(val_df)
                
                resample_strategy = "adasyn" if spec.kind == "xapi" else "smote"
                if ds_name == "student-por":
                    resample_strategy = "none"
                    
                X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr = apply_resampling(
                    X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr, strategy=resample_strategy, seed=seed
                )
                
                train_config = TrainingConfig(max_epochs=args.epochs)
                train_loader = create_dataloaders(X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr, train_config.batch_size, shuffle=True)
                val_loader = create_dataloaders(X_seq_va, X_num_va, X_cat_va, y_cls_va, y_ord_va, train_config.batch_size, shuffle=False)
                
                model_config = HybridModelConfig()
                model = HybridCNNBiLSTMAttentionOrdinal(
                    seq_input_dim=len(seq_cols),
                    num_numeric_context=len(num_cols),
                    categorical_cardinalities=processor.cat_cardinalities,
                    num_classes=spec.n_classes,
                    **model_config.__dict__
                )
                
                class_weights = compute_class_weights(y_cls_tr, spec.n_classes)
                model, _ = train_hybrid_model(model, train_loader, val_loader, train_config, class_weights)
                
                val_metrics, _, _, _ = evaluate_model(model, val_loader)
                
                cv_f1_scores.append(val_metrics['f1_macro'])
                cv_accuracy.append(val_metrics['accuracy'])
                cv_rmse.append(val_metrics.get('rmse', 0.0))
                
                print(f"  -> Fold F1: {val_metrics['f1_macro']:.4f}")
                
        print(f"\n{ds_name} CV Results: ")
        print(f"Macro-F1: {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}")
        print(f"Accuracy: {np.mean(cv_accuracy):.4f} ± {np.std(cv_accuracy):.4f}")
        
        # Finally train on ALL train_val_df and test on test_df
        print(f"\nTraining Final Model for {ds_name}...")
        processor = HybridDataProcessor(seq_cols, num_cols, cat_cols)
        processor.fit(train_val_df)
        
        X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr = processor.transform(train_val_df)
        X_seq_te, X_num_te, X_cat_te, y_cls_te, y_ord_te = processor.transform(test_df)
        
        X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr = apply_resampling(
            X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr, strategy=resample_strategy, seed=42
        )
        
        train_loader = create_dataloaders(X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr, train_config.batch_size, shuffle=True)
        test_loader = create_dataloaders(X_seq_te, X_num_te, X_cat_te, y_cls_te, y_ord_te, train_config.batch_size, shuffle=False)
        
        model = HybridCNNBiLSTMAttentionOrdinal(
            seq_input_dim=len(seq_cols),
            num_numeric_context=len(num_cols),
            categorical_cardinalities=processor.cat_cardinalities,
            num_classes=spec.n_classes,
            **model_config.__dict__
        )
        
        class_weights = compute_class_weights(y_cls_tr, spec.n_classes)
        model, _ = train_hybrid_model(model, train_loader, test_loader, train_config, class_weights)
        
        test_metrics, _, _, _ = evaluate_model(model, test_loader)
        
        print(f"Final Locked Test F1: {test_metrics['f1_macro']:.4f}")
        
        report_data = {
            "cv": {
                "Macro-F1": cv_f1_scores,
                "Accuracy": cv_accuracy,
                "RMSE": cv_rmse
            },
            "final_test": test_metrics
        }
        
        write_json(RESULTS_DIR / f"hybrid_repeated_cv_{ds_name}.json", report_data)
        create_markdown_report(report_data, spec, RESULTS_DIR / f"hybrid_cv_report_{ds_name}.md")

if __name__ == "__main__":
    main()
