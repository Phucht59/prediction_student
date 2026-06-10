import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from src.improved_hybrid.config import DATASETS, HybridModelConfig, TrainingConfig, ensure_dirs, RESULTS_DIR
from src.improved_hybrid.data import read_raw_data, add_targets, HybridDataProcessor, apply_resampling, create_dataloaders
from src.improved_hybrid.features import apply_feature_engineering
from src.improved_hybrid.models import HybridCNNBiLSTMAttentionOrdinal
from src.improved_hybrid.train import compute_class_weights, train_hybrid_model, evaluate_model
from src.improved_hybrid.evaluate import write_json

def objective(trial, X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr,
              X_seq_va, X_num_va, X_cat_va, y_cls_va, y_ord_va, spec, seq_cols, num_cols, cat_cardinalities):
              
    # Suggest hyperparameters
    conv_channels = trial.suggest_categorical("conv_channels", [32, 64, 128])
    bilstm_hidden = trial.suggest_categorical("bilstm_hidden", [32, 64, 128])
    context_hidden = trial.suggest_categorical("context_hidden", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.15, 0.5)
    
    lr = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)
    ordinal_lambda = trial.suggest_float("ordinal_lambda", 0.05, 0.5)
    
    model_config = HybridModelConfig(
        conv_channels=conv_channels,
        bilstm_hidden=bilstm_hidden,
        context_hidden=context_hidden,
        dropout=dropout
    )
    
    train_config = TrainingConfig(
        learning_rate=lr,
        focal_gamma=focal_gamma,
        ordinal_lambda=ordinal_lambda,
        max_epochs=100,
        patience=10
    )
    
    train_loader = create_dataloaders(X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr, train_config.batch_size, shuffle=True)
    val_loader = create_dataloaders(X_seq_va, X_num_va, X_cat_va, y_cls_va, y_ord_va, train_config.batch_size, shuffle=False)
    
    model = HybridCNNBiLSTMAttentionOrdinal(
        seq_input_dim=len(seq_cols),
        num_numeric_context=len(num_cols),
        categorical_cardinalities=cat_cardinalities,
        num_classes=spec.n_classes,
        **model_config.__dict__
    )
    
    class_weights = compute_class_weights(y_cls_tr, spec.n_classes)
    model, _ = train_hybrid_model(model, train_loader, val_loader, train_config, class_weights)
    
    val_metrics, _, _, _ = evaluate_model(model, val_loader)
    
    return val_metrics['f1_macro']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["student-mat", "student-por", "xapi"])
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    ensure_dirs()
    spec = DATASETS[args.dataset]
    print(f"=== Running Optuna for {args.dataset} ===")
    
    raw = read_raw_data(spec)
    df = add_targets(raw, spec)
    df, seq_cols, num_cols, cat_cols = apply_feature_engineering(df, spec)
    
    train_val_df, _ = train_test_split(df, test_size=0.20, stratify=df['target_class'], random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.20, stratify=train_val_df['target_class'], random_state=42)
    
    processor = HybridDataProcessor(seq_cols, num_cols, cat_cols)
    processor.fit(train_df)
    
    X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr = processor.transform(train_df)
    X_seq_va, X_num_va, X_cat_va, y_cls_va, y_ord_va = processor.transform(val_df)
    
    resample_strategy = "adasyn" if spec.kind == "xapi" else "smote"
    if args.dataset == "student-por":
        resample_strategy = "none"
        
    X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr = apply_resampling(
        X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr, strategy=resample_strategy, seed=42
    )
    
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_seq_tr, X_num_tr, X_cat_tr, y_cls_tr, y_ord_tr,
                                X_seq_va, X_num_va, X_cat_va, y_cls_va, y_ord_va, 
                                spec, seq_cols, num_cols, processor.cat_cardinalities),
        n_trials=args.trials
    )
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Val F1-Macro): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    write_json(RESULTS_DIR / f"hybrid_optuna_best_{args.dataset}.json", trial.params)

if __name__ == "__main__":
    main()
