import argparse
from pathlib import Path
import json
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ensure_dirs, MODELS_DIR, REPORTS_DIR, DATASETS, FIXED_SEEDS, DEFAULT_SEED, METRICS_DIR, PREDICTIONS_DIR, EXPLANATIONS_DIR
from src.utils import setup_logger, set_seed
from src.data_pipeline import create_and_save_locked_test, load_splits, apply_feature_engineering, FeatureSelector, DataPreprocessor, StudentDataset
from src.models import create_model
# Actually, train_pipeline functions and evaluation functions need to be imported if we put logic there.
# Let's import the whole pipeline execution logic. 
# Wait, currently train_pipeline just contains the classes/functions from train.py and optuna_search.py.
# To make it simple, we can keep the logic of run_full_v27_pipeline.py. But since we delete old scripts, we must put the run_* logic here.

logger = setup_logger("run_pipeline")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--target-mode", type=str, default="3class")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--params-json", type=str, default=None, help="JSON string of best params to skip Optuna")
    args = parser.parse_args()
    
    ensure_dirs()
    set_seed(DEFAULT_SEED)
    
    logger.info(f"=== Starting E2E Pipeline for {args.dataset} ({args.target_mode}) ===")
    
    # 1. Data Splitting
    from src.data_pipeline import load_splits
    spec = DATASETS[args.dataset]
    try:
        train_pool, locked_test = load_splits(args.dataset, args.target_mode)
    except FileNotFoundError:
        logger.info("Splits not found. Creating locked test...")
        import pandas as pd
        from src.config import RAW_DIR
        df = pd.read_csv(RAW_DIR / spec.raw_file, sep=spec.csv_sep)
        from src.data_pipeline import create_and_save_locked_test
        create_and_save_locked_test(df, args.dataset, args.target_mode)
        train_pool, locked_test = load_splits(args.dataset, args.target_mode)
        
    if args.params_json:
        logger.info("Using provided params-json. Skipping Optuna.")
        p_path = Path(args.params_json)
        if p_path.exists():
            with open(p_path, "r") as f:
                best_p = json.load(f)
        else:
            best_p = json.loads(args.params_json)
        class DummyStudy:
            def __init__(self, val, params):
                self.best_value = val
                self.best_params = params
        study = DummyStudy(0.0, best_p)
    else:
        # 2. Run Optuna Optimization
        logger.info(f"Starting Optuna optimization with {args.n_trials} trials...")
        import optuna
        from src.train_pipeline import objective
        
        study = optuna.create_study(direction="maximize")
        # For speed in debug mode, use 1 trial
        n_trials = 1 if args.debug else args.n_trials
        study.optimize(lambda trial: objective(trial, train_pool, spec, args.target_mode, cv_folds=5), n_trials=n_trials)
        
        logger.info("Best Optuna Trial:")
        logger.info(f"  F1-Macro (CV): {study.best_value:.4f}")
        logger.info(f"  Params: {study.best_params}")
        
    # 3. Final Retraining on Locked Test using Seed Ensemble
    logger.info("Starting Final Evaluation on Locked Test with Fixed Seeds...")
    best_p = study.best_params

    
    from src.data_pipeline import DataPreprocessor, apply_feature_engineering, FeatureSelector, StudentDataset
    from src.models import create_model, HybridLoss
    from src.train_pipeline import train_model, calculate_class_weights
    from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, mean_squared_error, r2_score
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import numpy as np
    from src.config import TrainingConfig, FIXED_SEEDS
    
    # Preprocess full train vs test
    preprocessor = DataPreprocessor(target_col=spec.target_col, oversample_method=best_p["oversample_method"])
    train_prep = preprocessor.fit_transform(train_pool)
    test_prep = preprocessor.transform(locked_test)
    
    train_eng = apply_feature_engineering(train_prep, spec.kind)
    test_eng = apply_feature_engineering(test_prep, spec.kind)
    
    # Feature Selection
    selector = FeatureSelector(target_col=spec.target_col, use_feature_selection=True)
    train_sel = selector.fit_transform(train_eng, preprocessor.numerical_cols, preprocessor.categorical_cols)
    test_sel = selector.transform(test_eng)
    
    # Sequence/Cat specs
    seq_cols_list = ["G1", "G2"] if spec.kind == "student" else []
    cat_cardinalities = [len(preprocessor.label_encoders[c].classes_) for c in preprocessor.categorical_cols if c in selector.selected_features and c not in seq_cols_list]
    num_numerical = len([c for c in preprocessor.numerical_cols if c in selector.selected_features and c not in seq_cols_list])
    
    # Dataset
    train_ds = StudentDataset(train_sel, spec.kind, spec.target_col, preprocessor.numerical_cols, preprocessor.categorical_cols)
    test_ds = StudentDataset(test_sel, spec.kind, spec.target_col, preprocessor.numerical_cols, preprocessor.categorical_cols)
    
    train_loader = DataLoader(train_ds, batch_size=best_p["batch_size"], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=best_p["batch_size"], shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3 if args.target_mode == "3class" else 5
    
    all_preds = []
    y_train_arr = train_sel[spec.target_col].values
    y_test_arr = test_sel[spec.target_col].values
    
    for seed in FIXED_SEEDS:
        set_seed(seed)
        model = create_model(spec.kind, best_p, num_numerical, cat_cardinalities).to(device)
        
        # Determine focal gamma
        fgamma = 0.0 if best_p["oversample_method"] != "none" else best_p["focal_gamma"]
        class_weights = calculate_class_weights(y_train_arr, num_classes).to(device)
        
        criterion = HybridLoss(class_weights=class_weights, gamma=fgamma, lambda_ordinal=best_p["lambda_ordinal"])
        optimizer = optim.Adam(model.parameters(), lr=best_p["learning_rate"], weight_decay=best_p["weight_decay"])
        
        tconfig = TrainingConfig(max_epochs=80, patience=20)
        
        logger.info(f"Training Seed {seed}...")
        model, _, _ = train_model(model, train_loader, train_loader, criterion, optimizer, tconfig, device)
        
        # Predict on locked test
        model.eval()
        preds = []
        with torch.no_grad():
            for sq, nx, cx, _, _ in test_loader:
                sq = sq.to(device) if sq is not None else None
                nx = nx.to(device) if nx is not None else None
                cx = cx.to(device) if cx is not None else None
                logits, expected_val = model(sq, nx, cx)
                probs = torch.softmax(logits, dim=1)
                final_preds = torch.argmax(probs, dim=1)
                preds.extend(final_preds.cpu().numpy())
        all_preds.append(preds)
        
    # Ensemble voting
    from scipy.stats import mode
    ensemble_preds, _ = mode(np.array(all_preds), axis=0, keepdims=False)
    
    final_f1 = f1_score(y_test_arr, ensemble_preds, average='macro')
    logger.info(f"=== FINAL LOCKED TEST RESULTS FOR {args.dataset} ===")
    logger.info(f"F1-Macro (Ensemble): {final_f1:.4f}")
    logger.info("\n" + classification_report(y_test_arr, ensemble_preds))
    
    # Save Report
    report_file = REPORTS_DIR / f"{args.dataset}_{args.target_mode}_final_report.txt"
    with open(report_file, "w") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Target Mode: {args.target_mode}\n")
        f.write(f"Optuna Best CV F1: {study.best_value:.4f}\n")
        f.write(f"Best Params: {json.dumps(best_p, indent=2)}\n\n")
        f.write(f"Final Locked Test F1-Macro: {final_f1:.4f}\n")
        f.write(classification_report(y_test_arr, ensemble_preds))
        
    logger.info(f"Report saved to {report_file}")

    # Save JSON metrics
    acc = accuracy_score(y_test_arr, ensemble_preds)
    prec_macro = precision_score(y_test_arr, ensemble_preds, average='macro', zero_division=0)
    rec_macro = recall_score(y_test_arr, ensemble_preds, average='macro', zero_division=0)
    rmse = float(np.sqrt(mean_squared_error(y_test_arr, ensemble_preds)))
    r2 = float(r2_score(y_test_arr, ensemble_preds))
    
    metrics_data = {
        "Accuracy": acc,
        "F1-Macro": final_f1,
        "Precision-Macro": prec_macro,
        "Recall-Macro": rec_macro,
        "RMSE": rmse,
        "R2": r2
    }
    
    metrics_json_file = METRICS_DIR / f"{args.dataset}_{args.target_mode}_locked_test_metrics.json"
    with open(metrics_json_file, "w") as f:
        json.dump(metrics_data, f, indent=4)
        
    logger.info(f"JSON metrics saved to {metrics_json_file}")

if __name__ == "__main__":
    main()
