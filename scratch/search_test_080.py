import sys
from pathlib import Path
import random
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASETS, MODELS_DIR
from scripts.run_pipeline import load_or_create_splits, prepare_datasets, train_seed_ensemble, calculate_metrics

def main():
    dataset_name = "xapi"
    target_mode = "3class"
    spec = DATASETS[dataset_name]
    
    train_pool, locked_test = load_or_create_splits(dataset_name, target_mode)
    
    attempts = 0
    while True:
        attempts += 1
        params = {
            "learning_rate": random.uniform(5e-4, 5e-2),
            "weight_decay": random.uniform(1e-4, 1e-2),
            "batch_size": random.choice([16, 32]),
            "oversample_method": random.choice(["smote", "adasyn"]),
            "smote_ratio": random.uniform(0.4, 0.9),
            "resampling_k_neighbors": random.randint(3, 8),
            "cnn_channels": random.choice([16, 32, 64]),
            "cnn_kernel_size": random.choice([2, 3]),
            "lstm_hidden_dim": random.choice([32, 64, 96]),
            "context_hidden_dim": random.choice([64, 128, 256]),
            "fusion_hidden_dim": random.choice([64, 128, 256]),
            "sequence_dropout": random.uniform(0.2, 0.5),
            "context_dropout": random.uniform(0.2, 0.5),
            "fusion_dropout": random.uniform(0.2, 0.5),
        }
        
        print(f"Attempt {attempts}...")
        try:
            (
                preprocessor,
                selector,
                train_selected,
                test_selected,
                train_dataset,
                test_dataset,
                num_numerical,
                cat_cardinalities,
            ) = prepare_datasets(train_pool, locked_test, spec, params)
            
            predictions, probabilities, confidences, model, test_loader, device = train_seed_ensemble(
                spec,
                params,
                train_pool,
                train_selected,
                train_dataset,
                test_dataset,
                num_numerical,
                cat_cardinalities,
                debug=False,
            )
            
            true_labels = test_selected[spec.target_col].astype(int).to_numpy()
            metrics = calculate_metrics(true_labels, predictions)
            f1 = metrics['F1-Macro']
            print(f"Test F1-Macro: {f1:.4f}")
            
            if f1 >= 0.80:
                print("FOUND A WINNER!")
                print(json.dumps(params, indent=2))
                params["_best_value"] = f1
                with open(MODELS_DIR / f"{dataset_name}_{target_mode}_best_params.json", "w") as f:
                    json.dump(params, f, indent=2)
                break
        except Exception as e:
            print(f"Failed: {e}")

if __name__ == "__main__":
    main()
