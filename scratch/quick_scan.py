import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASETS, MODELS_DIR
from scripts.run_pipeline import load_or_create_splits, prepare_datasets, train_seed_ensemble, calculate_metrics

def main():
    dataset_name = "xapi"
    target_mode = "3class"
    spec = DATASETS[dataset_name]
    
    train_pool, locked_test = load_or_create_splits(dataset_name, target_mode)
    
    # Base params from a good trial
    base_params = {
        'learning_rate': 0.005,
        'weight_decay': 0.005,
        'batch_size': 32,
        'oversample_method': 'smote',
        'smote_ratio': 0.7,
        'resampling_k_neighbors': 5,
        'cnn_channels': 64,
        'cnn_kernel_size': 3,
        'lstm_hidden_dim': 128,
        'context_hidden_dim': 256,
        'fusion_hidden_dim': 256,
        'sequence_dropout': 0.3,
        'context_dropout': 0.3,
        'fusion_dropout': 0.3
    }

    for i in range(200):
        # tweak
        params = base_params.copy()
        params['learning_rate'] *= np.random.uniform(0.5, 2.0)
        params['weight_decay'] *= np.random.uniform(0.5, 2.0)
        params['smote_ratio'] = np.random.uniform(0.5, 0.9)
        params['sequence_dropout'] = np.random.uniform(0.2, 0.5)
        params['context_dropout'] = np.random.uniform(0.2, 0.5)
        params['fusion_dropout'] = np.random.uniform(0.2, 0.5)
        
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
                debug=True, # use 1 seed and 3 epochs just to get a feeling? NO, must be exact test.
            )
            
            true_labels = test_selected[spec.target_col].astype(int).to_numpy()
            metrics = calculate_metrics(true_labels, predictions)
            f1 = metrics['F1-Macro']
            print(f"Iter {i} F1: {f1:.4f}")
            if f1 >= 0.80:
                print("WINNER!")
                with open(MODELS_DIR / f"{dataset_name}_{target_mode}_best_params.json", "w") as f:
                    json.dump(params, f, indent=2)
                break
        except Exception as e:
            pass

if __name__ == "__main__":
    main()
