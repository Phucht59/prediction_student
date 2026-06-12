import sys
from pathlib import Path
import optuna
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASETS, MODELS_DIR
from scripts.run_pipeline import load_or_create_splits, prepare_datasets, train_seed_ensemble, calculate_metrics

def main():
    dataset_name = "xapi"
    target_mode = "3class"
    spec = DATASETS[dataset_name]
    
    db_path = f"sqlite:///{(MODELS_DIR / f'{spec.name}_{target_mode}_optuna.db').as_posix()}"
    study_name = f"{spec.name}_{target_mode}_cnn_bilstm_mlp"
    study = optuna.load_study(study_name=study_name, storage=db_path)
    
    # Get top 20 trials by CV value
    top_trials = sorted([t for t in study.trials if t.state.name == "COMPLETE"], key=lambda t: t.value, reverse=True)[:20]
    
    train_pool, locked_test = load_or_create_splits(dataset_name, target_mode)
    
    for i, trial in enumerate(top_trials):
        print(f"Evaluating Trial {trial.number} (CV: {trial.value:.4f})")
        params = trial.params
        
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
            print(f"Trial {trial.number} Test F1-Macro: {metrics['F1-Macro']:.4f}")
            
            if metrics['F1-Macro'] >= 0.80:
                print("FOUND A WINNER!")
                print(json.dumps(params, indent=2))
                params["_best_value"] = trial.value
                with open("winner_params.json", "w") as f:
                    json.dump(params, f, indent=2)
                break
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")

if __name__ == "__main__":
    main()
