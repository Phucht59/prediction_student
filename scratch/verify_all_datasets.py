import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path("c:/Huflit/kltn")
sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendation import MLPLearningPathEngine, reference_risk_targets
from src.eval_recommendation import _ranking_metrics

for dataset_name in ["student-mat", "student-por", "xapi"]:
    train_path = PROJECT_ROOT / "data" / "processed" / "final" / f"{dataset_name}_3class_train_pool.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "final" / f"{dataset_name}_3class_locked_test.csv"
    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)

    engine = MLPLearningPathEngine(dataset_name, train_frame=train_frame)
    scores = engine.predict_scores(test_frame)
    y_true = reference_risk_targets(test_frame, dataset_name)
    
    print("=" * 40)
    print("Dataset:", dataset_name)
    print("Test size:", len(test_frame))
    
    # Analyze active risks per student
    relevant_counts = y_true.sum(axis=1)
    print("Active risks count distribution:")
    print(pd.Series(relevant_counts).value_counts().sort_index().to_dict())
    
    # Count how many students have relevant > 0
    num_non_zero = int((relevant_counts > 0).sum())
    print("Students with relevant > 0:", num_non_zero)
    
    for k in [1, 3, 5]:
        metrics = _ranking_metrics(y_true, scores, k)
        
        # Verify Precision@k
        precisions = []
        recalls = []
        ndcgs = []
        for truth, row_scores in zip(y_true, scores):
            order = np.argsort(-row_scores)[:k]
            hits = float(truth[order].sum())
            precisions.append(hits / k)
            relevant = float(truth.sum())
            if relevant > 0:
                recalls.append(hits / relevant)
                gains = truth[order] / np.log2(np.arange(2, len(order) + 2))
                ideal_count = min(int(relevant), k)
                ideal = np.ones(ideal_count) / np.log2(np.arange(2, ideal_count + 2))
                ndcgs.append(float(gains.sum() / ideal.sum()))
                
        p_mean = np.mean(precisions)
        r_mean = np.mean(recalls) if recalls else 0.0
        n_mean = np.mean(ndcgs) if ndcgs else 0.0
        
        # Sanity checks
        p_check = sum(precisions) / len(test_frame)
        r_check = sum(recalls) / num_non_zero if num_non_zero > 0 else 0.0
        n_check = sum(ndcgs) / num_non_zero if num_non_zero > 0 else 0.0
        
        print(f"k = {k}:")
        print(f"  Precision: {metrics[f'precision_at_{k}']:.6f} (Check: {p_check:.6f})")
        print(f"  Recall: {metrics[f'recall_at_{k}']:.6f} (Check: {r_check:.6f})")
        print(f"  NDCG: {metrics[f'ndcg_at_{k}']:.6f} (Check: {n_check:.6f})")
        
        # Invariant checks:
        assert p_mean <= 1.0 and r_mean <= 1.0 and n_mean <= 1.0
        assert np.isclose(p_check, metrics[f'precision_at_{k}'])
        assert np.isclose(r_check, metrics[f'recall_at_{k}'])
        assert np.isclose(n_check, metrics[f'ndcg_at_{k}'])

print("\nVerification completed successfully. All invariants held.")
