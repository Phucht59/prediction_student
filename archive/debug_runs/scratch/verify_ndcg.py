import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path("c:/Huflit/kltn")
sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendation import MLPLearningPathEngine, reference_risk_targets

# Load data and engine
dataset_name = "student-mat"
train_path = PROJECT_ROOT / "data" / "processed" / "final" / f"{dataset_name}_3class_train_pool.csv"
test_path = PROJECT_ROOT / "data" / "processed" / "final" / f"{dataset_name}_3class_locked_test.csv"
train_frame = pd.read_csv(train_path)
test_frame = pd.read_csv(test_path)

engine = MLPLearningPathEngine(dataset_name, train_frame=train_frame)
scores = engine.predict_scores(test_frame)
y_true = reference_risk_targets(test_frame, dataset_name)

print("Test set size:", len(test_frame))
print("y_true sum per student (active risks):")
relevant_counts = y_true.sum(axis=1)
print(pd.Series(relevant_counts).value_counts())

# Let's calculate row by row
recalls_1 = []
ndcgs_1 = []
precisions_1 = []

recalls_3 = []
ndcgs_3 = []
precisions_3 = []

for idx, (truth, row_scores) in enumerate(zip(y_true, scores)):
    relevant = float(truth.sum())
    
    # k = 1
    order_1 = np.argsort(-row_scores)[:1]
    hits_1 = float(truth[order_1].sum())
    precisions_1.append(hits_1 / 1)
    if relevant > 0:
        recalls_1.append(hits_1 / relevant)
        gains_1 = truth[order_1] / np.log2([2])
        ideal_1 = np.ones(1) / np.log2([2])
        ndcgs_1.append(float(gains_1.sum() / ideal_1.sum()))
        
    # k = 3
    order_3 = np.argsort(-row_scores)[:3]
    hits_3 = float(truth[order_3].sum())
    precisions_3.append(hits_3 / 3)
    if relevant > 0:
        recalls_3.append(hits_3 / relevant)
        gains_3 = truth[order_3] / np.log2(np.arange(2, 5))
        ideal_count_3 = min(int(relevant), 3)
        ideal_3 = np.ones(ideal_count_3) / np.log2(np.arange(2, ideal_count_3 + 2))
        ndcgs_3.append(float(gains_3.sum() / ideal_3.sum()))

print("\nComputed means:")
print("Precision@1:", np.mean(precisions_1))
print("Recall@1:", np.mean(recalls_1))
print("NDCG@1:", np.mean(ndcgs_1))
print("Precision@3:", np.mean(precisions_3))
print("Recall@3:", np.mean(recalls_3))
print("NDCG@3:", np.mean(ndcgs_3))

# Inspect cases where hits_1 is 0 and relevant > 0 (if any)
failures = []
for idx, (truth, row_scores) in enumerate(zip(y_true, scores)):
    relevant = float(truth.sum())
    if relevant > 0:
        order_1 = np.argsort(-row_scores)[:1]
        hits_1 = float(truth[order_1].sum())
        if hits_1 == 0:
            failures.append((idx, truth, row_scores, order_1))

print("\nNumber of students with relevant > 0 but top-1 prediction is incorrect:", len(failures))
for idx, truth, row_scores, order_1 in failures[:5]:
    print(f"Student {idx}: truth={truth}, scores={np.round(row_scores, 4)}, top-1 predicted={order_1}")
