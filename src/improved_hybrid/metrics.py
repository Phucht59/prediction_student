import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    r2_score,
    confusion_matrix
)

def compute_metrics(y_true_class, y_pred_class, y_true_ord=None, y_pred_ord=None):
    metrics = {
        "accuracy": accuracy_score(y_true_class, y_pred_class),
        "balanced_accuracy": balanced_accuracy_score(y_true_class, y_pred_class),
        "precision_macro": precision_score(y_true_class, y_pred_class, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_class, y_pred_class, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true_class, y_pred_class, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true_class, y_pred_class, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true_class, y_pred_class, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true_class, y_pred_class, average="weighted", zero_division=0),
    }
    
    if y_true_ord is not None and y_pred_ord is not None:
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true_ord, y_pred_ord))
        metrics["r2"] = r2_score(y_true_ord, y_pred_ord)
        
    return metrics
