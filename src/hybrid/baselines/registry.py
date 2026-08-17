"""Registry and score interface for the six preregistered baseline families."""
from __future__ import annotations

BASELINE_FAMILIES = (
    "logistic_regression", "svm", "random_forest", "xgboost", "catboost", "mlp",
)

def get_ranking_score(estimator, X):
    """Return a ranking score without calibration; SVM deliberately uses margin."""
    if getattr(estimator, "_hybrid_family", None) == "svm" or estimator.__class__.__name__ == "SVC":
        return estimator.decision_function(X)
    return estimator.predict_proba(X)[:, 1]
