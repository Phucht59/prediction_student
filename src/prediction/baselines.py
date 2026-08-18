"""Active comparator catalog; the historical boosted-tree comparator is excluded."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ACTIVE_BASELINES = (
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "SVM",
    "MLP",
)


def build_baseline(name: str, *, random_state: int = 42):
    if name == "Logistic Regression":
        return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=random_state)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=random_state, n_jobs=-1)
    if name == "SVM":
        return SVC(class_weight="balanced", probability=True, random_state=random_state)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=250, random_state=random_state)
    raise ValueError(f"unknown active baseline: {name}")


__all__ = ["ACTIVE_BASELINES", "build_baseline"]
