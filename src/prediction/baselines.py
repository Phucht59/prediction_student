"""Active comparator factory. XGBoost is not constructible."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


ACTIVE_BASELINES = (
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "SVM",
    "MLP",
)

ACTIVE_BASELINE_IDS = ("LR", "DT", "RF", "SVM", "MLP")


def make_svm(seed: int, *, kernel: str = "rbf", C: float = 1.0, gamma: str = "scale", class_weight: str = "balanced"):
    cw = None if class_weight in {None, "none"} else class_weight
    if kernel == "linear":
        base = LinearSVC(C=C, class_weight=cw, max_iter=4000, dual="auto", random_state=seed)
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    return SVC(kernel="rbf", C=C, gamma=gamma, class_weight=cw, probability=True, random_state=seed)


def build_baseline(name: str, *, random_state: int = 42, dataset: str | None = None):
    aliases = {
        "LR": "Logistic Regression",
        "DT": "Decision Tree",
        "RF": "Random Forest",
    }
    name = aliases.get(name, name)
    if name == "Logistic Regression":
        return LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=random_state)
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=random_state)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=random_state, n_jobs=-1)
    if name == "SVM":
        if dataset == "oulad":
            return make_svm(random_state, kernel="linear", C=1.0, class_weight="balanced")
        return make_svm(random_state, kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=250, random_state=random_state)
    if name.upper() in {"XGB", "XGBOOST"}:
        raise ValueError("XGBoost is not an active baseline")
    raise ValueError(f"unknown active baseline: {name}")


__all__ = ["ACTIVE_BASELINES", "ACTIVE_BASELINE_IDS", "build_baseline", "make_svm"]
