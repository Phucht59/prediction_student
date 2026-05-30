from __future__ import annotations

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def get_classification_baselines(seed: int = 42) -> dict:
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=seed),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "decision_tree": DecisionTreeClassifier(random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "svm_rbf": SVC(kernel="rbf", random_state=seed),
    }


def get_regression_baselines(seed: int = 42) -> dict:
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "ridge": Ridge(random_state=seed),
        "decision_tree_regressor": DecisionTreeRegressor(random_state=seed),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting_regressor": GradientBoostingRegressor(random_state=seed),
        "svr_rbf": SVR(kernel="rbf"),
    }


def optional_model_skip_reasons() -> list[dict[str, str]]:
    return []
