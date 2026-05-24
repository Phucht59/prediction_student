from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def get_basic_models(random_seed: int) -> dict:
    return {
        "decision_tree": DecisionTreeClassifier(random_state=random_seed),
        "random_forest": RandomForestClassifier(random_state=random_seed),
        "svm": SVC(),
        "xgboost": XGBClassifier(
            random_state=random_seed,
            eval_metric="mlogloss",
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_seed),
    }

