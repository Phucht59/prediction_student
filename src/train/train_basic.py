import joblib
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from sklearn.preprocessing import LabelEncoder

from src.evaluate.metrics import calculate_metrics, get_classification_report
from src.models.model_selector import get_basic_model
from src.recommend.feature_importance import get_feature_importance, save_feature_importance
from src.utils.config import make_path


def get_resampler(method_name: str, random_seed: int):
    if method_name == "none":
        return None
    if method_name == "smote":
        return SMOTE(random_state=random_seed)
    if method_name == "adasyn":
        return ADASYN(random_state=random_seed)
    if method_name == "borderline_smote":
        return BorderlineSMOTE(random_state=random_seed)
    raise ValueError(f"Unknown imbalance method '{method_name}'.")


def split_features_and_target(data: pd.DataFrame):
    X = data.drop(columns=["target"])
    y = data["target"]
    return X, y


def train_basic_model(
    dataset_name: str,
    model_name: str,
    imbalance_method: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    random_seed: int,
) -> dict:
    X_train, y_train = split_features_and_target(train_data)
    X_test, y_test = split_features_and_target(test_data)
    feature_names = X_train.columns.tolist()

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    resampler = get_resampler(imbalance_method, random_seed)
    if resampler is not None:
        X_train, y_train_encoded = resampler.fit_resample(X_train, y_train_encoded)

    model = get_basic_model(model_name, random_seed)
    model.fit(X_train, y_train_encoded)

    predictions_encoded = model.predict(X_test)
    predictions = label_encoder.inverse_transform(predictions_encoded)

    metrics = calculate_metrics(y_test, predictions)
    result = {
        "dataset": dataset_name,
        "model": model_name,
        "imbalance_method": imbalance_method,
        **metrics,
    }

    save_basic_model(model, label_encoder, dataset_name, model_name, imbalance_method)
    save_basic_report(y_test, predictions, dataset_name, model_name, imbalance_method)
    save_basic_feature_importance(model, feature_names, dataset_name, model_name, imbalance_method)

    return result


def save_basic_model(model, label_encoder, dataset_name: str, model_name: str, imbalance_method: str) -> None:
    output_path = make_path(f"saved_models/basic/{dataset_name}_{model_name}_{imbalance_method}.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "label_encoder": label_encoder}, output_path)


def save_basic_report(y_true, y_pred, dataset_name: str, model_name: str, imbalance_method: str) -> None:
    report = get_classification_report(y_true, y_pred)
    output_path = make_path(f"results/reports/{dataset_name}_{model_name}_{imbalance_method}_report.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path)


def save_basic_feature_importance(
    model,
    feature_names: list[str],
    dataset_name: str,
    model_name: str,
    imbalance_method: str,
) -> None:
    if model_name not in ["decision_tree", "random_forest", "xgboost"]:
        return

    importance = get_feature_importance(model, feature_names)
    output_path = make_path(f"results/reports/{dataset_name}_{model_name}_{imbalance_method}_feature_importance.csv")
    save_feature_importance(importance, output_path)
