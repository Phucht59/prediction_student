import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

from src.evaluate.metrics import calculate_metrics, get_classification_report
from src.evaluate.plot_results import plot_confusion_matrix, plot_training_history
from src.models.model_selector import get_deep_model
from src.utils.config import make_path


def split_features_and_target(data: pd.DataFrame):
    X = data.drop(columns=["target"]).values
    y = data["target"].values
    return X, y


def reshape_for_deep_learning(X: np.ndarray) -> np.ndarray:
    return X.reshape((X.shape[0], X.shape[1], 1))


def train_deep_model(
    dataset_name: str,
    model_name: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    deep_learning_config: dict,
) -> dict:
    X_train, y_train = split_features_and_target(train_data)
    X_test, y_test = split_features_and_target(test_data)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    X_train = reshape_for_deep_learning(X_train)
    X_test = reshape_for_deep_learning(X_test)

    number_of_classes = len(label_encoder.classes_)
    model = get_deep_model(model_name, X_train.shape[1:], number_of_classes)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=deep_learning_config["early_stopping_patience"],
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train_encoded,
        epochs=deep_learning_config["epochs"],
        batch_size=deep_learning_config["batch_size"],
        validation_split=deep_learning_config["validation_split"],
        callbacks=[early_stopping],
        verbose=1,
    )

    prediction_probabilities = model.predict(X_test)
    predictions_encoded = np.argmax(prediction_probabilities, axis=1)
    predictions = label_encoder.inverse_transform(predictions_encoded)

    metrics = calculate_metrics(y_test, predictions)
    result = {
        "dataset": dataset_name,
        "model": model_name,
        **metrics,
    }

    save_deep_model(model, dataset_name, model_name)
    save_deep_label_encoder(label_encoder, dataset_name, model_name)
    save_deep_report(y_test, predictions, dataset_name, model_name)
    save_deep_training_plot(history, dataset_name, model_name)
    save_deep_confusion_matrix(y_test, predictions, label_encoder.classes_.tolist(), dataset_name, model_name)

    return result


def save_deep_model(model, dataset_name: str, model_name: str) -> None:
    output_path = make_path(f"saved_models/deep/{dataset_name}_{model_name}.keras")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)


def save_deep_label_encoder(label_encoder, dataset_name: str, model_name: str) -> None:
    output_path = make_path(f"saved_models/deep/{dataset_name}_{model_name}_label_encoder.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(label_encoder, output_path)


def save_deep_report(y_true, y_pred, dataset_name: str, model_name: str) -> None:
    report = get_classification_report(y_true, y_pred)
    output_path = make_path(f"results/reports/{dataset_name}_{model_name}_report.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path)


def save_deep_training_plot(history, dataset_name: str, model_name: str) -> None:
    output_path = make_path(f"results/figures/{dataset_name}_{model_name}_training.png")
    plot_training_history(history, f"{dataset_name} {model_name}", output_path)


def save_deep_confusion_matrix(y_true, y_pred, labels: list[str], dataset_name: str, model_name: str) -> None:
    output_path = make_path(f"results/figures/{dataset_name}_{model_name}_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, labels, f"{dataset_name} {model_name}", output_path)
