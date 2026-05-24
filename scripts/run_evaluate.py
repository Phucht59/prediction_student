from pathlib import Path
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.split_data import load_train_test_split
from src.evaluate.plot_results import plot_confusion_matrix, plot_f1_comparison
from src.recommend.feature_importance import get_feature_importance
from src.recommend.study_advice import create_study_advice, save_study_advice
from src.utils.config import load_config, make_path


def create_comparison_plots() -> None:
    basic_results = make_path("results/metrics/basic_results.csv")
    deep_results = make_path("results/metrics/deep_results.csv")

    if basic_results.exists():
        plot_f1_comparison(
            basic_results,
            make_path("results/figures/basic_f1_comparison.png"),
            "Basic Model F1-score Comparison",
        )

    if deep_results.exists():
        plot_f1_comparison(
            deep_results,
            make_path("results/figures/deep_f1_comparison.png"),
            "Deep Learning F1-score Comparison",
        )


def create_basic_confusion_matrices(config: dict) -> None:
    labels = ["low", "medium", "high"]

    for dataset_name in config["datasets"]:
        _, test_data = load_train_test_split(dataset_name)
        X_test = test_data.drop(columns=["target"])
        y_test = test_data["target"]

        for model_name in config["basic_models"]:
            model_path = make_path(f"saved_models/basic/{dataset_name}_{model_name}_none.joblib")
            if not model_path.exists():
                continue

            saved_model = joblib.load(model_path)
            model = saved_model["model"]
            label_encoder = saved_model["label_encoder"]
            predictions = label_encoder.inverse_transform(model.predict(X_test))

            plot_confusion_matrix(
                y_test,
                predictions,
                labels,
                f"{dataset_name} {model_name}",
                make_path(f"results/figures/{dataset_name}_{model_name}_confusion_matrix.png"),
            )


def create_study_advice_report(config: dict) -> None:
    advice_lines = []

    for dataset_name in config["datasets"]:
        train_data, _ = load_train_test_split(dataset_name)
        feature_names = train_data.drop(columns=["target"]).columns.tolist()

        for model_name in ["decision_tree", "random_forest", "xgboost"]:
            model_path = make_path(f"saved_models/basic/{dataset_name}_{model_name}_none.joblib")
            if not model_path.exists():
                continue

            saved_model = joblib.load(model_path)
            importance = get_feature_importance(saved_model["model"], feature_names)
            top_features = importance.head(10)["feature"].tolist()
            advice = create_study_advice(top_features)

            advice_lines.append(f"{dataset_name} - {model_name}")
            advice_lines.extend(advice)
            advice_lines.append("")

    save_study_advice(advice_lines, make_path("results/reports/study_advice.txt"))


def main() -> None:
    config = load_config()
    create_comparison_plots()
    create_basic_confusion_matrices(config)
    create_study_advice_report(config)
    print("Evaluation files created.")


if __name__ == "__main__":
    main()

