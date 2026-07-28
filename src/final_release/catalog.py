"""Stable public identities used by every final dataset table."""

COMPARISON_MODELS = (
    ("cnn_bilstm", "CNN-BiLSTM"),
    ("cnn_only", "CNN-only"),
    ("bilstm_only", "BiLSTM-only"),
    ("logistic_regression", "Logistic Regression"),
    ("decision_tree", "Decision Tree"),
    ("random_forest", "Random Forest"),
    ("hist_gradient_boosting", "HistGradientBoosting"),
    ("svm", "SVM"),
    ("xgboost", "XGBoost"),
    ("mlp", "MLP"),
)

OFFICIAL_MODELS = {
    "student_mat": {
        "model_id": "cnn_bilstm_mat",
        "official_name": "CNN-BiLSTM MAT",
        "dataset": "student-mat",
        "task": "multiclass_student_performance",
        "classes": ["Low", "Medium", "High"],
        "class_name": "CNNBiLSTMStudentMatModel",
    },
    "student_por": {
        "model_id": "cnn_bilstm_por",
        "official_name": "CNN-BiLSTM POR",
        "dataset": "student-por",
        "task": "multiclass_student_performance",
        "classes": ["Low", "Medium", "High"],
        "class_name": "CNNBiLSTMStudentPorModel",
    },
    "oulad": {
        "model_id": "cnn_bilstm_oulad",
        "official_name": "CNN-BiLSTM OULAD",
        "dataset": "oulad",
        "task": "binary_student_risk",
        "classes": ["Not-at-risk", "At-risk"],
        "class_name": "CNNBiLSTMOULADModel",
    },
}

RECOMMENDATION_SYSTEM = {
    "system_id": "student_risk_recommendation_system",
    "official_name": "Student Risk-Based Recommendation System",
}
