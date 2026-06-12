"""Evaluation reporting and PostgreSQL persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DATABASE_URL, METRICS_DIR, MODELS_DIR, POSTGRES_CONFIG, REPORTS_DIR, ROOT_DIR
from src.explainability import CLASS_NAMES
from src.utils import setup_logger

logger = setup_logger("evaluation")


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if pd.isna(value):
        return None
    return value


def ensure_database_schema(connection) -> None:
    schema_path = ROOT_DIR / "database" / "schema.sql"
    with connection.cursor() as cursor:
        cursor.execute(schema_path.read_text(encoding="utf-8"))


def persist_evaluation_to_postgres(
    dataset_name: str,
    model_name: str,
    original_features: pd.DataFrame,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    confidences: np.ndarray,
    probabilities: np.ndarray,
    learning_paths: pd.DataFrame,
    metrics: dict[str, float],
    postgres_config: dict[str, Any] | None = None,
) -> int:
    """Insert locked-test features, predictions and learning paths atomically."""

    try:
        import psycopg2
        from psycopg2.extras import Json, execute_values
    except ImportError as exc:
        raise RuntimeError("psycopg2-binary is required for PostgreSQL persistence.") from exc

    row_count = len(original_features)
    arrays = [true_labels, predicted_labels, confidences, probabilities, learning_paths]
    if any(len(values) != row_count for values in arrays):
        raise ValueError("Prediction, confidence, feature and learning-path row counts must match.")

    config = dict(postgres_config or POSTGRES_CONFIG)
    database_url = config.pop("database_url", None) or DATABASE_URL
    if database_url.startswith("postgresql+psycopg2://"):
        database_url = database_url.replace("postgresql+psycopg2://", "postgresql://", 1)
    connection = psycopg2.connect(database_url) if database_url else psycopg2.connect(**config)
    try:
        ensure_database_schema(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO paper_runs (
                    generated_at, result_rows, summary_rows,
                    postgres_status, postgres_message, run_payload
                ) VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING paper_run_id
                """,
                (
                    datetime.now(timezone.utc),
                    row_count,
                    1,
                    "completed",
                    "Locked-test evaluation persisted successfully.",
                    Json({"dataset": dataset_name, "architecture": "CNN-BiLSTM + MLP"}),
                ),
            )
            run_id = cursor.fetchone()[0]

            prediction_rows = []
            for row_index, (_, feature_row) in enumerate(original_features.reset_index(drop=True).iterrows()):
                features = _json_safe(feature_row.to_dict())
                probability = {
                    CLASS_NAMES[class_index]: float(probabilities[row_index][class_index])
                    for class_index in range(probabilities.shape[1])
                }
                prediction_rows.append(
                    (
                        run_id,
                        model_name,
                        dataset_name,
                        "locked_test",
                        row_index,
                        int(true_labels[row_index]),
                        int(predicted_labels[row_index]),
                        CLASS_NAMES[int(true_labels[row_index])],
                        CLASS_NAMES[int(predicted_labels[row_index])],
                        Json(probability),
                        float(confidences[row_index]),
                        Json(features),
                        features.get("G1"),
                        features.get("G2"),
                        features.get("G3"),
                        features.get("Class"),
                    )
                )

            execute_values(
                cursor,
                """
                INSERT INTO paper_predictions (
                    run_id, model_name, dataset, split, row_index,
                    true_label, predicted_label, true_label_name,
                    predicted_label_name, probability, confidence,
                    original_features, G1, G2, G3, xapi_class
                ) VALUES %s
                """,
                prediction_rows,
            )

            recommendation_rows = []
            for row_index, recommendation in learning_paths.reset_index(drop=True).iterrows():
                feature_snapshot = _json_safe(original_features.iloc[row_index].to_dict())
                path_payload = json.loads(recommendation["learning_path"])
                risk_payload = json.loads(recommendation["risk_factors"])
                recommendation_rows.append(
                    (
                        run_id,
                        dataset_name,
                        model_name,
                        row_index,
                        int(true_labels[row_index]),
                        int(predicted_labels[row_index]),
                        float(confidences[row_index]),
                        recommendation["risk_band"],
                        Json(feature_snapshot),
                        Json(path_payload),
                        Json(
                            {
                                "headline": recommendation["headline"],
                                "risk_factors": risk_payload,
                            }
                        ),
                    )
                )

            execute_values(
                cursor,
                """
                INSERT INTO paper_learning_recommendations (
                    run_id, dataset, model_name, row_index, true_label,
                    predicted_label, confidence, risk_band, feature_snapshot,
                    recommended_learning_path, recommendation_payload
                ) VALUES %s
                """,
                recommendation_rows,
            )

            cursor.execute(
                """
                INSERT INTO paper_evaluation_metrics (
                    run_id, dataset, model_name, protocol_class,
                    accuracy, precision_macro, recall_macro, f1_macro,
                    rmse, r2, metric_payload
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    dataset_name,
                    model_name,
                    "locked_test_20_percent",
                    metrics["Accuracy"],
                    metrics["Precision-Macro"],
                    metrics["Recall-Macro"],
                    metrics["F1-Macro"],
                    metrics["RMSE"],
                    metrics["R2"],
                    Json(_json_safe(metrics)),
                ),
            )

        connection.commit()
        logger.info("Persisted PostgreSQL evaluation run %s for %s.", run_id, dataset_name)
        return int(run_id)
    except Exception as exc:
        connection.rollback()
        raise RuntimeError(f"PostgreSQL persistence failed for {dataset_name}: {exc}") from exc
    finally:
        connection.close()


def create_summary_report(dataset_name: str, target_mode: str) -> Path:
    metrics_path = METRICS_DIR / f"{dataset_name}_{target_mode}_locked_test_metrics.json"
    params_path = MODELS_DIR / f"{dataset_name}_{target_mode}_best_params.json"
    report_path = REPORTS_DIR / f"summary_report_{dataset_name}_{target_mode}.md"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    best_params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}
    lines = [
        "# Báo cáo tổng kết mô hình CNN-BiLSTM + MLP",
        "",
        f"- **Dataset**: {dataset_name}",
        f"- **Bài toán**: {target_mode}",
        "- **Đánh giá**: locked test 20%, không tham gia Optuna",
        "",
        "## Kết quả",
    ]
    for key, value in metrics.items():
        lines.append(f"- **{key}**: {value:.4f}")
    lines.extend(["", "## Siêu tham số"])
    for key, value in best_params.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Khuyến nghị",
            "Hệ thống ánh xạ các yếu tố rủi ro sang lộ trình học tập theo tuần, gồm mục tiêu, hành động và mốc theo dõi.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
