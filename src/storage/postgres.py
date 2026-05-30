from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is optional at runtime.
    load_dotenv = None


metadata = MetaData()


student_records = Table(
    "student_records",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("dataset_name", String(64), nullable=False),
    Column("scenario", String(64), nullable=False),
    Column("split_name", String(32), nullable=True),
    Column("source_row_id", Integer, nullable=False),
    Column("target_class", Integer, nullable=True),
    Column("target_class_name", String(32), nullable=True),
    Column("target_regression", Float, nullable=True),
    Column("features", JSON, nullable=False),
    Column("raw_record", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    UniqueConstraint(
        "dataset_name",
        "scenario",
        "split_name",
        "source_row_id",
        name="uq_student_records_source",
    ),
)


prediction_results = Table(
    "prediction_results",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("student_record_id", ForeignKey("student_records.id", ondelete="SET NULL"), nullable=True),
    Column("dataset_name", String(64), nullable=False),
    Column("scenario", String(64), nullable=False),
    Column("model_name", String(128), nullable=False),
    Column("model_version", String(128), nullable=True),
    Column("predicted_class", Integer, nullable=False),
    Column("predicted_label", String(32), nullable=True),
    Column("confidence", Float, nullable=True),
    Column("probabilities", JSON, nullable=True),
    Column("input_features", JSON, nullable=True),
    Column("run_metadata", JSON, nullable=True),
    Column("predicted_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)


def get_database_url(env_file: str | Path = ".env") -> str:
    """Read DATABASE_URL from the environment or an optional .env file."""
    if load_dotenv is not None:
        load_dotenv(env_file)
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured. Example: "
            "postgresql+psycopg2://user:password@localhost:5432/student_performance"
        )
    return database_url


def create_postgres_engine(database_url: str | None = None) -> Engine:
    return create_engine(database_url or get_database_url(), future=True)


def create_tables(engine: Engine) -> None:
    metadata.create_all(engine)


def _json_ready(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def dataframe_records(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    scenario: str,
    split_name: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    target_columns = {"target_class", "target_class_name", "target_regression"}
    for source_row_id, row in df.reset_index(drop=True).iterrows():
        raw_record = {column: _json_ready(value) for column, value in row.items()}
        features = {
            column: value
            for column, value in raw_record.items()
            if column not in target_columns and column not in {"G3", "Class"}
        }
        records.append(
            {
                "dataset_name": dataset_name,
                "scenario": scenario,
                "split_name": split_name,
                "source_row_id": int(source_row_id),
                "target_class": raw_record.get("target_class"),
                "target_class_name": raw_record.get("target_class_name"),
                "target_regression": raw_record.get("target_regression"),
                "features": features,
                "raw_record": raw_record,
            }
        )
    return records


def import_student_records(
    engine: Engine,
    df: pd.DataFrame,
    *,
    dataset_name: str,
    scenario: str,
    split_name: str | None = None,
) -> int:
    records = dataframe_records(
        df,
        dataset_name=dataset_name,
        scenario=scenario,
        split_name=split_name,
    )
    if not records:
        return 0

    statement = insert(student_records).values(records)
    statement = statement.on_conflict_do_nothing(
        constraint="uq_student_records_source",
    )
    with engine.begin() as connection:
        result = connection.execute(statement)
    return int(result.rowcount or 0)


def save_prediction_result(
    engine: Engine,
    *,
    dataset_name: str,
    scenario: str,
    model_name: str,
    predicted_class: int,
    student_record_id: int | None = None,
    model_version: str | None = None,
    predicted_label: str | None = None,
    confidence: float | None = None,
    probabilities: dict[str, float] | None = None,
    input_features: dict[str, Any] | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> int:
    payload = {
        "student_record_id": student_record_id,
        "dataset_name": dataset_name,
        "scenario": scenario,
        "model_name": model_name,
        "model_version": model_version,
        "predicted_class": int(predicted_class),
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "input_features": input_features,
        "run_metadata": run_metadata,
    }
    with engine.begin() as connection:
        result = connection.execute(insert(prediction_results).values(payload))
    return int(result.inserted_primary_key[0])
