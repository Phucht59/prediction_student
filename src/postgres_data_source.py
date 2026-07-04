"""PostgreSQL-backed source dataset loading and CSV seed ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd

from src.config import DATASETS, RAW_DIR, ROOT_DIR
from src.data_pipeline import SOURCE_ROW_NUMBER_COLUMN, attach_source_row_numbers, drop_protected_metadata
from src.evaluation.evaluation import (
    HASH_ALGORITHM,
    _connect,
    _dict_cursor,
    _insert_dataset_version,
    _insert_source_records,
    _json_safe,
    build_ingestion_contract,
    project_uri,
    sha256_file,
    sha256_json,
)


def _resolve_local_uri(uri: str) -> Path:
    path_candidate = Path(uri)
    if path_candidate.drive:
        return path_candidate
    parsed = urlparse(uri)
    if parsed.scheme == "project":
        return ROOT_DIR / unquote(parsed.netloc + parsed.path)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    path = Path(uri)
    return path if path.is_absolute() else ROOT_DIR / path


def _source_records_from_frame(raw_frame: pd.DataFrame) -> list[dict[str, Any]]:
    raw_with_source = attach_source_row_numbers(raw_frame)
    raw_without_metadata = drop_protected_metadata(raw_with_source)
    source_rows = raw_with_source[SOURCE_ROW_NUMBER_COLUMN].astype(int).tolist()
    return [
        {
            "source_row_number": int(source_row),
            "raw_payload": _json_safe(record),
        }
        for source_row, record in zip(source_rows, raw_without_metadata.to_dict("records"), strict=True)
    ]


def build_dataset_version_payload(
    *,
    dataset_code: str,
    raw_path: Path,
    csv_sep: str,
    raw_frame: pd.DataFrame,
) -> dict[str, Any]:
    raw_with_source = attach_source_row_numbers(raw_frame)
    raw_without_metadata = drop_protected_metadata(raw_with_source)
    ingestion_contract = build_ingestion_contract(csv_sep, list(raw_without_metadata.columns))
    return {
        "dataset_code": dataset_code,
        "source_locator": project_uri(raw_path),
        "hash_algorithm": HASH_ALGORITHM,
        "content_hash": sha256_file(raw_path),
        "ingestion_contract": ingestion_contract,
        "ingestion_contract_hash_algorithm": HASH_ALGORITHM,
        "ingestion_contract_hash": sha256_json(ingestion_contract),
        "row_count": int(len(raw_with_source)),
        "metadata": {},
    }


def ingest_dataset_csv_to_postgres(
    dataset_code: str,
    *,
    raw_path: Path | None = None,
    csv_sep: str | None = None,
) -> dict[str, Any]:
    """Seed source_dataset_versions/source_records from CSV without creating ML runs."""
    spec = DATASETS[dataset_code]
    resolved_path = raw_path or (RAW_DIR / spec.raw_file)
    resolved_sep = csv_sep or spec.csv_sep
    raw_frame = attach_source_row_numbers(pd.read_csv(resolved_path, sep=resolved_sep))
    dataset_version = build_dataset_version_payload(
        dataset_code=dataset_code,
        raw_path=resolved_path,
        csv_sep=resolved_sep,
        raw_frame=raw_frame,
    )
    source_records = _source_records_from_frame(raw_frame)

    connection = _connect()
    try:
        with _dict_cursor(connection) as cursor:
            dataset_version_id = _insert_dataset_version(cursor, dataset_version)
            record_ids_by_row = _insert_source_records(cursor, dataset_version_id, source_records)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    expected_rows = set(range(dataset_version["row_count"]))
    if set(record_ids_by_row) != expected_rows:
        raise RuntimeError("source_records row-number range is incomplete after CSV ingest.")
    return {
        "dataset_version_id": dataset_version_id,
        "dataset_version": dataset_version,
        "row_count": dataset_version["row_count"],
        "source_record_count": len(record_ids_by_row),
    }


def load_dataset_version_from_postgres(
    dataset_code: str,
    dataset_version_id: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load source_records as the raw training DataFrame for a dataset version."""
    connection = _connect()
    try:
        with _dict_cursor(connection) as cursor:
            if dataset_version_id is None:
                cursor.execute(
                    """
                    SELECT *
                    FROM source_dataset_versions
                    WHERE dataset_code = %s
                    ORDER BY created_at DESC, dataset_version_id DESC
                    LIMIT 1
                    """,
                    (dataset_code,),
                )
            else:
                cursor.execute(
                    """
                    SELECT *
                    FROM source_dataset_versions
                    WHERE dataset_code = %s
                      AND dataset_version_id = %s
                    """,
                    (dataset_code, dataset_version_id),
                )
            version = cursor.fetchone()
            if version is None:
                suffix = f" id={dataset_version_id}" if dataset_version_id is not None else ""
                raise RuntimeError(f"dataset version not found for dataset_code={dataset_code}{suffix}")
            version = dict(version)

            cursor.execute(
                """
                SELECT source_row_number, raw_payload
                FROM source_records
                WHERE dataset_version_id = %s
                ORDER BY source_row_number
                """,
                (version["dataset_version_id"],),
            )
            rows = [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()

    row_count = int(version["row_count"])
    if len(rows) != row_count:
        raise RuntimeError("source_records count does not match source_dataset_versions.row_count.")

    source_row_numbers = [int(row["source_row_number"]) for row in rows]
    if source_row_numbers != list(range(row_count)):
        raise RuntimeError("source_row_number range is not contiguous 0..row_count-1.")

    ingestion_contract = version["ingestion_contract"]
    canonical_columns = list(ingestion_contract.get("canonical_columns", []))
    if not canonical_columns:
        raise RuntimeError("ingestion_contract.canonical_columns is missing.")

    records = []
    for row in rows:
        payload = row["raw_payload"]
        if not isinstance(payload, dict):
            raise RuntimeError(f"raw_payload for source row {row['source_row_number']} is not an object.")
        if list(payload.keys()) != canonical_columns:
            if set(payload) != set(canonical_columns):
                raise RuntimeError("raw_payload columns do not match ingestion_contract.canonical_columns.")
        records.append({column: payload.get(column) for column in canonical_columns})

    frame = pd.DataFrame(records, columns=canonical_columns)
    frame.insert(0, SOURCE_ROW_NUMBER_COLUMN, source_row_numbers)
    metadata = {
        "dataset_version_id": int(version["dataset_version_id"]),
        "dataset_code": version["dataset_code"],
        "source_locator": version["source_locator"],
        "hash_algorithm": version["hash_algorithm"],
        "content_hash": version["content_hash"],
        "ingestion_contract": ingestion_contract,
        "ingestion_contract_hash_algorithm": version["ingestion_contract_hash_algorithm"],
        "ingestion_contract_hash": version["ingestion_contract_hash"],
        "row_count": row_count,
        "metadata": version["metadata"],
        "created_at": version["created_at"],
    }
    return frame, metadata


def source_locator_path(dataset_version: dict[str, Any]) -> Path:
    return _resolve_local_uri(str(dataset_version["source_locator"]))


def split_manifest_path(run: dict[str, Any]) -> Path:
    return _resolve_local_uri(str(run["split_manifest_uri"]))


def load_experiment_run(run_id: str) -> dict[str, Any] | None:
    connection = _connect()
    try:
        with _dict_cursor(connection) as cursor:
            cursor.execute("SELECT * FROM ml_experiment_runs WHERE run_id = %s", (run_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    finally:
        connection.close()


def load_run_split_membership(run_id: str) -> list[dict[str, Any]]:
    connection = _connect()
    try:
        with _dict_cursor(connection) as cursor:
            cursor.execute(
                """
                SELECT sr.source_row_number, s.split_name, s.exclusion_reason
                FROM ml_run_record_splits s
                JOIN source_records sr
                  ON sr.dataset_version_id = s.dataset_version_id
                 AND sr.record_id = s.record_id
                WHERE s.run_id = %s
                ORDER BY sr.source_row_number
                """,
                (run_id,),
            )
            return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def reconstruct_splits_from_run(
    raw_frame: pd.DataFrame,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = load_run_split_membership(run_id)
    if not membership:
        raise RuntimeError(f"run {run_id} has no split membership in ml_run_record_splits.")
    train_rows = {
        int(row["source_row_number"])
        for row in membership
        if row["split_name"] == "train"
    }
    test_rows = {
        int(row["source_row_number"])
        for row in membership
        if row["split_name"] == "test"
    }
    if not test_rows:
        raise RuntimeError(f"run {run_id} has no test split membership.")
    frame = attach_source_row_numbers(raw_frame)
    train_pool = frame[frame[SOURCE_ROW_NUMBER_COLUMN].isin(train_rows)].copy()
    locked_test = frame[frame[SOURCE_ROW_NUMBER_COLUMN].isin(test_rows)].copy()
    if len(train_pool) != len(train_rows) or len(locked_test) != len(test_rows):
        raise RuntimeError(f"run {run_id} split membership does not match loaded source records.")
    return train_pool, locked_test


def verify_run_split_manifest(run: dict[str, Any]) -> None:
    path = split_manifest_path(run)
    if not path.exists():
        raise FileNotFoundError(f"split manifest artifact does not exist: {run['split_manifest_uri']}")
    actual_hash = sha256_file(path)
    if actual_hash != run["split_manifest_hash"]:
        raise RuntimeError(
            f"split manifest hash mismatch for run {run['run_id']}: "
            f"expected {run['split_manifest_hash']}, got {actual_hash}"
        )
