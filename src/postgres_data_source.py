"""PostgreSQL-backed source dataset loading and CSV seed ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd

from src.config import DATASETS, RAW_DIR, ROOT_DIR, STUDENT_G3_3CLASS_BINS, XAPI_CLASS_MAPPING
from src.data_pipeline import SOURCE_ROW_NUMBER_COLUMN, attach_source_row_numbers, drop_protected_metadata
from src.ingestion.csv_reader import read_csv
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


def _target_rows_from_frame(raw_frame: pd.DataFrame, dataset_code: str) -> list[dict[str, Any]]:
    """Build immutable DB target rows separately from source feature payloads."""
    spec = DATASETS[dataset_code]
    frame = attach_source_row_numbers(raw_frame)
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        raw_value = record.get(spec.target_col)
        if pd.isna(raw_value):
            raise ValueError(f"Missing target {spec.target_col} at source row {record[SOURCE_ROW_NUMBER_COLUMN]}")
        if spec.kind == "student":
            numeric = float(raw_value)
            if numeric <= STUDENT_G3_3CLASS_BINS[1]:
                encoded = 0
            elif numeric <= STUDENT_G3_3CLASS_BINS[2]:
                encoded = 1
            else:
                encoded = 2
        else:
            normalized = str(raw_value).strip().upper()
            if normalized not in XAPI_CLASS_MAPPING:
                raise ValueError(f"Unknown xAPI target {raw_value!r}")
            encoded = int(XAPI_CLASS_MAPPING[normalized])
        target_contract = {
            "dataset_code": dataset_code,
            "target_column": spec.target_col,
            "target_mode": "3class",
            "bins": list(STUDENT_G3_3CLASS_BINS) if spec.kind == "student" else None,
            "mapping": dict(XAPI_CLASS_MAPPING) if spec.kind == "xapi" else None,
        }
        rows.append({
            "source_row_number": int(record[SOURCE_ROW_NUMBER_COLUMN]),
            "target_name": spec.target_col,
            "raw_target_value": _json_safe(raw_value),
            "encoded_target_value": encoded,
            "target_contract_hash": sha256_json(target_contract),
        })
    return rows


def _insert_source_record_targets(cursor, dataset_version_id: int, record_ids_by_row: dict[int, int], target_rows: list[dict[str, Any]]) -> None:
    from psycopg2.extras import Json, execute_values

    values = [
        (
            dataset_version_id,
            record_ids_by_row[row["source_row_number"]],
            row["target_name"],
            Json(row["raw_target_value"]),
            row["encoded_target_value"],
            row["target_contract_hash"],
        )
        for row in target_rows
    ]
    execute_values(
        cursor,
        """
        INSERT INTO source_record_targets (
            dataset_version_id, record_id, target_name,
            raw_target_value, encoded_target_value, target_contract_hash
        ) VALUES %s
        ON CONFLICT (dataset_version_id, record_id, target_name) DO NOTHING
        """,
        values,
    )
    cursor.execute(
        """
        SELECT sr.source_row_number, t.target_name, t.encoded_target_value
        FROM source_record_targets t
        JOIN source_records sr
          ON sr.dataset_version_id = t.dataset_version_id
         AND sr.record_id = t.record_id
        WHERE t.dataset_version_id = %s
        ORDER BY sr.source_row_number
        """,
        (dataset_version_id,),
    )
    persisted = cursor.fetchall()
    if len(persisted) != len(target_rows):
        raise RuntimeError("source_record_targets count does not match dataset row_count after ingest.")


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
    raw_frame = attach_source_row_numbers(read_csv(resolved_path, sep=resolved_sep))
    dataset_version = build_dataset_version_payload(
        dataset_code=dataset_code,
        raw_path=resolved_path,
        csv_sep=resolved_sep,
        raw_frame=raw_frame,
    )
    source_records = _source_records_from_frame(raw_frame)
    target_rows = _target_rows_from_frame(raw_frame, dataset_code)

    connection = _connect()
    try:
        with _dict_cursor(connection) as cursor:
            dataset_version_id = _insert_dataset_version(cursor, dataset_version)
            record_ids_by_row = _insert_source_records(cursor, dataset_version_id, source_records)
            _insert_source_record_targets(cursor, dataset_version_id, record_ids_by_row, target_rows)
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
    *,
    include_target: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a dataset version from PostgreSQL, joining labels from target storage.

    ``source_records.raw_payload`` is retained for lineage compatibility.  The
    target table is authoritative when present; callers that only need model
    features can pass ``include_target=False`` and receive no target column.
    """
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
            target_rows: list[dict[str, Any]] = []
            if include_target:
                cursor.execute(
                    """
                    SELECT sr.source_row_number, t.target_name,
                           t.raw_target_value, t.encoded_target_value,
                           t.target_contract_hash
                    FROM source_record_targets t
                    JOIN source_records sr
                      ON sr.dataset_version_id = t.dataset_version_id
                     AND sr.record_id = t.record_id
                    WHERE t.dataset_version_id = %s
                    ORDER BY sr.source_row_number
                    """,
                    (version["dataset_version_id"],),
                )
                target_rows = [dict(row) for row in cursor.fetchall()]
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
    if target_rows:
        by_row = {int(row["source_row_number"]): row for row in target_rows}
        if len(by_row) != row_count:
            raise RuntimeError("source_record_targets does not cover every source record.")
        target_name = str(target_rows[0]["target_name"])
        if include_target:
            frame[target_name] = [by_row[row]["raw_target_value"] for row in source_row_numbers]
        else:
            frame = frame.drop(columns=[target_name], errors="ignore")
    elif not include_target:
        target_name = DATASETS[dataset_code].target_col
        frame = frame.drop(columns=[target_name, "G3_raw"], errors="ignore")
    else:
        raise RuntimeError(
            "source_record_targets is required for training/evaluation. Apply migration 003 and backfill targets; final paths do not fall back to raw_payload."
        )
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


def load_development_subset_from_postgres(
    dataset_code: str,
    dataset_version_id: int,
    source_row_numbers: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only pre-approved development rows in a read-only transaction.

    Phase A-B must not fetch the observed legacy holdout and filter it later in
    memory.  This loader places the immutable development allowlist directly in
    the SQL predicate and verifies exact record coverage before returning.
    """

    requested = sorted({int(value) for value in source_row_numbers})
    if not requested or len(requested) != len(source_row_numbers):
        raise ValueError("Development source_row_numbers must be non-empty and unique.")
    connection = _connect()
    try:
        connection.set_session(readonly=True, autocommit=False)
        with _dict_cursor(connection) as cursor:
            cursor.execute(
                """
                SELECT *
                FROM source_dataset_versions
                WHERE dataset_code = %s
                  AND dataset_version_id = %s
                """,
                (dataset_code, dataset_version_id),
            )
            version_row = cursor.fetchone()
            if version_row is None:
                raise RuntimeError(
                    f"dataset version not found for dataset_code={dataset_code} id={dataset_version_id}"
                )
            version = dict(version_row)
            cursor.execute(
                """
                SELECT sr.source_row_number,
                       sr.raw_payload,
                       t.target_name,
                       t.raw_target_value,
                       t.encoded_target_value,
                       t.target_contract_hash
                FROM source_records sr
                JOIN source_record_targets t
                  ON t.dataset_version_id = sr.dataset_version_id
                 AND t.record_id = sr.record_id
                WHERE sr.dataset_version_id = %s
                  AND sr.source_row_number = ANY(%s)
                ORDER BY sr.source_row_number
                """,
                (dataset_version_id, requested),
            )
            rows = [dict(row) for row in cursor.fetchall()]
        connection.rollback()
    finally:
        connection.close()

    returned = [int(row["source_row_number"]) for row in rows]
    if returned != requested:
        missing = sorted(set(requested) - set(returned))
        unexpected = sorted(set(returned) - set(requested))
        raise RuntimeError(
            f"Development-only DB load does not match the immutable allowlist; missing={missing}, unexpected={unexpected}"
        )
    canonical_columns = list(version["ingestion_contract"].get("canonical_columns", []))
    if not canonical_columns:
        raise RuntimeError("ingestion_contract.canonical_columns is missing.")
    target_name = str(rows[0]["target_name"])
    feature_columns = [column for column in canonical_columns if column != target_name]
    records: list[dict[str, Any]] = []
    for row in rows:
        payload = row["raw_payload"]
        if not isinstance(payload, dict):
            raise RuntimeError(f"raw_payload for source row {row['source_row_number']} is not an object.")
        record = {column: payload.get(column) for column in feature_columns}
        # The authoritative target comes from source_record_targets, never from
        # the raw feature payload retained for historical lineage compatibility.
        record[target_name] = row["raw_target_value"]
        records.append(record)
    frame = pd.DataFrame(records, columns=feature_columns + [target_name])
    frame.insert(0, SOURCE_ROW_NUMBER_COLUMN, returned)
    target_hashes = sorted({str(row["target_contract_hash"]) for row in rows})
    if len(target_hashes) != 1:
        raise RuntimeError("Development records do not share one immutable target contract hash.")
    metadata = {
        "dataset_version_id": int(version["dataset_version_id"]),
        "dataset_code": str(version["dataset_code"]),
        "source_locator": version["source_locator"],
        "hash_algorithm": str(version["hash_algorithm"]),
        "content_hash": str(version["content_hash"]),
        "ingestion_contract": version["ingestion_contract"],
        "ingestion_contract_hash_algorithm": str(version["ingestion_contract_hash_algorithm"]),
        "ingestion_contract_hash": str(version["ingestion_contract_hash"]),
        "dataset_row_count": int(version["row_count"]),
        "loaded_development_row_count": len(frame),
        "loaded_source_row_numbers": returned,
        "target_contract_hash": target_hashes[0],
        "transaction_read_only": True,
    }
    return frame, metadata


def load_dataset_version(
    dataset_version_id: int,
    *,
    include_target: bool,
    target_mode: str = "3class",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stable DB-native loader API used by selection, training, and inference."""
    if target_mode != "3class":
        raise ValueError("Only target_mode='3class' is supported by the frozen project protocol.")
    connection = _connect()
    try:
        with _dict_cursor(connection) as cursor:
            cursor.execute(
                "SELECT dataset_code FROM source_dataset_versions WHERE dataset_version_id = %s",
                (dataset_version_id,),
            )
            row = cursor.fetchone()
    finally:
        connection.close()
    if row is None:
        raise RuntimeError(f"dataset version not found: {dataset_version_id}")
    return load_dataset_version_from_postgres(
        str(row["dataset_code"]), dataset_version_id, include_target=include_target
    )


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
