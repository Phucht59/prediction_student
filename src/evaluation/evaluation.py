"""Evaluation reporting and PostgreSQL source/ML lineage persistence."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

from src.config import (
    DATABASE_URL,
    METRICS_DIR,
    MODELS_DIR,
    POSTGRES_CONFIG,
    REPORTS_DIR,
    ROOT_DIR,
    STUDENT_G3_3CLASS_BINS,
    XAPI_CLASS_MAPPING,
)
from src.data_pipeline import SOURCE_ROW_NUMBER_COLUMN, attach_source_row_numbers, drop_protected_metadata
from src.explainability import CLASS_NAMES
from src.reproducibility import sha256_file
from src.utils import setup_logger

logger = setup_logger("evaluation")

HASH_ALGORITHM = "sha256"
DEFAULT_RECOMMENDATION_POLICY = "mlp_learning_path_v1"


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_ingestion_contract(csv_sep: str, columns: list[str]) -> dict[str, Any]:
    return {
        "source_format": "csv",
        "delimiter": csv_sep,
        "encoding": "utf-8",
        "header_policy": "first_row_header",
        "null_value_policy": "pandas_default",
        "parser": "pandas.read_csv",
        "parser_version": pd.__version__,
        "canonical_columns": list(columns),
        "schema_fingerprint": sha256_json(list(columns)),
    }


def project_uri(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(ROOT_DIR.resolve())
        return f"project://{relative.as_posix()}"
    except ValueError:
        return f"logical://{path.name}"


def _resolve_local_uri(uri: str, *, field_name: str) -> Path:
    path_candidate = Path(uri)
    if path_candidate.drive:
        return path_candidate
    parsed = urlparse(uri)
    if parsed.scheme == "project":
        return ROOT_DIR / unquote(parsed.netloc + parsed.path)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme and parsed.scheme not in {"project", "file"}:
        raise ValueError(f"{field_name} uses unsupported remote URI scheme '{parsed.scheme}'.")
    path = Path(uri)
    return path if path.is_absolute() else ROOT_DIR / path


def validate_local_artifact_uri(
    uri: str,
    *,
    field_name: str,
    expected_hash: str | None = None,
) -> None:
    path = _resolve_local_uri(uri, field_name=field_name)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{field_name} must reference an existing local file: {uri}")
    if expected_hash is not None:
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"{field_name} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}")


def validate_run_artifact_provenance(run: dict[str, Any]) -> None:
    validate_local_artifact_uri(
        run["environment_lock_uri"],
        field_name="environment_lock_uri",
        expected_hash=run["environment_lock_hash"],
    )
    validate_local_artifact_uri(run["artifact_uri"], field_name="artifact_uri")
    if run["working_tree_state"] == "dirty":
        if not run.get("source_diff_uri") or not run.get("source_diff_hash"):
            raise ValueError("Dirty run requires source_diff_uri and source_diff_hash.")
        validate_local_artifact_uri(
            run["source_diff_uri"],
            field_name="source_diff_uri",
            expected_hash=run["source_diff_hash"],
        )
    elif run["source_diff_uri"] is not None or run["source_diff_hash"] is not None:
        raise ValueError("Clean run must not include source diff provenance.")


def build_target_definition(dataset_name: str, target_col: str, target_mode: str, dataset_kind: str) -> dict[str, Any]:
    definition: dict[str, Any] = {
        "task_type": "classification",
        "dataset_code": dataset_name,
        "target_column": target_col,
        "target_mode": target_mode,
        "label_mapping": {str(key): value for key, value in CLASS_NAMES.items()},
    }
    if dataset_kind == "student":
        definition["derivation"] = {
            "type": "pd.cut",
            "bin_edges": list(STUDENT_G3_3CLASS_BINS),
            "labels": [0, 1, 2],
            "include_lowest": True,
        }
    elif dataset_kind == "xapi":
        definition["derivation"] = {
            "type": "categorical_mapping",
            "mapping": dict(XAPI_CLASS_MAPPING),
        }
    return definition


def build_split_manifest(
    *,
    dataset_version_identity: dict[str, Any],
    target_definition: dict[str, Any],
    split_protocol: dict[str, Any],
    raw_frame: pd.DataFrame,
    train_pool: pd.DataFrame,
    locked_test: pd.DataFrame,
) -> dict[str, Any]:
    raw_with_source = attach_source_row_numbers(raw_frame)
    all_rows = set(raw_with_source[SOURCE_ROW_NUMBER_COLUMN].astype(int).tolist())
    train_rows = set(train_pool[SOURCE_ROW_NUMBER_COLUMN].astype(int).tolist())
    test_rows = set(locked_test[SOURCE_ROW_NUMBER_COLUMN].astype(int).tolist())
    validation_rows: set[int] = set()
    eligible_rows = train_rows | test_rows | validation_rows
    excluded_rows = all_rows - eligible_rows

    if train_rows & test_rows:
        raise ValueError("Train and test source row memberships overlap.")
    if train_rows & validation_rows or test_rows & validation_rows:
        raise ValueError("Validation source row memberships overlap another split.")
    if eligible_rows | excluded_rows != all_rows:
        raise ValueError("Split manifest does not cover every source row.")

    def members(rows: set[int]) -> list[dict[str, int]]:
        return [{"source_row_number": int(row)} for row in sorted(rows)]

    manifest = {
        "dataset_version_identity": dataset_version_identity,
        "target_definition_hash": sha256_json(target_definition),
        "target_definition": target_definition,
        "split_protocol": split_protocol,
        "membership": {
            "train": members(train_rows),
            "validation": members(validation_rows),
            "test": members(test_rows),
            "excluded": [
                {
                    "source_row_number": int(row),
                    "exclusion_reason": "excluded_before_split_or_missing_target",
                }
                for row in sorted(excluded_rows)
            ],
        },
        "counts": {
            "total_source_records": int(len(all_rows)),
            "eligible": int(len(eligible_rows)),
            "excluded": int(len(excluded_rows)),
            "train": int(len(train_rows)),
            "validation": int(len(validation_rows)),
            "test": int(len(test_rows)),
        },
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    return manifest


def write_split_manifest(manifest: dict[str, Any], path: Path) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path), sha256_file(path)


def validate_split_manifest(manifest: dict[str, Any]) -> None:
    membership = manifest["membership"]
    train = {item["source_row_number"] for item in membership["train"]}
    validation = {item["source_row_number"] for item in membership["validation"]}
    test = {item["source_row_number"] for item in membership["test"]}
    excluded = {item["source_row_number"] for item in membership["excluded"]}
    if train & validation or train & test or validation & test:
        raise ValueError("Split manifest train/validation/test memberships overlap.")
    if (train | validation | test) & excluded:
        raise ValueError("Split manifest excluded records overlap eligible records.")
    counts = manifest["counts"]
    eligible = train | validation | test
    all_rows = eligible | excluded
    if len(eligible) != counts["eligible"]:
        raise ValueError("Split manifest eligible count mismatch.")
    if len(excluded) != counts["excluded"]:
        raise ValueError("Split manifest excluded count mismatch.")
    if len(all_rows) != counts["total_source_records"]:
        raise ValueError("Split manifest total source count mismatch.")
    if len(test) <= 0:
        raise ValueError("Split manifest must contain at least one test record.")


def prepare_storage_context(
    *,
    dataset_name: str,
    target_mode: str,
    dataset_kind: str,
    target_col: str,
    raw_path: Path,
    csv_sep: str,
    raw_frame: pd.DataFrame,
    train_pool: pd.DataFrame,
    locked_test: pd.DataFrame,
    run_id: str,
    model_name: str,
    train_config: dict[str, Any],
    artifact_uri: str,
    git_commit: str,
    working_tree_state: str,
    source_diff_uri: str | None,
    source_diff_hash: str | None,
    environment_lock_uri: str,
    environment_lock_hash: str,
    split_manifest_path: Path,
    started_at: datetime | None = None,
) -> dict[str, Any]:
    raw_with_source = attach_source_row_numbers(raw_frame)
    raw_without_metadata = drop_protected_metadata(raw_with_source)
    ingestion_contract = build_ingestion_contract(csv_sep, list(raw_without_metadata.columns))
    target_definition = build_target_definition(dataset_name, target_col, target_mode, dataset_kind)
    dataset_version_identity = {
        "dataset_code": dataset_name,
        "hash_algorithm": HASH_ALGORITHM,
        "content_hash": sha256_file(raw_path),
        "ingestion_contract_hash_algorithm": HASH_ALGORITHM,
        "ingestion_contract_hash": sha256_json(ingestion_contract),
    }
    split_manifest = build_split_manifest(
        dataset_version_identity=dataset_version_identity,
        target_definition=target_definition,
        split_protocol={
            "name": "stratified_locked_test",
            "membership_names": ["train", "validation", "test", "excluded"],
            "test_split_alias": "locked_test",
            "random_seed": 42,
        },
        raw_frame=raw_with_source,
        train_pool=train_pool,
        locked_test=locked_test,
    )
    validate_split_manifest(split_manifest)
    split_manifest_uri, split_manifest_hash = write_split_manifest(split_manifest, split_manifest_path)

    source_records = []
    source_row_numbers = raw_with_source[SOURCE_ROW_NUMBER_COLUMN].astype(int).tolist()
    for source_row_number, record in zip(source_row_numbers, raw_without_metadata.to_dict("records"), strict=True):
        source_records.append(
            {
                "source_row_number": int(source_row_number),
                "raw_payload": _json_safe(record),
            }
        )

    split_memberships = []
    for split_name in ("train", "validation", "test"):
        for item in split_manifest["membership"][split_name]:
            split_memberships.append(
                {
                    "source_row_number": int(item["source_row_number"]),
                    "split_name": split_name,
                    "exclusion_reason": None,
                }
            )
    for item in split_manifest["membership"]["excluded"]:
        split_memberships.append(
            {
                "source_row_number": int(item["source_row_number"]),
                "split_name": "excluded",
                "exclusion_reason": item["exclusion_reason"],
            }
        )

    storage_context = {
        "run": {
            "run_id": str(run_id),
            "model_name": model_name,
            "task_type": "classification",
            "target_definition": target_definition,
            "target_definition_hash": sha256_json(target_definition),
            "split_manifest_uri": split_manifest_uri,
            "split_manifest_hash": split_manifest_hash,
            "git_commit": git_commit,
            "working_tree_state": working_tree_state,
            "source_diff_uri": source_diff_uri,
            "source_diff_hash": source_diff_hash,
            "environment_lock_uri": environment_lock_uri,
            "environment_lock_hash": environment_lock_hash,
            "train_config": train_config,
            "artifact_uri": artifact_uri,
            "status": "running",
            "started_at": (started_at or datetime.now(timezone.utc)).isoformat(),
            "metadata": {
                "split_manifest_hash_algorithm": HASH_ALGORITHM,
                "target_definition_hash_algorithm": HASH_ALGORITHM,
                "source_diff_hash_algorithm": HASH_ALGORITHM if source_diff_hash else None,
                "environment_lock_hash_algorithm": HASH_ALGORITHM,
            },
        },
        "dataset_version": {
            **dataset_version_identity,
            "source_locator": project_uri(raw_path),
            "ingestion_contract": ingestion_contract,
            "row_count": int(len(raw_frame)),
            "metadata": {},
        },
        "source_records": source_records,
        "split_memberships": split_memberships,
        "split_manifest": split_manifest,
    }
    validate_run_artifact_provenance(storage_context["run"])
    return storage_context


def _normalise_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql+psycopg2://"):
        return database_url.replace("postgresql+psycopg2://", "postgresql://", 1)
    return database_url


def _connect(postgres_config: dict[str, Any] | None = None):
    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError("psycopg2-binary is required for PostgreSQL persistence.") from exc

    config = dict(postgres_config or POSTGRES_CONFIG)
    database_url = config.pop("database_url", None) or DATABASE_URL
    return psycopg2.connect(_normalise_database_url(database_url)) if database_url else psycopg2.connect(**config)


def _dict_cursor(connection):
    from psycopg2.extras import RealDictCursor

    return connection.cursor(cursor_factory=RealDictCursor)


def _compare_value(expected: Any, actual: Any) -> bool:
    if isinstance(expected, Real) and isinstance(actual, Real) and not isinstance(expected, bool) and not isinstance(actual, bool):
        expected_float = float(expected)
        actual_float = float(actual)
        if math.isnan(expected_float) or math.isnan(actual_float):
            return math.isnan(expected_float) and math.isnan(actual_float)
        return abs(expected_float - actual_float) <= 1e-6
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(str(key) for key in expected) != set(str(key) for key in actual):
            return False
        return all(_compare_value(expected[key], actual.get(key)) for key in expected)
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        return len(expected) == len(actual) and all(
            _compare_value(expected_item, actual_item)
            for expected_item, actual_item in zip(expected, actual, strict=True)
        )
    return canonical_json(expected) == canonical_json(actual)


def _require_same(table: str, key: Any, expected: dict[str, Any], actual: dict[str, Any]) -> None:
    mismatches = []
    for column, value in expected.items():
        if not _compare_value(value, actual.get(column)):
            mismatches.append(column)
    if mismatches:
        raise RuntimeError(f"{table} existing row for {key} differs in immutable columns: {', '.join(mismatches)}")


def _fetch_one(cursor, query: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
    cursor.execute(query, params)
    row = cursor.fetchone()
    return dict(row) if row else None


def _insert_dataset_version(cursor, dataset_version: dict[str, Any]) -> int:
    from psycopg2.extras import Json

    cursor.execute(
        """
        INSERT INTO source_dataset_versions (
            dataset_code, source_locator, hash_algorithm, content_hash,
            ingestion_contract, ingestion_contract_hash_algorithm,
            ingestion_contract_hash, row_count, metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (dataset_code, hash_algorithm, content_hash, ingestion_contract_hash_algorithm, ingestion_contract_hash)
        DO NOTHING
        RETURNING dataset_version_id
        """,
        (
            dataset_version["dataset_code"],
            dataset_version["source_locator"],
            dataset_version["hash_algorithm"],
            dataset_version["content_hash"],
            Json(dataset_version["ingestion_contract"]),
            dataset_version["ingestion_contract_hash_algorithm"],
            dataset_version["ingestion_contract_hash"],
            dataset_version["row_count"],
            Json(dataset_version["metadata"]),
        ),
    )
    inserted = cursor.fetchone()
    if inserted:
        return int(inserted["dataset_version_id"])

    existing = _fetch_one(
        cursor,
        """
        SELECT *
        FROM source_dataset_versions
        WHERE dataset_code = %s
          AND hash_algorithm = %s
          AND content_hash = %s
          AND ingestion_contract_hash_algorithm = %s
          AND ingestion_contract_hash = %s
        """,
        (
            dataset_version["dataset_code"],
            dataset_version["hash_algorithm"],
            dataset_version["content_hash"],
            dataset_version["ingestion_contract_hash_algorithm"],
            dataset_version["ingestion_contract_hash"],
        ),
    )
    if existing is None:
        raise RuntimeError("Failed to read existing source dataset version after conflict.")
    _require_same(
        "source_dataset_versions",
        dataset_version["dataset_code"],
        {
            "dataset_code": dataset_version["dataset_code"],
            "hash_algorithm": dataset_version["hash_algorithm"],
            "content_hash": dataset_version["content_hash"],
            "ingestion_contract": dataset_version["ingestion_contract"],
            "ingestion_contract_hash_algorithm": dataset_version["ingestion_contract_hash_algorithm"],
            "ingestion_contract_hash": dataset_version["ingestion_contract_hash"],
            "row_count": dataset_version["row_count"],
            "metadata": dataset_version["metadata"],
        },
        existing,
    )
    return int(existing["dataset_version_id"])


def _insert_source_records(cursor, dataset_version_id: int, source_records: list[dict[str, Any]]) -> dict[int, int]:
    from psycopg2.extras import Json, execute_values

    cursor.execute(
        "SELECT EXISTS (SELECT 1 FROM ml_experiment_runs WHERE dataset_version_id = %s)",
        (dataset_version_id,),
    )
    dataset_is_sealed = bool(cursor.fetchone()["exists"])
    if not dataset_is_sealed:
        rows = [
            (dataset_version_id, record["source_row_number"], Json(record["raw_payload"]))
            for record in source_records
        ]
        execute_values(
            cursor,
            """
            INSERT INTO source_records (dataset_version_id, source_row_number, raw_payload)
            VALUES %s
            ON CONFLICT (dataset_version_id, source_row_number) DO NOTHING
            """,
            rows,
        )
    cursor.execute(
        """
        SELECT record_id, source_row_number, raw_payload
        FROM source_records
        WHERE dataset_version_id = %s
        ORDER BY source_row_number
        """,
        (dataset_version_id,),
    )
    existing_rows = [dict(row) for row in cursor.fetchall()]
    if len(existing_rows) != len(source_records):
        raise RuntimeError("source_records count does not match dataset row_count after ingest.")

    expected_by_row = {record["source_row_number"]: record["raw_payload"] for record in source_records}
    record_ids_by_row: dict[int, int] = {}
    for row in existing_rows:
        source_row_number = int(row["source_row_number"])
        expected_payload = expected_by_row.get(source_row_number)
        if expected_payload is None:
            raise RuntimeError(f"Unexpected source row number persisted: {source_row_number}")
        if not _compare_value(expected_payload, row["raw_payload"]):
            raise RuntimeError(f"source_records raw_payload differs for source row {source_row_number}")
        record_ids_by_row[source_row_number] = int(row["record_id"])

    if set(record_ids_by_row) != set(expected_by_row):
        raise RuntimeError("source_records row-number range is incomplete after ingest.")
    return record_ids_by_row


def _insert_experiment_run(cursor, dataset_version_id: int, run: dict[str, Any]) -> str:
    from psycopg2.extras import Json

    cursor.execute(
        """
        INSERT INTO ml_experiment_runs (
            run_id, dataset_version_id, model_name, task_type,
            target_definition, target_definition_hash, split_manifest_uri,
            split_manifest_hash, git_commit, working_tree_state,
            source_diff_uri, source_diff_hash, environment_lock_uri,
            environment_lock_hash, train_config, artifact_uri,
            status, started_at, completed_at, metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, %s)
        ON CONFLICT (run_id) DO NOTHING
        """,
        (
            run["run_id"],
            dataset_version_id,
            run["model_name"],
            run["task_type"],
            Json(run["target_definition"]),
            run["target_definition_hash"],
            run["split_manifest_uri"],
            run["split_manifest_hash"],
            run["git_commit"],
            run["working_tree_state"],
            run["source_diff_uri"],
            run["source_diff_hash"],
            run["environment_lock_uri"],
            run["environment_lock_hash"],
            Json(run["train_config"]),
            run["artifact_uri"],
            run["status"],
            run["started_at"],
            Json(run["metadata"]),
        ),
    )
    existing = _fetch_one(cursor, "SELECT * FROM ml_experiment_runs WHERE run_id = %s", (run["run_id"],))
    if existing is None:
        raise RuntimeError(f"Failed to read experiment run {run['run_id']} after insert.")
    _require_same(
        "ml_experiment_runs",
        run["run_id"],
        {
            "run_id": run["run_id"],
            "dataset_version_id": dataset_version_id,
            "model_name": run["model_name"],
            "task_type": run["task_type"],
            "target_definition": run["target_definition"],
            "target_definition_hash": run["target_definition_hash"],
            "split_manifest_uri": run["split_manifest_uri"],
            "split_manifest_hash": run["split_manifest_hash"],
            "git_commit": run["git_commit"],
            "working_tree_state": run["working_tree_state"],
            "source_diff_uri": run["source_diff_uri"],
            "source_diff_hash": run["source_diff_hash"],
            "environment_lock_uri": run["environment_lock_uri"],
            "environment_lock_hash": run["environment_lock_hash"],
            "train_config": run["train_config"],
            "artifact_uri": run["artifact_uri"],
            "metadata": run["metadata"],
        },
        existing,
    )
    if existing["status"] == "failed":
        raise RuntimeError(f"Run {run['run_id']} is failed; retry writes are not allowed.")
    return str(existing["status"])


def _insert_split_memberships(
    cursor,
    *,
    run_id: str,
    dataset_version_id: int,
    record_ids_by_row: dict[int, int],
    split_memberships: list[dict[str, Any]],
    allow_insert: bool = True,
) -> None:
    from psycopg2.extras import execute_values

    rows = [
        (
            run_id,
            dataset_version_id,
            record_ids_by_row[item["source_row_number"]],
            item["split_name"],
            item["exclusion_reason"],
        )
        for item in split_memberships
    ]
    if allow_insert:
        execute_values(
            cursor,
            """
            INSERT INTO ml_run_record_splits (
                run_id, dataset_version_id, record_id, split_name, exclusion_reason
            )
            VALUES %s
            ON CONFLICT (run_id, record_id) DO NOTHING
            """,
            rows,
        )
    cursor.execute(
        """
        SELECT record_id, split_name, exclusion_reason
        FROM ml_run_record_splits
        WHERE run_id = %s
        """,
        (run_id,),
    )
    existing = {int(row["record_id"]): dict(row) for row in cursor.fetchall()}
    expected = {
        record_ids_by_row[item["source_row_number"]]: item
        for item in split_memberships
    }
    if len(existing) != len(expected):
        raise RuntimeError("Run split ledger does not cover every expected source record.")
    for record_id, item in expected.items():
        row = existing.get(record_id)
        if row is None:
            raise RuntimeError(f"Missing split membership for record_id {record_id}")
        _require_same(
            "ml_run_record_splits",
            (run_id, record_id),
            {
                "split_name": item["split_name"],
                "exclusion_reason": item["exclusion_reason"],
            },
            row,
        )


def initialize_experiment_run_in_postgres(
    storage_context: dict[str, Any],
    postgres_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register dataset version, full source records, run row, and split ledger."""
    connection = _connect(postgres_config)
    try:
        with _dict_cursor(connection) as cursor:
            dataset_version_id = _insert_dataset_version(cursor, storage_context["dataset_version"])
            record_ids_by_row = _insert_source_records(cursor, dataset_version_id, storage_context["source_records"])
            run_status = _insert_experiment_run(cursor, dataset_version_id, storage_context["run"])
            _insert_split_memberships(
                cursor,
                run_id=storage_context["run"]["run_id"],
                dataset_version_id=dataset_version_id,
                record_ids_by_row=record_ids_by_row,
                split_memberships=storage_context["split_memberships"],
                allow_insert=run_status == "running",
            )
        connection.commit()
        logger.info("Initialized PostgreSQL source/ML run %s.", storage_context["run"]["run_id"])
        return {"dataset_version_id": dataset_version_id, "record_ids_by_row": record_ids_by_row}
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _validate_probability(probability: dict[str, float], confidence: float, target_definition: dict[str, Any]) -> None:
    labels = set(str(label) for label in target_definition["label_mapping"].values())
    if set(probability) != labels:
        raise ValueError(f"Probability labels {sorted(probability)} do not match target labels {sorted(labels)}.")
    values = [float(value) for value in probability.values()]
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("Probability values must be in [0, 1].")
    if abs(sum(values) - 1.0) > 1e-4:
        raise ValueError("Probability values must sum to 1 within tolerance.")
    if abs(max(values) - float(confidence)) > 1e-4:
        raise ValueError("Confidence must match max probability within tolerance.")


def _run_status(cursor, run_id: str) -> str:
    row = _fetch_one(cursor, "SELECT status FROM ml_experiment_runs WHERE run_id = %s", (run_id,))
    if row is None:
        raise RuntimeError(f"Run {run_id} has not been initialized.")
    return str(row["status"])


def _select_record_ids_for_source_rows(cursor, run_id: str, source_rows: list[int]) -> dict[int, int]:
    cursor.execute(
        """
        SELECT sr.source_row_number, sr.record_id
        FROM ml_experiment_runs r
        JOIN source_records sr
          ON sr.dataset_version_id = r.dataset_version_id
        WHERE r.run_id = %s
          AND sr.source_row_number = ANY(%s)
        """,
        (run_id, source_rows),
    )
    rows = {int(row["source_row_number"]): int(row["record_id"]) for row in cursor.fetchall()}
    if set(rows) != set(source_rows):
        missing = sorted(set(source_rows) - set(rows))
        raise RuntimeError(f"Missing source_records for source rows: {missing[:10]}")
    return rows


def _insert_or_compare_prediction(
    cursor,
    *,
    allow_insert: bool,
    run_id: str,
    record_id: int,
    true_label: int,
    predicted_label: int,
    confidence: float,
    probability: dict[str, float],
) -> int:
    from psycopg2.extras import Json

    if allow_insert:
        cursor.execute(
            """
            INSERT INTO ml_predictions (
                run_id, record_id, split_name, true_label,
                predicted_label, confidence, probability
            )
            VALUES (%s, %s, 'test', %s, %s, %s, %s)
            ON CONFLICT (run_id, record_id, split_name) DO NOTHING
            """,
            (run_id, record_id, true_label, predicted_label, confidence, Json(probability)),
        )
    row = _fetch_one(
        cursor,
        """
        SELECT *
        FROM ml_predictions
        WHERE run_id = %s
          AND record_id = %s
          AND split_name = 'test'
        """,
        (run_id, record_id),
    )
    if row is None:
        raise RuntimeError(f"Missing prediction for run {run_id}, record {record_id}.")
    _require_same(
        "ml_predictions",
        (run_id, record_id, "test"),
        {
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probability": probability,
        },
        row,
    )
    return int(row["prediction_id"])


def _insert_or_compare_metric(
    cursor,
    *,
    allow_insert: bool,
    run_id: str,
    metric_name: str,
    metric_value: float,
    metric_context: dict[str, Any],
) -> None:
    from psycopg2.extras import Json

    if allow_insert:
        cursor.execute(
            """
            INSERT INTO ml_run_metrics (
                run_id, split_name, metric_name, metric_value, label_scope, metric_context
            )
            VALUES (%s, 'test', %s, %s, '__all__', %s)
            ON CONFLICT (run_id, split_name, metric_name, label_scope) DO NOTHING
            """,
            (run_id, metric_name, metric_value, Json(metric_context)),
        )
    row = _fetch_one(
        cursor,
        """
        SELECT *
        FROM ml_run_metrics
        WHERE run_id = %s
          AND split_name = 'test'
          AND metric_name = %s
          AND label_scope = '__all__'
        """,
        (run_id, metric_name),
    )
    if row is None:
        raise RuntimeError(f"Missing metric {metric_name} for run {run_id}.")
    _require_same(
        "ml_run_metrics",
        (run_id, metric_name),
        {
            "metric_value": metric_value,
            "metric_context": metric_context,
        },
        row,
    )


def _insert_or_compare_recommendation(
    cursor,
    *,
    allow_insert: bool,
    prediction_id: int,
    risk_band: str,
    learning_path: Any,
    explanation: dict[str, Any],
    policy_version: str = DEFAULT_RECOMMENDATION_POLICY,
) -> None:
    from psycopg2.extras import Json

    path_payload = learning_path if isinstance(learning_path, dict) else {"steps": learning_path}
    if allow_insert:
        cursor.execute(
            """
            INSERT INTO ml_recommendations (
                prediction_id, policy_version, risk_band, learning_path, explanation
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (prediction_id, policy_version) DO NOTHING
            """,
            (prediction_id, policy_version, risk_band, Json(path_payload), Json(explanation)),
        )
    row = _fetch_one(
        cursor,
        """
        SELECT *
        FROM ml_recommendations
        WHERE prediction_id = %s
          AND policy_version = %s
        """,
        (prediction_id, policy_version),
    )
    if row is None:
        raise RuntimeError(f"Missing recommendation for prediction {prediction_id}.")
    _require_same(
        "ml_recommendations",
        (prediction_id, policy_version),
        {
            "risk_band": risk_band,
            "learning_path": path_payload,
            "explanation": explanation,
        },
        row,
    )


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
    storage_context: dict[str, Any] | None = None,
) -> str:
    """Persist predictions, metrics, recommendations, and complete a source/ML run."""
    if storage_context is None:
        raise ValueError("storage_context is required for source/ML PostgreSQL persistence.")

    row_count = len(original_features)
    arrays = [true_labels, predicted_labels, confidences, probabilities, learning_paths]
    if any(len(values) != row_count for values in arrays):
        raise ValueError("Prediction, confidence, feature and learning-path row counts must match.")
    if SOURCE_ROW_NUMBER_COLUMN not in original_features.columns:
        raise ValueError(f"Missing protected metadata column: {SOURCE_ROW_NUMBER_COLUMN}")

    run_id = storage_context["run"]["run_id"]
    target_definition = storage_context["run"]["target_definition"]
    source_rows = [int(value) for value in original_features[SOURCE_ROW_NUMBER_COLUMN].tolist()]

    connection = _connect(postgres_config)
    try:
        with _dict_cursor(connection) as cursor:
            status = _run_status(cursor, run_id)
            if status == "failed":
                raise RuntimeError(f"Run {run_id} is failed; retry persistence is not allowed.")
            allow_insert = status == "running"
            record_ids_by_source = _select_record_ids_for_source_rows(cursor, run_id, source_rows)

            prediction_ids: list[int] = []
            for row_index, source_row_number in enumerate(source_rows):
                probability = {
                    CLASS_NAMES[class_index]: float(probabilities[row_index][class_index])
                    for class_index in range(probabilities.shape[1])
                }
                _validate_probability(probability, float(confidences[row_index]), target_definition)
                prediction_ids.append(
                    _insert_or_compare_prediction(
                        cursor,
                        allow_insert=allow_insert,
                        run_id=run_id,
                        record_id=record_ids_by_source[source_row_number],
                        true_label=int(true_labels[row_index]),
                        predicted_label=int(predicted_labels[row_index]),
                        confidence=float(confidences[row_index]),
                        probability=probability,
                    )
                )

            for metric_name, metric_value in metrics.items():
                context: dict[str, Any] = {}
                if metric_name in {"RMSE", "R2"}:
                    context = {
                        "metric_type": "ordinal_label_diagnostic",
                        "not_regression_output": True,
                    }
                _insert_or_compare_metric(
                    cursor,
                    allow_insert=allow_insert,
                    run_id=run_id,
                    metric_name=str(metric_name),
                    metric_value=float(metric_value),
                    metric_context=context,
                )

            for row_index, recommendation in learning_paths.reset_index(drop=True).iterrows():
                path_payload = json.loads(recommendation["learning_path"])
                risk_payload = json.loads(recommendation["risk_factors"])
                explanation = {
                    "headline": recommendation["headline"],
                    "risk_factors": risk_payload,
                }
                if "risk_scores" in recommendation:
                    explanation["risk_scores"] = json.loads(recommendation["risk_scores"])
                _insert_or_compare_recommendation(
                    cursor,
                    allow_insert=allow_insert,
                    prediction_id=prediction_ids[row_index],
                    risk_band=str(recommendation["risk_band"]),
                    learning_path=path_payload,
                    explanation=explanation,
                )

            if allow_insert:
                cursor.execute(
                    """
                    UPDATE ml_experiment_runs
                    SET status = 'completed',
                        completed_at = %s
                    WHERE run_id = %s
                    """,
                    (datetime.now(timezone.utc), run_id),
                )
        connection.commit()
        logger.info("Persisted PostgreSQL source/ML run %s for %s.", run_id, dataset_name)
        return str(run_id)
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
        "# Bao cao tong ket mo hinh CNN-BiLSTM + MLP",
        "",
        f"- **Dataset**: {dataset_name}",
        f"- **Bai toan**: {target_mode}",
        "- **Danh gia**: locked test 20%, khong tham gia Optuna",
        "",
        "## Ket qua",
    ]
    for key, value in metrics.items():
        lines.append(f"- **{key}**: {value:.4f}")
    lines.extend(["", "## Sieu tham so"])
    for key, value in best_params.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Khuyen nghi",
            "He thong anh xa cac yeu to rui ro sang lo trinh hoc tap theo tuan.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
