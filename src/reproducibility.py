"""Utilities for reproducible raw-data manifests and locked splits.

These functions deliberately preserve the existing modelling pipeline. They only
establish an auditable identity for each raw dataset and create/reuse a single
locked split per dataset and target mode.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_pipeline import process_target_and_stratify


class ReproducibilityError(RuntimeError):
    """Raised when a raw file or stored split cannot be trusted."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_json(value: Any) -> Any:
    """Convert pandas/numpy scalar values into JSON-safe values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _distribution(values: pd.Series) -> dict[str, int]:
    counts = values.value_counts(dropna=False).sort_index()
    return {str(_as_json(label)): int(count) for label, count in counts.items()}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paths(dataset_name: str, target_mode: str, processed_dir: Path, manifests_dir: Path) -> dict[str, Path]:
    prefix = f"{dataset_name}_{target_mode}"
    return {
        "raw_manifest": manifests_dir / f"{dataset_name}_raw_manifest.json",
        "split_manifest": processed_dir / f"{prefix}_split_indices.json",
        "train_csv": processed_dir / f"{prefix}_train_pool.csv",
        "test_csv": processed_dir / f"{prefix}_locked_test.csv",
    }


def build_raw_manifest(
    raw_frame: pd.DataFrame,
    raw_path: Path,
    dataset_name: str,
    target_col: str,
    csv_sep: str,
    prepared_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Create a compact, non-sensitive identity record for a raw CSV."""
    return {
        "dataset": dataset_name,
        "raw_file": raw_path.name,
        "raw_file_sha256": sha256_file(raw_path),
        "raw_file_size_bytes": raw_path.stat().st_size,
        "csv_separator": csv_sep,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(raw_frame)),
        "column_count": int(len(raw_frame.columns)),
        "columns": list(raw_frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in raw_frame.dtypes.items()},
        "missing_values": {column: int(count) for column, count in raw_frame.isna().sum().items()},
        "duplicate_row_count": int(raw_frame.duplicated().sum()),
        "target_column": target_col,
        "raw_target_distribution": _distribution(raw_frame[target_col]),
        "prepared_target_distribution": _distribution(prepared_frame[target_col]),
    }


def _validate_raw_frame(raw_frame: pd.DataFrame, target_col: str) -> None:
    if raw_frame.empty:
        raise ReproducibilityError("Raw dataset is empty.")
    if target_col not in raw_frame.columns:
        raise ReproducibilityError(
            f"Expected target column '{target_col}' was not found in the raw dataset."
        )


def prepare_locked_split(
    *,
    raw_path: Path,
    dataset_name: str,
    target_col: str,
    dataset_kind: str,
    csv_sep: str,
    processed_dir: Path,
    manifests_dir: Path,
    target_mode: str = "3class",
    seed: int = 42,
    test_size: float = 0.2,
    force: bool = False,
    verify_only: bool = False,
) -> dict[str, Any]:
    """Create or verify one deterministic locked split.

    Existing splits are never silently replaced. If the raw file hash changes,
    callers must use ``force=True`` after confirming that a new experiment is
    intended. This prevents accidental comparisons across different datasets.
    """
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing raw file: {raw_path}. Place the verified source CSV in data/raw first."
        )

    raw_frame = pd.read_csv(raw_path, sep=csv_sep)
    _validate_raw_frame(raw_frame, target_col)

    prepared_frame = process_target_and_stratify(
        raw_frame.copy(), target_col, dataset_kind, target_mode
    ).dropna(subset=["_strat_target"])
    if prepared_frame.empty:
        raise ReproducibilityError("No rows remain after target preparation.")

    paths = _paths(dataset_name, target_mode, processed_dir, manifests_dir)
    raw_manifest = build_raw_manifest(
        raw_frame=raw_frame,
        raw_path=raw_path,
        dataset_name=dataset_name,
        target_col=target_col,
        csv_sep=csv_sep,
        prepared_frame=prepared_frame,
    )
    raw_hash = raw_manifest["raw_file_sha256"]

    existing_manifest = _read_json(paths["split_manifest"]) if paths["split_manifest"].exists() else None
    if existing_manifest is not None:
        saved_hash = existing_manifest.get("raw_file_sha256")
        if saved_hash != raw_hash and not force:
            raise ReproducibilityError(
                "The raw file hash differs from the split manifest. Refusing to overwrite "
                "the locked split. Review the data version, then rerun with --force only "
                "when intentionally creating a new experiment."
            )

    if verify_only:
        if existing_manifest is None:
            raise ReproducibilityError("No split manifest exists to verify.")
        if existing_manifest.get("raw_file_sha256") != raw_hash:
            raise ReproducibilityError("Raw file hash does not match the stored split manifest.")
        train_indices = existing_manifest.get("train_row_indices", [])
        test_indices = existing_manifest.get("locked_test_row_indices", [])
        _validate_indices(train_indices, test_indices, len(raw_frame))
        _verify_materialized_files(paths, existing_manifest)
        return {
            "status": "verified",
            "dataset": dataset_name,
            "raw_file_sha256": raw_hash,
            "train_rows": len(train_indices),
            "locked_test_rows": len(test_indices),
            "paths": {name: str(path) for name, path in paths.items()},
        }

    if existing_manifest is not None and not force:
        train_indices = existing_manifest.get("train_row_indices", [])
        test_indices = existing_manifest.get("locked_test_row_indices", [])
        _validate_indices(train_indices, test_indices, len(raw_frame))
        _verify_materialized_files(paths, existing_manifest)
        if not paths["raw_manifest"].exists():
            _write_json(paths["raw_manifest"], raw_manifest)
        return {
            "status": "reused",
            "dataset": dataset_name,
            "raw_file_sha256": raw_hash,
            "train_rows": len(train_indices),
            "locked_test_rows": len(test_indices),
            "paths": {name: str(path) for name, path in paths.items()},
        }

    candidate_indices = prepared_frame.index.to_numpy(dtype=int)
    stratify_target = prepared_frame["_strat_target"].astype(int).to_numpy()
    train_indices, test_indices = train_test_split(
        candidate_indices,
        test_size=test_size,
        stratify=stratify_target,
        random_state=seed,
    )
    _validate_indices(train_indices.tolist(), test_indices.tolist(), len(raw_frame))

    train_frame = prepared_frame.loc[train_indices].drop(columns=["_strat_target"])
    test_frame = prepared_frame.loc[test_indices].drop(columns=["_strat_target"])
    paths["train_csv"].parent.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(paths["train_csv"], index=False)
    test_frame.to_csv(paths["test_csv"], index=False)

    split_manifest = {
        "dataset": dataset_name,
        "target_mode": target_mode,
        "raw_file": raw_path.name,
        "raw_file_sha256": raw_hash,
        "seed": int(seed),
        "locked_test_size": float(test_size),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "train_row_indices": [int(index) for index in train_indices],
        "locked_test_row_indices": [int(index) for index in test_indices],
        "train_rows": int(len(train_frame)),
        "locked_test_rows": int(len(test_frame)),
        "prepared_target_distribution": _distribution(prepared_frame[target_col]),
        "train_target_distribution": _distribution(train_frame[target_col]),
        "locked_test_target_distribution": _distribution(test_frame[target_col]),
        "train_csv_sha256": sha256_file(paths["train_csv"]),
        "locked_test_csv_sha256": sha256_file(paths["test_csv"]),
    }
    _write_json(paths["raw_manifest"], raw_manifest)
    _write_json(paths["split_manifest"], split_manifest)

    return {
        "status": "created" if existing_manifest is None else "recreated_with_force",
        "dataset": dataset_name,
        "raw_file_sha256": raw_hash,
        "train_rows": int(len(train_frame)),
        "locked_test_rows": int(len(test_frame)),
        "paths": {name: str(path) for name, path in paths.items()},
    }


def _validate_indices(train_indices: list[int], test_indices: list[int], raw_row_count: int) -> None:
    train_set = set(train_indices)
    test_set = set(test_indices)
    if not train_indices or not test_indices:
        raise ReproducibilityError("Stored split must contain both train and locked-test rows.")
    if len(train_set) != len(train_indices) or len(test_set) != len(test_indices):
        raise ReproducibilityError("Stored split contains duplicate row indices.")
    if train_set.intersection(test_set):
        raise ReproducibilityError("Stored split contains overlapping train and locked-test rows.")
    all_indices = train_set.union(test_set)
    if any(index < 0 or index >= raw_row_count for index in all_indices):
        raise ReproducibilityError("Stored split contains row indices outside the raw dataset.")


def _verify_materialized_files(paths: dict[str, Path], split_manifest: dict[str, Any]) -> None:
    for key, hash_key in (("train_csv", "train_csv_sha256"), ("test_csv", "locked_test_csv_sha256")):
        path = paths[key]
        if not path.exists():
            raise ReproducibilityError(f"Missing materialized split file: {path}")
        expected_hash = split_manifest.get(hash_key)
        if expected_hash and sha256_file(path) != expected_hash:
            raise ReproducibilityError(
                f"Split file hash mismatch for {path.name}. Refusing to trust modified split data."
            )
