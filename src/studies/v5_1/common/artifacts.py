from __future__ import annotations

from pathlib import Path

from src.studies.v5.common.artifacts import (
    atomic_write_json,
    result_fingerprint,
    verify_checksum_manifest,
)
from src.studies.v5_1.common.protocol import sha256_file


def build_checksum_manifest(root: str | Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    """Hash portable V5.1 release evidence, excluding resumable local state."""

    directory = Path(root)
    excluded = exclude or {"artifact_checksums.json"}
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.name not in excluded
        and path.name != "optuna.db"
        and "runtime_cache" not in path.parts
        and "ml_models" not in path.parts
        and path.suffix.lower() != ".joblib"
        and not path.name.endswith((".tmp", ".log"))
    }


def safe_v5_1_root(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    normalized = resolved.as_posix().lower()
    if "/v4/" in normalized or "/v5/" in normalized:
        raise RuntimeError("V5.1 output may not enter V4 or V5 namespaces")
    if "/v5_1/" not in normalized and not normalized.endswith("/v5_1"):
        raise RuntimeError(f"Expected an isolated V5.1 output path: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


__all__ = [
    "atomic_write_json",
    "build_checksum_manifest",
    "result_fingerprint",
    "safe_v5_1_root",
    "verify_checksum_manifest",
]
