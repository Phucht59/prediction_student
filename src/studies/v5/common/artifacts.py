from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .protocol import canonical_hash, sha256_file


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, destination)


def safe_v5_root(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    normalized = resolved.as_posix().lower()
    if "/v4/" in normalized or normalized.endswith("/v4"):
        raise RuntimeError("V5 output may not enter a V4 namespace")
    if "/v5/" not in normalized and not normalized.endswith("/v5"):
        raise RuntimeError(f"Expected an isolated V5 output path: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def build_checksum_manifest(root: str | Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    directory = Path(root)
    excluded = exclude or {"artifact_checksums.json"}
    manifest = {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.name not in excluded
        and path.name != "optuna.db"
        and "runtime_cache" not in path.parts
        and not path.name.endswith(".tmp")
    }
    return manifest


def verify_checksum_manifest(root: str | Path, manifest: dict[str, str]) -> bool:
    directory = Path(root)
    return all((directory / name).is_file() and sha256_file(directory / name) == digest for name, digest in manifest.items())


def result_fingerprint(*, protocol_hash: str, source_hashes: dict[str, str], config: dict[str, Any]) -> str:
    return canonical_hash({"protocol": protocol_hash, "sources": source_hashes, "config": config})


__all__ = [
    "atomic_write_json",
    "build_checksum_manifest",
    "result_fingerprint",
    "safe_v5_root",
    "verify_checksum_manifest",
]
