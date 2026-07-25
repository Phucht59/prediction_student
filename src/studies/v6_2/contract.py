from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = ROOT / "artifacts/v6_2_final_validation"
REPORT_ROOT = ROOT / "reports/v6_2"
LOG_ROOT = ARTIFACT_ROOT / "logs"
SCHEMA_VERSION = "v6_2_recommendation_scientific_validation_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as handle:
        handle.write(text.rstrip() + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def frozen_evidence_paths() -> tuple[Path, ...]:
    return (
        ROOT / "artifacts/v5",
        ROOT / "artifacts/v5_1",
        ROOT / "artifacts/v6",
        ROOT / "artifacts/v6_1_oulad_architecture_diagnosis",
        ROOT / "reports/v5",
        ROOT / "reports/v5_1",
        ROOT / "reports/v6",
        ROOT / "reports/v6_1",
    )


__all__ = [
    "ARTIFACT_ROOT",
    "LOG_ROOT",
    "REPORT_ROOT",
    "ROOT",
    "SCHEMA_VERSION",
    "atomic_json",
    "atomic_text",
    "canonical_json",
    "canonical_sha256",
    "frozen_evidence_paths",
    "sha256_file",
]
