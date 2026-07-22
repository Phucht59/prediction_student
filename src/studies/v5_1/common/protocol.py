from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[4]


def load_protocol(name: str) -> dict[str, Any]:
    path = ROOT / "configs" / "v5_1" / f"{name.replace('-', '_')}_v5_1.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("protocol_status") != "frozen_before_outer_evaluation":
        raise ValueError(f"V5.1 protocol is not frozen: {path}")
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_hash(name: str) -> str:
    path = ROOT / "configs" / "v5_1" / f"{name.replace('-', '_')}_v5_1.yaml"
    return sha256_file(path)


def verify_source(protocol: dict[str, Any]) -> dict[str, str]:
    source = protocol["source"]
    path = ROOT / source["path"]
    observed = sha256_file(path)
    if observed != source["sha256"]:
        raise RuntimeError(f"Source hash mismatch: {path}")
    return {str(source["path"]): observed}


__all__ = ["ROOT", "load_protocol", "protocol_hash", "sha256_file", "verify_source"]

