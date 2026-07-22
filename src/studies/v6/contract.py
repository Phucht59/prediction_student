from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / "configs/v6/integrated_system_protocol.yaml"
ARTIFACT_ROOT = ROOT / "artifacts/v6"
REPORT_ROOT = ROOT / "reports/v6"


def load_protocol() -> dict[str, Any]:
    value = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if value.get("status") != "FROZEN_BEFORE_V6_AUDIT":
        raise RuntimeError("V6 protocol is not frozen")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def git_tree_hash(revision: str, path: str) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", f"{revision}:{path}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def protected_hash_status() -> dict[str, Any]:
    manifest = json.loads(
        (ARTIFACT_ROOT / "protected_baseline_hashes.json").read_text(encoding="utf-8")
    )
    base = str(manifest["base_sha"])
    rows: dict[str, Any] = {}
    for path, expected in manifest["paths"].items():
        base_hash = git_tree_hash(base, path)
        head_hash = git_tree_hash("HEAD", path)
        rows[path] = {
            "expected": expected,
            "base": base_hash,
            "head": head_hash,
            "pass": expected == base_hash == head_hash,
        }
    return {"pass": all(row["pass"] for row in rows.values()), "paths": rows}


__all__ = [
    "ARTIFACT_ROOT",
    "PROTOCOL_PATH",
    "REPORT_ROOT",
    "ROOT",
    "atomic_json",
    "atomic_text",
    "git_tree_hash",
    "load_protocol",
    "protected_hash_status",
    "sha256_file",
]

