from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
PROJECT_PROTOCOL = ROOT / "configs" / "project_v5_protocol.yaml"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json_yaml(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Protocol root must be an object: {path}")
    return value


def load_project_protocol() -> dict[str, Any]:
    protocol = load_json_yaml(PROJECT_PROTOCOL)
    if protocol.get("protocol_status") != "frozen_before_v5_results":
        raise RuntimeError("V5 protocol is not frozen before results")
    if protocol.get("v4_evidence", {}).get("immutable") is not True:
        raise RuntimeError("V4 immutability is not declared")
    return protocol


def load_study_protocol(study: str) -> dict[str, Any]:
    project = load_project_protocol()
    if study not in project["studies"]:
        raise KeyError(f"Unknown V5 study: {study}")
    return load_json_yaml(ROOT / project["studies"][study])


def verify_declared_sources(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[tuple[str, dict[str, Any]]] = []
    if "source" in protocol:
        sources.append((protocol["dataset"], protocol["source"]))
    for name, source in protocol.get("sources", {}).items():
        sources.append((name, source))
    result = []
    for name, source in sources:
        path = ROOT / source["path"]
        observed = sha256_file(path) if path.is_file() else None
        result.append(
            {
                "source": name,
                "path": source["path"],
                "expected_sha256": source["sha256"],
                "observed_sha256": observed,
                "status": "PASS" if observed == source["sha256"] else "FAIL",
            }
        )
    return result


def protocol_fingerprint(study: str) -> str:
    project = load_project_protocol()
    study_protocol = load_study_protocol(study)
    return canonical_hash({"project": project, "study": study_protocol})


__all__ = [
    "PROJECT_PROTOCOL",
    "ROOT",
    "canonical_hash",
    "load_json_yaml",
    "load_project_protocol",
    "load_study_protocol",
    "protocol_fingerprint",
    "sha256_file",
    "verify_declared_sources",
]

