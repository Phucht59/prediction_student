"""Preflight the local authorities required for full counterfactual evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.checkpoint_authority import validate_checkpoint_authority

OUT = ROOT / "artifacts/recommend_hybrid/counterfactual/preflight.json"
RAW_MANIFEST = ROOT / "data/manifests/extension_raw_manifest.json"
CHECKPOINT_MANIFEST = (
    ROOT / "artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json"
)
SPLIT_MANIFEST = (
    ROOT / "data/processed/study_c_oulad/manifests/split_manifest.csv"
)
OOF_PREDICTIONS = (
    ROOT / "artifacts/canonical_v3/predictions/oulad_oof_predictions.parquet"
)
TRAINING_AUTHORITY = ROOT / "artifacts/canonical_v3/oulad_h1_training_authority.json"
RELEASE_STAGE_MAPPING = (
    ROOT / "artifacts/final/unified_stage_aware_oulad/checkpoint_stage_mapping.json"
)
RELEASE_CHECKSUMS = ROOT / "artifacts/final/unified_stage_aware_oulad/checksums.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_check(
    path: Path,
    *,
    expected_sha256: str | None = None,
    verify_hashes: bool,
) -> dict[str, Any]:
    exists = path.is_file()
    actual_hash: str | None = None
    hash_match: bool | None = None
    if exists and verify_hashes and expected_sha256:
        actual_hash = _sha256(path)
        hash_match = actual_hash == expected_sha256
    status = exists and (hash_match is not False)
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_hash,
        "hash_match": hash_match,
        "status": "PASS" if status else "FAIL",
    }


def evaluate(*, verify_hashes: bool) -> dict[str, Any]:
    if not RAW_MANIFEST.is_file():
        raise FileNotFoundError(RAW_MANIFEST)
    if not CHECKPOINT_MANIFEST.is_file():
        raise FileNotFoundError(CHECKPOINT_MANIFEST)

    raw_manifest = json.loads(RAW_MANIFEST.read_text(encoding="utf-8"))
    raw_checks = []
    for row in raw_manifest["files"]:
        if not str(row["logical_dataset"]).startswith("oulad_"):
            continue
        raw_checks.append(
            _file_check(
                ROOT / row["relative_repository_path"],
                expected_sha256=str(row["sha256"]),
                verify_hashes=verify_hashes,
            )
        )

    checkpoint_manifest = json.loads(
        CHECKPOINT_MANIFEST.read_text(encoding="utf-8")
    )
    checkpoint_checks = []
    unique_paths: dict[str, str] = {}
    for row in checkpoint_manifest["checkpoints"]:
        source_path = str(row["provenance"]["source_checkpoint_path"])
        expected = str(row["sha256"])
        existing = unique_paths.get(source_path)
        if existing is not None and existing != expected:
            raise RuntimeError(
                f"checkpoint manifest has conflicting hashes for {source_path}"
            )
        unique_paths[source_path] = expected
    for source_path, expected in sorted(unique_paths.items()):
        checkpoint_checks.append(
            _file_check(
                ROOT / source_path,
                expected_sha256=expected,
                verify_hashes=verify_hashes,
            )
        )

    release_mapping_sha = None
    if RELEASE_CHECKSUMS.is_file():
        checksum_payload = json.loads(RELEASE_CHECKSUMS.read_text(encoding="utf-8"))
        release_mapping_sha = next(
            (
                str(row["sha256"])
                for row in checksum_payload.get("files", [])
                if row.get("path") == str(RELEASE_STAGE_MAPPING.relative_to(ROOT)).replace("\\", "/")
            ),
            None,
        )
    authority_checks = [
        _file_check(
            SPLIT_MANIFEST,
            verify_hashes=False,
        ),
        _file_check(
            OOF_PREDICTIONS,
            verify_hashes=False,
        ),
        _file_check(
            TRAINING_AUTHORITY,
            verify_hashes=False,
        ),
        _file_check(
            RELEASE_STAGE_MAPPING,
            expected_sha256=release_mapping_sha,
            verify_hashes=verify_hashes,
        ),
    ]
    checkpoint_authority = validate_checkpoint_authority(ROOT)
    all_checks = (*raw_checks, *checkpoint_checks, *authority_checks)
    failed = [row["path"] for row in all_checks if row["status"] != "PASS"]
    if checkpoint_authority["status"] != "PASS":
        failed.append("checkpoint_authority_validation")
    payload = {
        "schema_version": "counterfactual_evaluation_preflight_v1",
        "generated_at": _utc_now(),
        "verify_hashes": verify_hashes,
        "raw_oulad": raw_checks,
        "frozen_checkpoints": checkpoint_checks,
        "evaluation_authorities": authority_checks,
        "checkpoint_authority": checkpoint_authority,
        "summary": {
            "raw_file_count": len(raw_checks),
            "checkpoint_file_count": len(checkpoint_checks),
            "authority_file_count": len(authority_checks),
            "failed_count": len(failed),
            "failed_paths": failed,
        },
        "status": "PASS" if not failed else "FAIL",
    }
    _write_json(OUT, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-hashes",
        action="store_true",
        help="Hash raw OULAD and checkpoint files; slower but release-safe.",
    )
    args = parser.parse_args()
    payload = evaluate(verify_hashes=args.verify_hashes)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
