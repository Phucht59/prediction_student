"""Publish the verified residual authority into its dedicated LFS namespace."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.checkpoint_authority import validate_checkpoint_authority  # noqa: E402

MANIFEST = ROOT / "artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json"
RELEASE_ROOT = ROOT / "artifacts/recommend_hybrid/checkpoints/residual_cnn_bilstm"
RELEASE_MANIFEST = ROOT / "artifacts/recommend_hybrid/RESIDUAL_CHECKPOINT_RELEASE_MANIFEST.json"
RELEASE_CHECKSUMS = ROOT / "artifacts/recommend_hybrid/RESIDUAL_CHECKPOINT_CHECKSUMS.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    authority = validate_checkpoint_authority(ROOT)
    if authority["status"] != "PASS":
        print(json.dumps({"status": "BLOCKED_MISSING_AUTHORITY_CHECKPOINT", "authority": authority}, indent=2))
        return 1
    inventory_path = ROOT / "artifacts/recommend_hybrid/counterfactual/residual_checkpoint_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.is_file() else {}
    if inventory.get("status") != "PASS" or inventory.get("exact_authority_matches") != 30:
        print(json.dumps({"status": "BLOCKED_MISSING_AUTHORITY_CHECKPOINT", "inventory": inventory}, indent=2))
        return 1

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    published: list[dict[str, Any]] = []
    for row in manifest["checkpoints"]:
        source = ROOT / row["provenance"]["source_checkpoint_path"]
        release_dir = "final" if row["usage"] == "EVALUATION_ONLY" else "shared"
        release_relative = Path(
            "artifacts/recommend_hybrid/checkpoints/residual_cnn_bilstm"
        ) / release_dir / source.name
        target = ROOT / release_relative
        source_sha = _sha256(source)
        if source_sha != row["sha256"]:
            raise RuntimeError(f"source SHA changed for {source}: {source_sha} != {row['sha256']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if _sha256(target) != source_sha:
                raise RuntimeError(f"refusing to overwrite mismatched release file: {target}")
        else:
            shutil.copyfile(source, target)
        if _sha256(target) != source_sha:
            raise RuntimeError(f"release SHA mismatch after publication: {target}")
        published.append(
            {
                **row,
                "historical_checkpoint_path": row["provenance"]["source_checkpoint_path"],
                "release_checkpoint_path": str(release_relative).replace("\\", "/"),
                "release_sha256": source_sha,
                "release_lfs_object_id": source_sha,
            }
        )

    release_manifest = {
        "schema_version": "recommend_hybrid_residual_release_manifest_v1",
        "status": "RELEASE_FROZEN",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "authority_manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "authority_architecture_hash": manifest["architecture_hash"],
        "authority_parameter_count": manifest["parameter_count"],
        "checkpoint_count": len(published),
        "stage_fold_seed_mapping_count": sum(len(row["stages"]) for row in published),
        "claim_boundary": "MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT",
        "checkpoints": published,
    }
    manifest_bytes = (json.dumps(release_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    checksums = {
        "schema_version": "recommend_hybrid_residual_release_checksums_v1",
        "status": "RELEASE_FROZEN",
        "architecture_hash": manifest["architecture_hash"],
        "parameter_count": manifest["parameter_count"],
        "manifest_path": str(RELEASE_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": manifest_sha,
        "files": {
            row["release_checkpoint_path"]: row["release_sha256"] for row in published
        },
    }
    RELEASE_MANIFEST.write_bytes(manifest_bytes)
    RELEASE_CHECKSUMS.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "checkpoint_count": len(published), "stage_fold_seed_mapping_count": sum(len(row["stages"]) for row in published)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
