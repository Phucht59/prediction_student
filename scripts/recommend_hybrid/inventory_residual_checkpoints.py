"""Inventory residual and base checkpoint candidates without modifying them."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARCHITECTURE_HASH = "df5cd885b96e5cea4b840bfc5ca59c08c095f5887df8dd8dcef738edfe8bf70e"
EXPECTED_PARAMETER_COUNT = 160492
OUT_JSON = ROOT / "artifacts/recommend_hybrid/counterfactual/residual_checkpoint_inventory.json"
OUT_CSV = ROOT / "artifacts/recommend_hybrid/counterfactual/residual_checkpoint_inventory.csv"
OUT_REPORT = ROOT / "reports/recommend_hybrid/RESIDUAL_CHECKPOINT_RECOVERY.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lfs_pointer(path: Path) -> bool:
    return path.read_bytes()[:160].startswith(b"version https://git-lfs.github.com/spec/v1")


def _parameter_count(payload: Any) -> int | None:
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        return None
    return int(sum(value.numel() for value in state.values() if isinstance(value, torch.Tensor)))


def main() -> int:
    manifest = json.loads((ROOT / "artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json").read_text(encoding="utf-8"))
    expected = {
        str(row["provenance"]["source_checkpoint_path"]): row
        for row in manifest["checkpoints"]
    }
    roots = [
        ROOT / "artifacts/canonical_v3/checkpoints/oulad_h1_shared",
        ROOT / "artifacts/canonical_v3/checkpoints/oulad_h1_final",
        ROOT / "artifacts/final/h1_final/runtime/checkpoints/H1_TABULAR_RESIDUAL_EXPERT",
        ROOT / "artifacts/final/unified_stage_aware_oulad/checkpoints/cnn_bilstm_oulad",
    ]
    rows: list[dict[str, Any]] = []
    for directory in roots:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.pt")):
            relative = str(path.relative_to(ROOT)).replace("\\", "/")
            record: dict[str, Any] = {
                "path": relative,
                "sha256": None,
                "classification": "CORRUPTED",
                "expected_checkpoint_id": None,
                "fold": None,
                "seed": None,
                "parameter_count": None,
                "architecture_hash": None,
                "candidate": None,
                "load_error": None,
            }
            if _lfs_pointer(path):
                record["classification"] = "LFS_POINTER_ONLY"
                rows.append(record)
                continue
            record["sha256"] = _sha256(path)
            expected_row = expected.get(relative)
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                record["parameter_count"] = _parameter_count(payload)
                record["architecture_hash"] = payload.get("architecture_hash")
                record["candidate"] = payload.get("candidate")
                if isinstance(payload, dict):
                    record["fold"] = payload.get("outer_fold")
                    record["seed"] = payload.get("seed")
                if record["parameter_count"] == 150202:
                    record["classification"] = "BASE_MODEL_150202"
                elif expected_row is not None and record["sha256"] == expected_row["sha256"]:
                    record["classification"] = "EXACT_AUTHORITY_MATCH"
                    record["expected_checkpoint_id"] = expected_row["checkpoint_id"]
                    record["fold"] = expected_row["outer_fold"]
                    record["seed"] = expected_row["seed"]
                elif (
                    record["architecture_hash"] == ARCHITECTURE_HASH
                    and record["parameter_count"] == EXPECTED_PARAMETER_COUNT
                ):
                    record["classification"] = "STRUCTURAL_MATCH_SHA_MISMATCH"
            except Exception as exc:  # inventory must retain corrupted candidates
                record["load_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(record)

    expected_paths = set(expected)
    exact = [row for row in rows if row["classification"] == "EXACT_AUTHORITY_MATCH"]
    missing = sorted(expected_paths - {row["path"] for row in exact})
    payload = {
        "schema_version": "residual_checkpoint_inventory_v1",
        "authority_architecture_hash": ARCHITECTURE_HASH,
        "authority_parameter_count": EXPECTED_PARAMETER_COUNT,
        "expected_authority_checkpoints": len(expected),
        "exact_authority_matches": len(exact),
        "missing_authority_checkpoints": missing,
        "status": "PASS" if len(exact) == len(expected) and not missing else "BLOCKED_MISSING_AUTHORITY_CHECKPOINT",
        "classification_counts": {
            value: sum(row["classification"] == value for row in rows)
            for value in sorted({row["classification"] for row in rows})
        },
        "candidates": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        fields = list(rows[0]) if rows else ["path", "classification"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Residual Checkpoint Recovery",
        "",
        f"- Status: **{payload['status']}**",
        f"- Expected exact residual checkpoints: `{len(expected)}`",
        f"- Exact authority matches: `{len(exact)}`",
        f"- Architecture hash: `{ARCHITECTURE_HASH}`",
        f"- Parameter count: `{EXPECTED_PARAMETER_COUNT}`",
        "- Historical residual sources: `artifacts/canonical_v3/checkpoints/oulad_h1_shared` and `artifacts/canonical_v3/checkpoints/oulad_h1_final`",
        "- Prohibited base source: `artifacts/final/unified_stage_aware_oulad/checkpoints/cnn_bilstm_oulad`",
        "",
        "| Classification | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| `{key}` | {value} |" for key, value in sorted(payload["classification_counts"].items()))
    lines += ["", "The inventory is technical authority evidence only; no causal effect is inferred.", ""]
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "expected_authority_checkpoints", "exact_authority_matches", "missing_authority_checkpoints", "classification_counts")}, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
