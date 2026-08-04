"""Normalize hybrid-only evidence and need onto frozen runtime semantics."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/hybrid_only_final/dataset"
PROTOCOL = yaml.safe_load(
    (ROOT / "configs/recommend_hybrid/hybrid_only_final_protocol.yaml").read_text(
        encoding="utf-8"
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _severity_need(value: float) -> float:
    """Map direct cutoff-safe deficits to the policy severity scale."""

    if value > 0.75:
        return 1.00
    if value > 0.50:
        return 0.75
    if value > 0.25:
        return 0.50
    if value > 0.00:
        return 0.25
    return 0.00


def main() -> None:
    path = OUT / "candidate_rows.parquet"
    frame = pd.read_parquet(path)
    floors = {
        key: float(value)
        for key, value in PROTOCOL["semantic_evidence_floor"].items()
        if key != "basis"
    }
    mapped = frame["runtime_action_id"].map(floors)
    if mapped.isna().any():
        unknown = sorted(frame.loc[mapped.isna(), "runtime_action_id"].unique())
        raise RuntimeError(f"missing semantic evidence floor: {unknown}")

    raw_evidence = pd.to_numeric(
        frame["evidence_strength"], errors="coerce"
    ).fillna(0.0)
    frame["raw_evidence_strength"] = raw_evidence.clip(0.0, 1.0)
    frame["evidence_strength"] = pd.concat(
        [frame["raw_evidence_strength"], mapped], axis=1
    ).max(axis=1).clip(0.0, 1.0)

    raw_need = pd.to_numeric(frame["deficit_score"], errors="coerce").fillna(0.0)
    frame["raw_deficit_score"] = raw_need.clip(lower=0.0)
    frame["deficit_score"] = frame["raw_deficit_score"].map(_severity_need)

    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)

    schema_path = OUT / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["semantic_evidence_floor"] = floors
    schema["semantic_evidence_basis"] = PROTOCOL["semantic_evidence_floor"][
        "basis"
    ]
    schema["raw_evidence_column"] = "raw_evidence_strength"
    schema["raw_need_column"] = "raw_deficit_score"
    schema["need_scale"] = {
        "LOW": 0.25,
        "MEDIUM": 0.50,
        "HIGH": 0.75,
        "CRITICAL": 1.00,
        "basis": "same_ordinal_severity_scale_as_runtime_policy_evidence",
    }
    schema_path.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    files = [
        item
        for item in OUT.iterdir()
        if item.is_file() and item.name != "CHECKSUMS.json"
    ]
    checksums = {
        str(item.relative_to(ROOT)).replace("\\", "/"): _sha256(item)
        for item in files
    }
    (OUT / "CHECKSUMS.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "rows": int(len(frame)),
                "floors": floors,
                "mean_raw_evidence": float(
                    frame["raw_evidence_strength"].mean()
                ),
                "mean_normalized_evidence": float(
                    frame["evidence_strength"].mean()
                ),
                "mean_raw_need": float(frame["raw_deficit_score"].mean()),
                "mean_ordinal_need": float(frame["deficit_score"].mean()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
