"""Validate the real LFS OULAD checkpoint set against recommender authority."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.checkpoint_authority import validate_checkpoint_authority

OUT = ROOT / "artifacts/recommend_hybrid/counterfactual/checkpoint_authority_validation.json"
REPORT = ROOT / "reports/recommend_hybrid/CHECKPOINT_AUTHORITY_VALIDATION.md"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_report(payload: dict[str, Any]) -> None:
    failed = [gate for gate in payload["gates"] if gate["status"] == "FAIL"]
    lines = [
        "# Checkpoint Authority Validation",
        "",
        f"- Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Status: **{payload['status']}**",
        f"- Release mapping: `{payload['release_mapping']}`",
        f"- Recommendation manifest: `{payload['recommendation_manifest']}`",
        f"- Authority model class: `{payload['authority_model_class']}`",
        f"- Release model class: `{payload['release_model_class']}`",
        f"- Authority parameter count: `{payload['authority_parameter_count']}`",
        f"- Failed gates: `{payload['failed_gate_count']}`",
        "",
        "This is technical checkpoint authority validation only. It does not establish educational, treatment, or causal effectiveness.",
        "",
        "## Gate failures",
        "",
    ]
    if failed:
        lines.extend(
            f"- `{gate.get('name')}` — {gate.get('detail', '')}"
            for gate in failed
        )
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Checkpoint fingerprints",
            "",
            "| Fold | Seed | Stage | Path | SHA-256 | State-dict fingerprint | Preprocessor fingerprint |",
            "|---:|---:|---|---|---|---|---|",
        ]
    )
    for row in payload["checkpoints"]:
        lines.append(
            "| {fold} | {seed} | {stage} | `{path}` | `{sha}` | `{state}` | `{pre}` |".format(
                fold=row["fold"],
                seed=row["seed"],
                stage=row["stage"],
                path=row["checkpoint_path"],
                sha=row["sha256"] or "MISSING",
                state=row["state_dict_fingerprint"] or "MISSING",
                pre=row["preprocessor_fingerprint"] or "MISSING",
            )
        )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    payload = validate_checkpoint_authority(ROOT)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    _write_json(OUT, payload)
    _write_report(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
