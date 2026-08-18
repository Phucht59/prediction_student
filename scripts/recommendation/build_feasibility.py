"""Build and validate action feasibility without assigning relevance labels."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.feasibility import build_feasibility_frame, validate_feasibility


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    state_path = ROOT / "artifacts/recommendation/states/oulad_student_states.parquet"
    out = ROOT / "artifacts/recommendation/feasibility/oulad_action_feasibility.parquet"
    state = pd.read_parquet(state_path)
    feasibility = build_feasibility_frame(state)
    errors = validate_feasibility(feasibility, state)
    if errors:
        raise ValueError(f"feasibility validation failed: {errors[:10]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    feasibility.to_parquet(out, index=False)
    distribution = feasibility.groupby(["stage", "action_id", "feasibility_status"]).size().astype(int).to_dict()
    report = [
        "# Feasibility Validation", "", "| Gate | Result |", "|---|---|",
        "| Five actions per case | PASS |", "| Status domain | PASS |",
        "| Deterministic reason/source | PASS |", "| No future/forbidden source columns | PASS |",
        "| Relevance-feasibility separation | PASS | rules use only missing_assessments, vle_available, quiz_activity and system contract |",
        "", "## Distribution", "", "| Action | Stage | Status | Rows |", "|---|---|---|---:|",
    ]
    dist_frame = feasibility.groupby(["action_id", "stage", "feasibility_status"]).size().reset_index(name="rows")
    report.extend(f"| {r.action_id} | {r.stage} | {r.feasibility_status} | {r.rows} |" for r in dist_frame.itertuples())
    report.extend(["", f"Artifact SHA-256: `{sha256(out)}`", f"State source SHA-256: `{sha256(state_path)}`"])
    report_path = ROOT / "reports/recommendation/FEASIBILITY_VALIDATION.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(feasibility), "distribution": {str(k): int(v) for k, v in distribution.items()}, "sha256": sha256(out)}, indent=2))


if __name__ == "__main__":
    main()
