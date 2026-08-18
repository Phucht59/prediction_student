"""Score all eligible OULAD Student State rows with the frozen EBM bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.finalization.bulk import score_states  # noqa: E402
from src.recommendation.finalization.freeze import write_freeze_artifacts  # noqa: E402
from src.recommendation.service import RecommendationService  # noqa: E402
from src.recommendation.weak_supervision.silver import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, default=ROOT / "artifacts/recommendation/states/oulad_student_states.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/recommendation/final")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    service = RecommendationService(ROOT, persist=False)
    states = pd.read_parquet(args.states)
    if args.limit:
        states = states.sort_values("case_id").head(args.limit).copy()
    if set(states["stage"].astype(str)) - {"20pct", "35pct", "50pct", "75pct"}:
        raise ValueError("unexpected stages in Student State artifact")
    scores, plans = score_states(states, service.models)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "oulad_recommendation_scores.parquet"
    plan_path = args.output_dir / "oulad_recommendation_plans.parquet"
    scores.to_parquet(score_path, index=False)
    plans.to_parquet(plan_path, index=False)
    write_freeze_artifacts(ROOT)
    checksums = ROOT / "artifacts/recommendation/final/checksums.sha256"
    checksums.write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in (score_path, plan_path)) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"cases": int(len(plans)), "score_rows": int(len(scores)), "plan_rows": int(len(plans))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
