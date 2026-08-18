"""Run frozen five-EBM inference and ranking. Never uses Panel B for training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.ranking.explain import compact_explanation  # noqa: E402
from src.recommendation.ranking.router import recommend_frame  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, default=ROOT / "artifacts/recommendation/panels/panel_b.parquet")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/recommendation/models/phase8_model_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/recommendation/inference")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if not args.manifest.exists():
        raise SystemExit("phase8_model_manifest.json is missing; freeze Phase 8 first")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    states = pd.read_parquet(args.states)
    recommendations = recommend_frame(states, manifest, ROOT, top_k=args.top_k)
    ranking_rows = []
    explanation_rows = []
    for rec in recommendations:
        for action in rec["actions"]:
            ranking_rows.append({
                "case_id": rec["case_id"],
                "action_id": action["action_id"],
                "raw_score": action["raw_score"],
                "relevance_score": action["relevance_score"],
                "rank": action["rank"],
                "feasibility_status": action["feasibility_status"],
                "release_status": action["release_status"],
                "quality_warning": action["quality_warning"],
                "in_top_k": action["in_top_k"],
                "plan_status": rec["plan_status"],
                "model_version": action["model_version"],
            })
            explanation_rows.append({"case_id": rec["case_id"], **compact_explanation(action)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "panel_b" if "panel_b" in args.states.name else "inference"
    pd.DataFrame(ranking_rows).to_parquet(args.output_dir / f"{stem}_rankings.parquet", index=False)
    pd.DataFrame(explanation_rows).to_parquet(args.output_dir / f"{stem}_explanations.parquet", index=False)
    print(json.dumps({"cases": len(recommendations), "rows": len(ranking_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
