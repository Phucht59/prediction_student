"""Build learner-stage features and action-expanded query-level evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.explainable_v2.data_builder import (
    build as build_learner_features,
)
from src.recommend_hybrid.explainable_v2.data_builder import (
    write_blocked_manifest,
)
from src.recommend_hybrid.explainable_v2.query_evidence import (
    persist_query_evidence,
)


def build_candidate_action_table(
    learner_features_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    _, candidates, _ = persist_query_evidence(
        learner_features_df,
        root=ROOT,
        candidate_output=output_path,
    )
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "artifacts/recommend_hybrid/explainable_v2/data"
            / "learner_stage_features.parquet"
        ),
    )
    parser.add_argument(
        "--candidates-output",
        type=Path,
        default=(
            ROOT
            / "artifacts/recommend_hybrid/explainable_v2/features"
            / "action_candidates.parquet"
        ),
    )
    parser.add_argument(
        "--lineage",
        type=Path,
        default=(
            ROOT
            / "artifacts/recommend_hybrid/explainable_v2/data"
            / "feature_lineage.parquet"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            ROOT
            / "artifacts/recommend_hybrid/explainable_v2/data"
            / "FEATURE_TABLE_MANIFEST.json"
        ),
    )
    args = parser.parse_args()
    try:
        learner = build_learner_features(
            args.output,
            args.lineage,
            args.manifest,
        )
        candidates = build_candidate_action_table(
            learner,
            args.candidates_output,
        )
        print(f"BUILD_FEATURE_TABLE_SUCCESS=TRUE, ROWS={len(learner)}")
        print(f"ACTION_CANDIDATE_ROWS={len(candidates)}")
        print("QUERY_LEVEL_EVIDENCE_INVARIANT=TRUE")
    except RuntimeError as exc:
        write_blocked_manifest(args.manifest, str(exc))
        print(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
