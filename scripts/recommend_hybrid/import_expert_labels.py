"""Validate and normalize a completed expert action-rating file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.expert_labels import (
    import_expert_case_reviews,
    import_expert_ratings,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_ratings", type=Path)
    parser.add_argument("case_export", type=Path)
    parser.add_argument("normalized_output", type=Path)
    parser.add_argument("--case-reviews", type=Path)
    parser.add_argument("--normalized-case-reviews", type=Path)
    args = parser.parse_args()
    ratings = import_expert_ratings(
        args.raw_ratings, args.case_export, args.normalized_output
    )
    print(f"NORMALIZED_REAL_EXPERT_RATINGS={len(ratings)}")
    if bool(args.case_reviews) != bool(args.normalized_case_reviews):
        parser.error("case review input and normalized output must be provided together")
    if args.case_reviews:
        reviews = import_expert_case_reviews(
            args.case_reviews, args.case_export, args.normalized_case_reviews
        )
        print(f"NORMALIZED_REAL_EXPERT_CASE_REVIEWS={len(reviews)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
