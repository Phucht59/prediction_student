"""CLI for frozen recommendation inference. No LLM, no training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommendation.service import RecommendationService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-json", type=Path)
    parser.add_argument("--external-enrollment-id")
    parser.add_argument("--stage")
    parser.add_argument("--states", type=Path, default=ROOT / "artifacts/recommendation/states/oulad_student_states.parquet")
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args()
    service = RecommendationService(ROOT, persist=args.persist)
    if args.health:
        print(json.dumps(service.health(), indent=2, sort_keys=True))
        return 0 if service.health()["model_bundle"] == "ok" else 1
    if args.state_json:
        payload = json.loads(args.state_json.read_text(encoding="utf-8"))
    elif args.external_enrollment_id and args.stage:
        frame = pd.read_parquet(args.states)
        match = frame[(frame["enrollment_identity"].astype(str) == args.external_enrollment_id) & (frame["stage"].astype(str) == args.stage)]
        if match.empty:
            raise SystemExit("no Student State row for that enrollment/stage")
        payload = match.iloc[0].to_dict()
    else:
        raise SystemExit("provide --state-json or --external-enrollment-id and --stage")
    print(json.dumps(service.recommend(payload), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
