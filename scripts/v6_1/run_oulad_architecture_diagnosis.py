from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies.v6_1.oulad_runner import (
    architecture_audit,
    recommendation_audit,
    run_all,
    run_final,
    run_inner,
    run_order_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["audit", "inner", "order", "final", "recommendation", "all"],
        default="all",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    actions = {
        "audit": architecture_audit,
        "inner": lambda: run_inner(args.device),
        "order": lambda: run_order_audit(args.device),
        "final": lambda: run_final(args.device),
        "recommendation": recommendation_audit,
        "all": lambda: run_all(args.device),
    }
    print(json.dumps(actions[args.phase](), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
