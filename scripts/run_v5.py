from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, choices=["student-mat", "student-por", "oulad", "joint-uci"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.study == "student-mat":
        from src.studies.v5.student_mat import run
    elif args.study == "student-por":
        from src.studies.v5.student_por import run
    elif args.study == "oulad":
        from src.studies.v5.oulad import run
    else:
        from src.studies.v5.common.joint_learning import run_joint_uci as run
    result = run(device=args.device, force=args.force)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result["status"] in {"COMPLETE", "SKIPPED_VALID_CACHE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
