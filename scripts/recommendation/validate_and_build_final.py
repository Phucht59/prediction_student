"""Safe orchestration for Phase 10-12. Default is validate-only. Never retrains."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(args: list[str]) -> int:
    print(">", " ".join(args))
    return subprocess.call(args, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true", default=True)
    parser.add_argument("--build-inference", action="store_true")
    parser.add_argument("--load-postgres", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    code = _run([sys.executable, "scripts/recommendation/validate_final_freeze.py", "--write"])
    if code:
        return code
    if args.build_inference:
        command = [sys.executable, "scripts/recommendation/run_final_bulk_inference.py"]
        if args.limit:
            command.extend(["--limit", str(args.limit)])
        code = _run(command)
        if code:
            return code
        code = _run([sys.executable, "scripts/recommendation/validate_final_freeze.py", "--write"])
        if code:
            return code
    if args.load_postgres:
        apply = _run([sys.executable, "scripts/database/apply_migrations.py"])
        if apply:
            return apply
        command = [sys.executable, "scripts/recommendation/load_final_results_to_postgres.py"]
        if args.dry_run:
            command.append("--dry-run")
        if args.limit:
            command.extend(["--limit", str(args.limit)])
        return _run(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
