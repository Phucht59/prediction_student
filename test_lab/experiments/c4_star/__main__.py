from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.c4_star")
    parser.add_argument("command", nargs="?", default="overnight", choices=["overnight", "recover", "tests", "protocol", "status"])
    args = parser.parse_args(argv)
    if args.command == "overnight":
        from .overnight import main as run

        return run()
    if args.command == "recover":
        from .recover import recover, write_recovery_report

        write_recovery_report(recover())
        return 0
    if args.command == "tests":
        import pytest

        return pytest.main(["-q", "tests/research/c4_star", "tests/research/hybrid_superiority_v2"])
    if args.command == "protocol":
        from .overnight import phase_protocol

        phase_protocol()
        return 0
    if args.command == "status":
        from .status import write_status

        write_status(phase="manual", completed=[], decision="status ping", next_step="overnight")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
