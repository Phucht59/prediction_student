"""Resume-safe overnight continuation. Waits for locks then runs remaining phases."""
from __future__ import annotations

import time
from pathlib import Path

from .cli import cmd_baselines, cmd_diagnose, cmd_optimize, cmd_report
from .paths import RUN_DIR
from .status import write_status


def wait_lock(domain: str, timeout: float = 8 * 3600) -> bool:
    path = RUN_DIR / f"baseline_lock_{domain}.json"
    start = time.time()
    while time.time() - start < timeout:
        if path.exists():
            return True
        time.sleep(30)
    return False


def main() -> int:
    write_status(
        phase="overnight continuation",
        completed=["waiting for UCI baseline lock"],
        evidence=[],
        decision="do not open confirmation until development gate",
        next_step="UCI lock then C3-G screen",
    )
    import argparse

    ns_uci = argparse.Namespace(dataset="uci", candidate="C3-G", resume=True)
    ns_all_uci = argparse.Namespace(dataset="uci", candidate="all", resume=True)
    if not wait_lock("uci"):
        write_status(phase="blocked", completed=[], evidence=[], decision="UCI lock timeout", next_step="inspect Optuna", blockers=["uci baseline lock missing"])
        return 2
    cmd_optimize(argparse.Namespace(dataset="uci", candidate="C3-G", resume=True))
    cmd_optimize(argparse.Namespace(dataset="uci", candidate="C2-S", resume=True))
    cmd_baselines(argparse.Namespace(dataset="oulad", resume=True))
    cmd_diagnose(argparse.Namespace(dataset="oulad", candidate="C0-R"))
    cmd_optimize(argparse.Namespace(dataset="oulad", candidate="C3-G", resume=True))
    cmd_report(argparse.Namespace())
    write_status(
        phase="overnight continuation finished-or-failed-open-report",
        completed=["see runs/"],
        evidence=[str(RUN_DIR)],
        decision="evaluate development gate before any authority promotion",
        next_step="read FINAL_DECISION.md",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
