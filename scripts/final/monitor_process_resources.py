#!/usr/bin/env python
"""Record lightweight resource samples for a comparator worker PID."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    rows = []
    while True:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    f"$p=Get-Process -Id {args.pid} -ErrorAction SilentlyContinue;"
                    "if($p){\"$($p.CPU),$($p.WorkingSet64),$($p.VirtualMemorySize64)\"}"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        value = result.stdout.strip()
        if not value:
            break
        cpu_seconds, rss_bytes, vms_bytes = value.split(",")
        rows.append(
            {
                "unix_time": time.time(),
                "pid": args.pid,
                "cpu_seconds": float(cpu_seconds),
                "rss_bytes": int(rss_bytes),
                "vms_bytes": int(vms_bytes),
            }
        )
        time.sleep(args.interval)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "unix_time",
                "pid",
                "cpu_seconds",
                "rss_bytes",
                "vms_bytes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
