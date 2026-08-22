"""When ablation JSON count hits target or overnight process exits, rewrite chapters."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.kltn_science_fix.paths import RUN
from research.kltn_science_fix.write_final_reports import main as write_reports

TARGET = 153


def main() -> None:
    while True:
        n = len(list(RUN.glob("*.json")))
        print(f"ablation json {n}/{TARGET}", flush=True)
        if n >= TARGET:
            break
        time.sleep(120)
    write_reports()
    print("finalize done", flush=True)


if __name__ == "__main__":
    main()
