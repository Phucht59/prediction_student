from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_module(module: str, args: list[str] | None = None) -> None:
    command = [sys.executable, "-m", module, *(args or [])]
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    run_module("src.data.prepare_datasets", ["--dataset", "all", "--scenario", "all", "--seed", "42"])
    run_module("src.data.prepare_xapi", ["--seed", "42"])
    print("Data preparation finished.")


if __name__ == "__main__":
    main()
