from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    command = [sys.executable, "-m", "src.evaluation.summarize_all"]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print("Test/evaluation summary finished.", flush=True)


if __name__ == "__main__":
    main()
