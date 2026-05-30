from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


SUMMARY_MODULES = (
    "src.evaluation.summarize_baselines",
    "src.evaluation.summarize_imbalance",
    "src.evaluation.summarize_deep_classification",
    "src.evaluation.summarize_xapi_results",
    "src.evaluation.summarize_final_imbalance_deep",
)


def main() -> int:
    """Run all maintained result-summary scripts from one stable entrypoint."""
    exit_code = 0
    for module in SUMMARY_MODULES:
        command = [sys.executable, "-m", module]
        print("Running:", " ".join(command))
        completed = subprocess.run(command, cwd=PROJECT_ROOT)
        exit_code = max(exit_code, int(completed.returncode))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
