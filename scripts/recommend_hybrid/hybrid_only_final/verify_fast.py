"""Run hybrid-only verification against the runtime-aligned fast evaluator."""
from __future__ import annotations

from pathlib import Path

import verify

verify.SCRIPT = Path(__file__).resolve().parent / "tune_and_evaluate_fast.py"

if __name__ == "__main__":
    verify.main()
