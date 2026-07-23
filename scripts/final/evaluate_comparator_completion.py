#!/usr/bin/env python
"""Evaluate and validate every final comparator prediction."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.final_release.comparator_evaluation import evaluate_all


if __name__ == "__main__":
    result = evaluate_all()
    print(result["status"])
