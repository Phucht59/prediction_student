"""Validate the frozen evidence-policy release without evaluating a neural ranker."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.recommend_hybrid.validate_phase5 import validate_release

if __name__ == "__main__":
    validate_release()
    print("FINAL_EVIDENCE_RECOMMENDER_RELEASE_PASS")
