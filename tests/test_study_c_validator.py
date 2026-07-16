from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_validator_uses_positive_no_legacy_access_check():
    text = (ROOT / "scripts" / "run_study_c_oulad.py").read_text(encoding="utf-8")
    assert '"no_legacy_79_access": True' in text
    assert '"legacy_79_accessed": False' not in text
