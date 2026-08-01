"""Deterministic student-group splits for scientific labeling."""
from __future__ import annotations

import hashlib


def split_for_student(student_key: str) -> str:
    bucket = int(hashlib.sha256(student_key.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


__all__ = ["split_for_student"]
