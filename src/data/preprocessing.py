"""Leakage guards shared by final preprocessing."""

from __future__ import annotations

from collections.abc import Iterable


def assert_train_only_fit(fit_record_ids: Iterable[str], test_record_ids: Iterable[str]) -> None:
    overlap = set(map(str, fit_record_ids)) & set(map(str, test_record_ids))
    if overlap:
        raise ValueError(f"preprocessor fit/test overlap: {len(overlap)} records")
