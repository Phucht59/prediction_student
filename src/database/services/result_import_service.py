from __future__ import annotations

from pathlib import Path


class ResultImportService:
    """Validate canonical source presence before the migration CLI imports it."""

    REQUIRED = (
        "artifacts/final/final_results.json",
        "artifacts/final/model_registry.json",
        "artifacts/v6/prediction/risk_profiles.parquet",
        "artifacts/v6/recommendation/plans.jsonl",
    )

    def __init__(self, root: Path):
        self.root = root

    def missing_sources(self) -> list[str]:
        return [relative for relative in self.REQUIRED if not (self.root / relative).is_file()]


__all__ = ["ResultImportService"]
