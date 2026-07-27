from __future__ import annotations

from pathlib import Path


class ResultImportService:
    """Validate canonical source presence before the migration CLI imports it."""

    REQUIRED = (
        "artifacts/final/final_results.json",
        "artifacts/final/model_registry.json",
        "artifacts/final/recommendation/risk_profiles.parquet",
        "artifacts/final/recommendation/recommendation_plans.jsonl",
    )

    def __init__(self, root: Path):
        self.root = root

    def missing_sources(self) -> list[str]:
        return [relative for relative in self.REQUIRED if not (self.root / relative).is_file()]


__all__ = ["ResultImportService"]
