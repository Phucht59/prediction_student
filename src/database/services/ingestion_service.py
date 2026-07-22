from __future__ import annotations

from pathlib import Path

from src.studies.v5.common.protocol import sha256_file

from ..connection import DatabaseSettings, transaction
from ..repositories import SourceRepository


class IngestionService:
    def __init__(self, settings: DatabaseSettings):
        self.settings = settings

    def register_source(self, *, slug: str, display_name: str, source_path: Path, version_label: str, row_count: int, data_schema: dict, license_note: str | None = None) -> dict[str, int | str]:
        digest = sha256_file(source_path)
        with transaction(self.settings) as connection:
            repository = SourceRepository(connection)
            dataset_id = repository.register_dataset(slug, display_name, None, license_note)
            version_id = repository.register_version(dataset_id, version_label, digest, row_count, data_schema)
            file_id = repository.register_file(
                version_id,
                source_path.name,
                source_path.as_posix(),
                digest,
                source_path.stat().st_size,
                row_count,
                "text/csv",
            )
        return {"dataset_id": dataset_id, "dataset_version_id": version_id, "source_file_id": file_id, "sha256": digest}


__all__ = ["IngestionService"]

