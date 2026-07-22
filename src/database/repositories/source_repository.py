from __future__ import annotations

from psycopg2.extras import Json

from .base import Repository


class SourceRepository(Repository):
    def register_dataset(self, slug: str, display_name: str, source_uri: str | None, license_note: str | None) -> int:
        return int(
            self.scalar(
                """INSERT INTO source.dataset(slug, display_name, source_uri, license_note)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (slug) DO UPDATE SET display_name = EXCLUDED.display_name
                   RETURNING dataset_id""",
                (slug, display_name, source_uri, license_note),
            )
        )

    def register_version(self, dataset_id: int, version_label: str, sha256: str, row_count: int, data_schema: dict) -> int:
        return int(
            self.scalar(
                """INSERT INTO source.dataset_version(dataset_id, version_label, source_sha256, row_count, data_schema)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (dataset_id, version_label) DO NOTHING
                   RETURNING dataset_version_id""",
                (dataset_id, version_label, sha256, row_count, Json(data_schema)),
            )
            or self.scalar(
                "SELECT dataset_version_id FROM source.dataset_version WHERE dataset_id=%s AND version_label=%s",
                (dataset_id, version_label),
            )
        )

    def register_file(self, dataset_version_id: int, logical_name: str, path: str, sha256: str, byte_count: int, row_count: int | None, media_type: str) -> int:
        return int(
            self.scalar(
                """INSERT INTO source.source_file(dataset_version_id, logical_name, storage_path, sha256, byte_count, row_count, media_type)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (dataset_version_id, logical_name) DO NOTHING
                   RETURNING source_file_id""",
                (dataset_version_id, logical_name, path, sha256, byte_count, row_count, media_type),
            )
            or self.scalar(
                "SELECT source_file_id FROM source.source_file WHERE dataset_version_id=%s AND logical_name=%s",
                (dataset_version_id, logical_name),
            )
        )


__all__ = ["SourceRepository"]

