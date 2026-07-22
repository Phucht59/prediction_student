from __future__ import annotations

from psycopg2.extras import Json

from .base import Repository


class FeatureRepository(Repository):
    def register_snapshot(self, dataset_version_id: int, name: str, cutoff: str | None, path: str, sha256: str, row_count: int, channel_count: int | None, contract: dict, target_location: str, generator: str, protocol_version: str) -> int:
        return int(
            self.scalar(
                """INSERT INTO feature.snapshot(dataset_version_id, snapshot_name, cutoff_id, storage_path, sha256, row_count, channel_count, feature_contract, target_location, generator, protocol_version)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING snapshot_id""",
                (dataset_version_id, name, cutoff, path, sha256, row_count, channel_count, Json(contract), target_location, generator, protocol_version),
            )
        )


__all__ = ["FeatureRepository"]

