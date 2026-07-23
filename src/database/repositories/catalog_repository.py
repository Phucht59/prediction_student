from __future__ import annotations

from psycopg2.extras import RealDictCursor

from ..models import Dataset
from .base import Repository


class CatalogRepository(Repository):
    def get_dataset(self, slug: str) -> Dataset | None:
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT dataset_id,slug,display_name,task_type,class_labels
                FROM catalog.dataset WHERE slug=%s
                """,
                (slug,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return Dataset(
            dataset_id=int(row["dataset_id"]),
            slug=row["slug"],
            display_name=row["display_name"],
            task_type=row["task_type"],
            class_labels=tuple(row["class_labels"]),
        )

    def record_pk(self, dataset_slug: str, source_record_id: str) -> int | None:
        value = self.scalar(
            """
            SELECT r.record_pk
            FROM catalog.record r
            JOIN catalog.dataset_version dv USING(dataset_version_id)
            JOIN catalog.dataset d USING(dataset_id)
            WHERE d.slug=%s AND r.source_record_id=%s AND dv.status='sealed'
            """,
            (dataset_slug, source_record_id),
        )
        return int(value) if value is not None else None


__all__ = ["CatalogRepository"]
