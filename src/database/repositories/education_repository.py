from __future__ import annotations

from psycopg2.extras import Json

from .base import Repository


class EducationRepository(Repository):
    def ensure_student(self, stable_key: str, identity_kind: str) -> int:
        return int(
            self.scalar(
                """INSERT INTO education.student(stable_key, identity_kind) VALUES (%s, %s)
                   ON CONFLICT (stable_key) DO UPDATE SET stable_key=EXCLUDED.stable_key RETURNING student_id""",
                (stable_key, identity_kind),
            )
        )

    def ensure_enrollment(self, student_id: int, dataset_version_id: int, source_record_key: str, source_row_number: int | None, subject: str | None, module: str | None, presentation: str | None, attributes: dict) -> int:
        return int(
            self.scalar(
                """INSERT INTO education.enrollment(student_id, dataset_version_id, source_record_key, source_row_number, subject, code_module, code_presentation, attributes)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (dataset_version_id, source_record_key) DO UPDATE SET attributes=EXCLUDED.attributes
                   RETURNING enrollment_id""",
                (student_id, dataset_version_id, source_record_key, source_row_number, subject, module, presentation, Json(attributes)),
            )
        )


__all__ = ["EducationRepository"]

