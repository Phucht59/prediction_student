from __future__ import annotations

from psycopg2.extras import Json

from .base import Repository


class ExperimentRepository(Repository):
    def register_study(self, name: str, dataset_id: int, question: str, metric: str, protocol_version: str) -> int:
        return int(
            self.scalar(
                """WITH inserted AS (
                       INSERT INTO experiment.study(study_name,dataset_id,research_question,primary_metric,protocol_version)
                       VALUES (%s,%s,%s,%s,%s)
                       ON CONFLICT (study_name) DO NOTHING
                       RETURNING study_id
                   )
                   SELECT study_id FROM inserted
                   UNION ALL SELECT study_id FROM experiment.study WHERE study_name=%s
                   LIMIT 1""",
                (name, dataset_id, question, metric, protocol_version, name),
            )
        )

    def register_model_version(self, name: str, family: str, version: str, config: dict, config_sha256: str, protocol_version: str) -> int:
        return int(
            self.scalar(
                """WITH inserted AS (
                       INSERT INTO experiment.model_version(model_name,model_family,version_label,config,config_sha256,protocol_version)
                       VALUES (%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (model_name,version_label) DO NOTHING
                       RETURNING model_version_id
                   )
                   SELECT model_version_id FROM inserted
                   UNION ALL SELECT model_version_id FROM experiment.model_version
                   WHERE model_name=%s AND version_label=%s
                   LIMIT 1""",
                (name, family, version, Json(config), config_sha256, protocol_version, name, version),
            )
        )


__all__ = ["ExperimentRepository"]
