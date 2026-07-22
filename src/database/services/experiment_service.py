from __future__ import annotations

from psycopg2.extras import Json

from ..connection import DatabaseSettings, transaction


class ExperimentService:
    def __init__(self, settings: DatabaseSettings):
        self.settings = settings

    def start_training_run(self, *, study_id: int, dataset_version_id: int, snapshot_id: int, split_id: int, model_version_id: int, seed: int, hardware: dict) -> int:
        with transaction(self.settings) as connection, connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO experiment.training_run(study_id,dataset_version_id,snapshot_id,split_id,model_version_id,seed,status,hardware,started_at)
                   VALUES (%s,%s,%s,%s,%s,%s,'running',%s,NOW()) RETURNING training_run_id""",
                (study_id, dataset_version_id, snapshot_id, split_id, model_version_id, seed, Json(hardware)),
            )
            return int(cursor.fetchone()[0])

    def complete_training_run(self, training_run_id: int, checkpoint_path: str, checkpoint_sha256: str) -> None:
        with transaction(self.settings) as connection, connection.cursor() as cursor:
            cursor.execute(
                """UPDATE experiment.training_run SET status='completed', checkpoint_path=%s, checkpoint_sha256=%s, completed_at=NOW()
                   WHERE training_run_id=%s AND status='running'""",
                (checkpoint_path, checkpoint_sha256, training_run_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Training run was not in a completable state")


__all__ = ["ExperimentService"]

