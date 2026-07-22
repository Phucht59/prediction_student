from __future__ import annotations

from typing import Any

from psycopg2.extensions import connection as Connection


class Repository:
    def __init__(self, connection: Connection):
        self.connection = connection

    def scalar(self, statement: str, parameters: tuple[Any, ...] = ()) -> Any:
        with self.connection.cursor() as cursor:
            cursor.execute(statement, parameters)
            row = cursor.fetchone()
            return row[0] if row else None

    def execute(self, statement: str, parameters: tuple[Any, ...] = ()) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(statement, parameters)


__all__ = ["Repository"]

