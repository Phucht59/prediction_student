from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import psycopg2
from psycopg2.extensions import connection as Connection


@dataclass(frozen=True)
class DatabaseSettings:
    dsn: str | None = None
    host: str | None = None
    port: int = 5432
    database: str | None = None
    user: str | None = None
    password: str | None = None

    @classmethod
    def from_environment(cls, *, require_v5_dsn: bool = False) -> "DatabaseSettings":
        dsn = (
            (os.getenv("V5_DATABASE_URL") or os.getenv("POSTGRES_TEST_DSN"))
            if require_v5_dsn
            else os.getenv("POSTGRES_RUNTIME_APP_DSN")
        )
        if require_v5_dsn and not dsn:
            raise RuntimeError("A mutating database command requires V5_DATABASE_URL or POSTGRES_TEST_DSN")
        if dsn:
            return cls(dsn=dsn)
        return cls(
            host=os.getenv("POSTGRES_HOST"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
        )

    def redacted(self) -> dict[str, object]:
        return {
            "dsn_present": bool(self.dsn),
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password_present": bool(self.password),
        }


def connect_with_retry(settings: DatabaseSettings, attempts: int = 3) -> Connection:
    delay = 0.2
    for attempt in range(1, attempts + 1):
        try:
            if settings.dsn:
                return psycopg2.connect(settings.dsn, connect_timeout=5)
            return psycopg2.connect(
                host=settings.host,
                port=settings.port,
                dbname=settings.database,
                user=settings.user,
                password=settings.password,
                connect_timeout=5,
            )
        except psycopg2.OperationalError:
            if attempt == attempts:
                raise
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


@contextmanager
def transaction(settings: DatabaseSettings) -> Iterator[Connection]:
    connection = connect_with_retry(settings)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


__all__ = ["DatabaseSettings", "connect_with_retry", "transaction"]

