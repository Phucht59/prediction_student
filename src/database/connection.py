from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import psycopg2
from psycopg2.extensions import connection as Connection

ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or (ROOT / ".env")
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class DatabaseSettings:
    dsn: str | None = None
    host: str | None = None
    port: int = 5432
    database: str | None = None
    user: str | None = None
    password: str | None = None

    @classmethod
    def from_environment(
        cls, *, require_mutating_dsn: bool = False
    ) -> "DatabaseSettings":
        load_dotenv()
        dsn = (
            os.getenv("POSTGRES_TEST_DSN")
            if require_mutating_dsn
            else (
                os.getenv("POSTGRES_RUNTIME_APP_DSN")
                or os.getenv("DATABASE_URL")
                or os.getenv("FINAL_DATABASE_URL")
            )
        )
        if require_mutating_dsn and not dsn and not os.getenv("DB_HOST") and not os.getenv("POSTGRES_HOST"):
            raise RuntimeError(
                "A mutating database command requires POSTGRES_TEST_DSN or DB_* / POSTGRES_* settings"
            )
        if dsn:
            return cls(dsn=dsn)
        host = os.getenv("POSTGRES_HOST") or os.getenv("DB_HOST")
        database = os.getenv("POSTGRES_DB") or os.getenv("DB_NAME")
        user = os.getenv("POSTGRES_USER") or os.getenv("DB_USER")
        password = os.getenv("POSTGRES_PASSWORD") or os.getenv("DB_PASSWORD")
        port = int(os.getenv("POSTGRES_PORT") or os.getenv("DB_PORT") or "5432")
        if not host or not database or not user:
            raise RuntimeError("PostgreSQL host/database/user are missing from the environment")
        return cls(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
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

