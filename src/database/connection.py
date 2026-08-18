import os
from contextlib import contextmanager
from collections.abc import Iterator

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required database environment variable: {name}")
    return value

def database_url() -> URL:
    return URL.create(drivername="postgresql+psycopg", username=_required("DB_USER"), password=_required("DB_PASSWORD"), host=os.getenv("DB_HOST", "localhost"), port=int(os.getenv("DB_PORT", "5432")), database=_required("DB_NAME"))

DATABASE_URL = database_url()

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


def test_connection() -> tuple[str, str]:
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT current_database(), current_user")
        ).one()

        return str(result[0]), str(result[1])

@contextmanager
def transaction() -> Iterator:
    with engine.begin() as conn:
        yield conn


if __name__ == "__main__":
    test_connection()
