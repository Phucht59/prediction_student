from __future__ import annotations

import os

import psycopg2
import pytest


@pytest.fixture()
def final_connection():
    dsn = os.getenv("FINAL_DATABASE_URL")
    if not dsn:
        pytest.skip("FINAL_DATABASE_URL is required for live final database tests")
    connection = psycopg2.connect(dsn)
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()
