import os
from src.database.connection import database_url
def test_url_uses_psycopg_and_env():
    assert database_url().drivername == 'postgresql+psycopg'
    assert os.getenv('DB_PASSWORD')
