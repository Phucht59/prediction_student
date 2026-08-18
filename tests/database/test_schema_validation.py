import pytest
from src.database.schema import check_database_schema
@pytest.mark.integration
def test_live_schema():
    assert check_database_schema()['ok']
