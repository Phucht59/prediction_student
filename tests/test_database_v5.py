import os
from pathlib import Path

import pytest

from scripts.database_v5 import MIGRATIONS, reset


def test_v5_migrations_are_ordered_transactional_and_compact():
    files = sorted(MIGRATIONS.glob("*.sql"))
    assert [path.name[:3] for path in files] == [f"{index:03d}" for index in range(1, 10)]
    assert len(files) == 9
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        assert text.startswith("BEGIN;")
        assert text.endswith("COMMIT;")


def test_v5_reset_requires_explicit_disposable_confirmation(monkeypatch):
    with pytest.raises(RuntimeError, match="confirm-disposable"):
        reset(False)


@pytest.mark.skipif(not (os.getenv("V5_DATABASE_URL") or os.getenv("POSTGRES_TEST_DSN")), reason="requires a dedicated disposable V5 PostgreSQL DSN")
def test_v5_live_database_integration_is_enabled_only_with_disposable_dsn():
    from scripts.database_v5 import audit, migrate
    assert migrate()["status"] == "PASS"
    assert audit()["status"] == "PASS"

