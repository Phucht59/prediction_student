import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_migration_idempotent():
    files = sorted((ROOT / "database/final/migrations").glob("*.sql"))
    assert [path.name[:3] for path in files] == [f"{index:03d}" for index in range(1, 11)]
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        assert text.startswith("BEGIN;")
        assert text.endswith("COMMIT;")
        assert "pg_advisory_xact_lock" in text
        assert "DROP DATABASE" not in text.upper()
        assert "TRUNCATE" not in text.upper()


def test_migration_checksum_immutable(final_connection):
    with final_connection.cursor() as cursor:
        cursor.execute("SELECT filename,sha256 FROM system.schema_migration ORDER BY filename")
        ledger = dict(cursor.fetchall())
    files = sorted((ROOT / "database/final/migrations").glob("*.sql"))
    assert len(ledger) == len(files)
    for path in files:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ledger[path.name].strip()
