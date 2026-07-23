import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _checks():
    payload = json.loads(
        (ROOT / "artifacts/final/database/permission_validation.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "PASS"
    return payload["checks"]


def test_reader_cannot_write():
    assert _checks()["reader_cannot_write"]


def test_writer_cannot_drop():
    assert _checks()["writer_cannot_drop"]


def test_migrator_not_used_at_runtime():
    assert _checks()["migrator_not_runtime"]
    source = (ROOT / "src/database/connection.py").read_text(encoding="utf-8")
    assert "POSTGRES_RUNTIME_APP_DSN" in source
