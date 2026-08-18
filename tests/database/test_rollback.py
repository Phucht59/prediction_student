import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_rollback_restores_previous_state():
    result = json.loads(
        (ROOT / "artifacts/final/database/rollback_validation.json").read_text(encoding="utf-8")
    )
    assert result == {
        "backup_restore_hash_matches": True,
        "credentials": "REDACTED",
        "schema_cutback_executed": True,
        "status": "PASS",
        "transaction_rollback": True,
    }
