import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_backup_manifest_valid():
    manifest = json.loads(
        (ROOT / "artifacts/final/database/backup_manifest.json").read_text(encoding="utf-8")
    )
    backup = ROOT / "backups" / manifest["backup_filename"]
    assert manifest["status"] == "PASS"
    assert backup.is_file()
    assert hashlib.sha256(backup.read_bytes()).hexdigest() == manifest["sha256"]


def test_backup_restore_pass():
    manifest = json.loads(
        (ROOT / "artifacts/final/database/backup_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["restore_test"]["status"] == "PASS"
    assert manifest["restore_test"]["schema_hash"] == manifest["starting_schema_hash"]
