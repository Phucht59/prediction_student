import hashlib
import json
from pathlib import Path

from scripts.database_final import _build_parser


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


def test_backup_check_and_plan_commands_are_explicit():
    parser = _build_parser()
    backup_check = parser.parse_args(
        ["backup-check", "--backup-manifest", "custom-manifest.json"]
    )
    assert backup_check.command == "backup-check"
    assert backup_check.backup_manifest == "custom-manifest.json"
    plan = parser.parse_args(["plan", "--dsn-env", "POSTGRES_TEST_DSN"])
    assert plan.command == "plan"
    assert plan.dsn_env == "POSTGRES_TEST_DSN"


def test_versioned_backup_requires_explicit_source_contract():
    parser = _build_parser()
    backup = parser.parse_args(
        [
            "backup",
            "--dsn-env",
            "POSTGRES_TEST_DSN",
            "--expected-database",
            "student_predict",
            "--expected-schema-hash",
            "ae06a0afce55148dbe2b5452a9fe4efbf4d37860c5c05209fdb73799f40bf57e",
        ]
    )
    assert backup.expected_database == "student_predict"
    assert backup.expected_schema_hash.startswith("ae06a0")
