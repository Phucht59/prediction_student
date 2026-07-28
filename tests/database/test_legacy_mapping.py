import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_all_legacy_tables_have_disposition():
    mapping = yaml.safe_load(
        (ROOT / "database/final/LEGACY_TO_FINAL_MAPPING.yaml").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (ROOT / "artifacts/final/database/audit_before/tables.json").read_text(encoding="utf-8")
    )
    mapped = {row["old_table"] for row in mapping["mappings"]}
    actual = {f"{row['schema_name']}.{row['table_name']}" for row in audit}
    assert len(mapped) == 29
    assert mapped == actual
    assert all(row["destination"] for row in mapping["mappings"])


def test_no_nonempty_table_dropped():
    cutover = json.loads(
        (ROOT / "artifacts/final/database/cutover_validation.json").read_text(encoding="utf-8")
    )
    assert all(
        row["rows"] == 0 or row["decision"] != "DROP_EMPTY_REDUNDANT"
        for row in cutover["legacy_disposition"]
    )


def test_empty_drop_requires_explicit_flag():
    source = (ROOT / "scripts/database_final.py").read_text(encoding="utf-8")
    assert "--confirm-drop-empty-legacy" in source
    assert "elif drop_empty:" in source


def test_cutover_uses_requested_backup_manifest():
    source = (ROOT / "scripts/database_final.py").read_text(encoding="utf-8")
    assert "backup_manifest=Path(args.backup_manifest)" in source
    assert "_validate_backup_manifest(backup_manifest)" in source


def test_database_plan_is_read_only_and_never_authorizes_cutover():
    source = (ROOT / "scripts/database_final.py").read_text(encoding="utf-8")
    plan_source = source.split("def command_plan", 1)[1].split(
        "def command_status", 1
    )[0]
    assert "_connect(dsn, readonly=True)" in plan_source
    assert '"dry_run": True' in plan_source
    assert '"cutover_performed": False' in plan_source
    assert '"cutover_authorized": False' in plan_source
