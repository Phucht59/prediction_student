# PostgreSQL migration 003: new-environment and recovery procedure

The current live `student_predict` database already has migration 003 applied.
It contains one dataset version, 395 source records and 395 target rows. This
procedure is for disaster recovery or provisioning another environment.

## Prerequisites

- Use an administrator role that can create tables, functions, triggers,
  indexes and foreign keys in the target schema.
- Set the administrator DSN only in the runtime environment. Never commit a
  password, DSN, `.env` file or database dump.
- Create and verify a backup before changing an existing database. Keep the
  dump outside the repository.

## Apply the migration

```powershell
psql $env:POSTGRES_ADMIN_DSN `
  -v ON_ERROR_STOP=1 `
  -f database/migrations/003_add_source_record_targets.sql
```

The migration must create `source_record_targets` with its composite lineage
foreign key, uniqueness constraint, index and immutability trigger.

## Restore target lineage

Ingest or backfill targets only from the checksum-verified ingestion source.
Join each target by `dataset_version_id` and stable `record_id`; never
derive targets from predictions or a volatile CSV row index. Keep G3 outside
the model feature payload.

## Verification checklist

- `source_dataset_versions`: 1 for the Student-Mat release dataset.
- `source_records`: 395.
- `source_record_targets`: 395.
- Encoded target distribution: 130 Low, 192 Medium, 73 High.
- Duplicate target rows: 0.
- Orphan target rows: 0.
- PostgreSQL integration tests: 5/5 passed.
- Frozen DB-first evaluation uses the unchanged selected-config checksum and
  unchanged split hashes.

After configuring isolated admin/application test DSNs, run:

```powershell
py -3.10 -m pytest -q tests/test_postgres_source_ml_integration.py
py -3.10 scripts/verify_final_evidence.py
```
