# Database Operations

## Environment

Runtime applications use only `POSTGRES_RUNTIME_APP_DSN`. Migration and test
DSNs are read from process environment variables and never printed or
persisted. Runtime startup should fail when a least-privileged runtime DSN is
not supplied.

## Read-only status

```powershell
python project.py db-final status
python project.py db-final inventory
```

## Disposable verification

Create or restore a database whose name contains `test`, `dev`, `disposable`,
or `final_dev`, set `FINAL_DATABASE_URL`, then run:

```powershell
python project.py db-final migrate
python project.py db-final load-results
python project.py db-final validate --strict-public
python -m pytest -q tests/database
```

The migrate and loader commands are idempotent. The migration ledger rejects
any file whose checksum changed after application.

## Target cutover

Cutover is refused unless backup, restore, disposable validation, permissions,
reconciliation, source checksums, and rollback are PASS:

```powershell
python project.py db-final cutover `
  --confirm-production-cutover `
  --confirm-drop-empty-legacy `
  --backup-manifest artifacts/final/database/backup_manifest.json
```

Empty-table removal has a separate explicit flag. A non-empty legacy table is
always moved to `legacy_202607` and never dropped.

## Roles

- `student_predict_migrator`: schema/migration operations; never runtime.
- `student_predict_writer`: read final metadata and insert allowed
  recommendation runtime rows; cannot drop.
- `student_predict_reader`: read-only.

No role is superuser, database creator, or role creator.
