# Database 30-Model Cutover Guide

The canonical artifacts contain 30 model-dataset identities (10 models for each of Student-Mat, Student-Por, and OULAD). The existing local `student_predict` database is intentionally not mutated by the closure audit and may still contain 27 identities.

Run these commands manually after merge from the repository root in PowerShell. Keep the DSN in an environment variable so credentials are not written to shell history or reports.

```powershell
$env:POSTGRES_TEST_DSN="postgresql://postgres:<password>@localhost:5432/student_predict"

python project.py db-final status --dsn-env POSTGRES_TEST_DSN
python project.py db-final plan --dsn-env POSTGRES_TEST_DSN
python project.py db-final backup-check --backup-manifest artifacts/final/database/backup_manifest.json
```

`plan` is read-only and reports `dry_run=true`, current counts, expected counts, and all cutover prerequisites. If backup-check is not PASS, create and restore-test a fresh backup before cutover:

```powershell
python project.py db-final backup --dsn-env POSTGRES_TEST_DSN --restore-database student_predict_restore_test
python project.py db-final backup-check --backup-manifest artifacts/final/database/backup_manifest.json
```

Only after the plan, backup/restore gate, disposable migration validation, and permission validation are PASS, execute the explicit cutover:

```powershell
python project.py db-final cutover --dsn-env POSTGRES_TEST_DSN --backup-manifest artifacts/final/database/backup_manifest.json --confirm-production-cutover
python project.py db-final validate --dsn-env POSTGRES_TEST_DSN --strict-public
python project.py db-final status --dsn-env POSTGRES_TEST_DSN
```

Do not add `--confirm-drop-empty-legacy` unless the separate empty-table disposition has been reviewed and dropping those empty legacy tables is explicitly intended. Without that flag, legacy tables are archived read-only.

Expected post-cutover invariants:

| Entity | Expected |
|---|---:|
| Model-dataset identities | 30 |
| Risk profiles | 15,378 |
| Recommendation plan objects | 15,378 |
| Recommendation actions | 27,355 |
| Fake expert reviews | 0 |

The cutover refuses to run without explicit confirmation, a PASS backup/restore manifest whose dump checksum matches, PASS disposable migration and permission evidence, and unchanged locked canonical sources. Migrations are checksum-ledgered and canonical loads are idempotent where supported by their natural keys.
