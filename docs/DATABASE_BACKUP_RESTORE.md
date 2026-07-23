# Database Backup and Restore

The cutover backup is a custom-format dump with no owner and no ACL:

```powershell
pg_dump --format=custom --no-owner --no-acl `
  --file backups/student_predict_before_final_database.dump `
  $env:POSTGRES_TEST_DSN
```

The dump itself is ignored by Git. Its filename, size, SHA-256, PostgreSQL
version, source schema hash, and restore result are committed in
`artifacts/final/database/backup_manifest.json`.

## Restore validation

Restore only into a newly created disposable database:

```powershell
createdb -T template0 student_predict_restore_test
pg_restore --exit-on-error --no-owner --no-acl `
  --dbname student_predict_restore_test `
  backups/student_predict_before_final_database.dump
```

PASS requires the restored structural signature to equal the recorded source
signature. Table counts alone are insufficient.

## Recovery

Never run `DROP DATABASE` against the active service. Create a new database,
restore the verified dump, validate its signature, and switch the external
connection only after application health checks. The schema-cutback SQL in
`database/final/rollback/` restores legacy search paths without deleting final
evidence.
