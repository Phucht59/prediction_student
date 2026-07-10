# Manual PostgreSQL migration (pending)

The source code is PostgreSQL-first, but migration 003 has not yet been applied
to the live `student_predict` database. The current application role
`student_predict_app` can read and write the ML lineage tables but cannot create
tables in schema `public`.

An administrator must apply the migration. The administrator role must be able
to create tables, functions, triggers, indexes and foreign keys in `public`.

A local lineage backup was created outside the repository at:

```text
C:\Huflit\backups\student_predict_before_003.dump
```

Do not commit that dump or any credentials.

Example command after setting an administrator-only DSN at runtime:

```powershell
psql $env:POSTGRES_ADMIN_DSN `
  -v ON_ERROR_STOP=1 `
  -f database/migrations/003_add_source_record_targets.sql
```

After migration, verify the `source_record_targets` foreign key, uniqueness
constraint and append-only trigger. Backfill or ingest the 395 targets through
the source-record lineage, then verify the expected class counts (130 Low, 192
Medium, 73 High). Finally set the PostgreSQL test DSNs and run the integration
tests and DB-first frozen verification. Until those steps pass, live DB
verification remains pending and the frozen scientific evidence remains the
pre-migration evidence bundle.
