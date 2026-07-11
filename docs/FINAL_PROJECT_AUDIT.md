# Final project audit

Audit date: 2026-07-10. DOCX files were not edited.

## Repository and frozen evidence

- Main is the only branch on the remote; cleanup is complete.
- Frozen scientific evidence remains
  `artifacts/final/final-a2945d79-9845-4979-b148-159f4853eca3/`.
- Selected-config SHA-256:
  `cda38460197627ac1d71e764f61d784e4c03cf6f86775339d38787c6890678ad`.
- Frozen prediction checksum:
  `d5b6f86d50a1a4c90b6a68139ec0eb6f4635e55c572c647d6d9b62d5a31f4a74`.
- Offline tests: 57 passed, 5 skipped. The skips are PostgreSQL integration
  tests because test DSNs/credentials are not configured.

## Live PostgreSQL status

The live `student_predict` database is reachable with the application role and
contains one dataset version and 395 source records. Migration
`003_add_source_record_targets.sql` has not been applied because that role
cannot create tables in `public`; `source_record_targets` is therefore absent.
The lineage backup is outside the repository at
`C:\Huflit\backups\student_predict_before_003.dump`.

The following are intentionally pending:

- administrator migration and target backfill (395 rows, expected distribution
  130/192/73);
- PostgreSQL integration tests without skips;
- a new live DB-first verification run and evidence bundle.

These pending items must not be reported as completed. The current scientific
conclusion and metrics remain those of the frozen pre-migration evidence.

## Technical contract

CSV is restricted to the explicit ingestion boundary. Model selection, final
training, recommendation and inference use the PostgreSQL loader. Target
storage migration and manual administrator steps are documented in
`MANUAL_POSTGRESQL_MIGRATION.md`.
