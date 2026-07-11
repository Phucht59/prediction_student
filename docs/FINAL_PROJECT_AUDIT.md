# Final project audit

Audit date: 2026-07-11. DOCX files were not edited.

## Repository and frozen evidence

- Main is the only branch on the remote; cleanup is complete.
- Frozen scientific evidence remains
  `artifacts/final/final-a2945d79-9845-4979-b148-159f4853eca3/`.
- Selected-config SHA-256:
  `cda38460197627ac1d71e764f61d784e4c03cf6f86775339d38787c6890678ad`.
- Frozen prediction checksum:
  `d5b6f86d50a1a4c90b6a68139ec0eb6f4635e55c572c647d6d9b62d5a31f4a74`.
- Tests: 62 passed, 0 skipped, including all five live PostgreSQL integration tests.

## Live PostgreSQL status

The live `student_predict` database contains one dataset version, 395 source
records and 395 target rows. Migration `003_add_source_record_targets.sql` is
applied; distribution is 130/192/73 with zero duplicates and zero orphans.
The lineage backup is outside the repository at
`C:\Huflit\backups\student_predict_before_003.dump`.

The live DB-first run `5a0b5041-5216-4a48-9e46-b0c16ab14866` reproduces all 79
predicted classes. Maximum probability drift is `2.78e-08`; principal metric
deltas are zero. Independent expert recommendation review remains pending.

## Technical contract

CSV is restricted to the explicit ingestion boundary. Model selection, final
training, recommendation and inference use the PostgreSQL loader. Target
storage migration and manual administrator steps are documented in
`MANUAL_POSTGRESQL_MIGRATION.md`.
