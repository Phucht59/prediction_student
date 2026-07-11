# Final release status

- Branch: `main`; remote branch set: `origin/main` only.
- Current source state: `main` HEAD (use `git rev-parse HEAD` for the exact release commit).
- Existing release tag: `thesis-project-postgresql-final-v2`.
- Frozen evidence: `artifacts/final/final-a2945d79-9845-4979-b148-159f4853eca3/`.
- Selected config SHA-256: `cda38460197627ac1d71e764f61d784e4c03cf6f86775339d38787c6890678ad`.
- Frozen prediction checksum: `d5b6f86d50a1a4c90b6a68139ec0eb6f4635e55c572c647d6d9b62d5a31f4a74`.
- Tests with live PostgreSQL credentials: 62 passed, 0 skipped.
- Live database: reachable; 1 dataset version and 395 source records.
- Migration 003: applied; 395 target rows cover 395 source records.
- `source_record_targets`: not yet present on live DB.
- DB-first live verification: passed (`5a0b5041-5216-4a48-9e46-b0c16ab14866`).
- Expert recommendation review: pending.
- DOCX: untouched; report rewrite is next.

See `MANUAL_POSTGRESQL_MIGRATION.md` for the administrator procedure.
