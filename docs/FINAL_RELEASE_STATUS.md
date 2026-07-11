# Final release status

- Branch: `main`; remote branch set: `origin/main` only.
- Current source state: `main` HEAD (use `git rev-parse HEAD` for the exact release commit).
- Existing release tag: `thesis-project-postgresql-final-v2`.
- Frozen evidence: `artifacts/final/final-a2945d79-9845-4979-b148-159f4853eca3/`.
- Selected config SHA-256: `cda38460197627ac1d71e764f61d784e4c03cf6f86775339d38787c6890678ad`.
- Frozen prediction checksum: `d5b6f86d50a1a4c90b6a68139ec0eb6f4635e55c572c647d6d9b62d5a31f4a74`.
- Offline tests: 57 passed, 5 skipped (PostgreSQL credentials pending).
- Live database: reachable; 1 dataset version and 395 source records.
- Migration 003: implementation complete, live application pending admin privilege.
- `source_record_targets`: not yet present on live DB.
- DB-first live verification: pending.
- Expert recommendation review: pending.
- DOCX: untouched; report rewrite is next.

See `MANUAL_POSTGRESQL_MIGRATION.md` for the administrator procedure.
