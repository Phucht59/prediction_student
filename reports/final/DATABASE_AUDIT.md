# Database Audit

The public PostgreSQL design is centered on `system`, `catalog`, `ml` and
`recommendation`. Frozen reconciliation evidence is in
`artifacts/final/database`.

Expected persisted counts are 15,378 risk profiles, 15,378 plan objects and
27,355 actions. An `ABSTAINED` plan exists for traceability and contains no
recommended actions.
