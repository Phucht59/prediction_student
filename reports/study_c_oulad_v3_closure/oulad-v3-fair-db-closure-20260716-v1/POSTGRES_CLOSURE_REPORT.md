# PostgreSQL Scientific Closure

## Outcome

- Migration dry-run: `PASS` with rollback; committed migration status: `PASS`.
- Applied migrations: `005_oulad_lineage_and_snapshot_registry.sql`, `006_oulad_v3_fair_evidence_registry.sql`, and set-based integrity optimization `007_optimize_bulk_lineage_integrity_triggers.sql`.
- Migration 007 preserves the same sealed-dataset/running-run rule with statement-level transition tables; append-only, FK, uniqueness, status, and completed-run triggers remain active.
- Rows removed: `0`; executed cleanup predicates: `[]`.
- Registered: `15378` source records, `8` completed candidate runs, `123024` predictions, and `3` evidence bundles.
- Reproduction: `PASS`; max probability difference `0`; max metric difference `1.11e-16`.
- Least-privileged app permission audit: `PASS` as `student_predict_app_local`; superuser app evidence forbidden.

## Before/after key counts

| Table | Before | After | Delta |
|---|---:|---:|---:|
| source_records | 395 | 15773 | 15378 |
| source_record_targets | 395 | 15773 | 15378 |
| ml_experiment_runs | 17 | 25 | 8 |
| ml_run_record_splits | 6715 | 129739 | 123024 |
| ml_predictions | 1027 | 124051 | 123024 |
| ml_run_metrics | 78 | 838 | 760 |
| ml_evidence_bundles | 0 | 3 | 3 |

## Integrity and query behavior

After-audit reports zero orphan splits, zero orphan predictions, zero duplicate prediction keys, and zero invalid run statuses. The record-key expression index and existing run/prediction indexes support exact artifact-to-database replay. See `postgres_query_plans.md` for `EXPLAIN (ANALYZE, BUFFERS)` output.

No production database, external benchmark, model training, or recommendation generation was performed.
