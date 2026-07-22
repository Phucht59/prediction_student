# Database Review Before V5

The current PostgreSQL instance was inspected read-only through `student_predict_app`. No schema, data, role or migration was changed.

## Current state

- PostgreSQL 18.4 is running locally.
- The current database is `student_predict`; all 29 application tables are in `public`.
- The repository contains 7 historical migrations, 62 visible indexes, 21 visible triggers and 368 constraints (including system-generated constraints visible for `public`).
- The application role is not superuser and cannot create databases or roles.
- Every lineage/evaluation table visible to the application role currently has zero rows. Recommendation governance tables correctly deny this role direct reads, so their counts were not inferred.
- No `POSTGRES_TEST_DSN`, `POSTGRES_ADMIN_DSN` or `V5_DATABASE_URL` is configured. Destructive/integration testing is therefore not authorized on the current database.

## Component decisions

| Current component | Used now | Problem | Decision |
|---|---|---|---|
| `source_dataset_versions`, `source_dataset_files` | Yes in prior code | Names are flat and source identity is split across later migrations. | Replace with `source.dataset`, `source.dataset_version`, `source.source_file`. |
| `source_records`, `source_record_targets` | Prior UCI path | Generic records and targets are disconnected from a clear enrollment entity. | Merge their role into typed `education.student`, `education.enrollment`, `education.grade_record`, `education.outcome`. |
| `ml_experiment_runs`, `study_extension_runs` | Historical | Two run concepts overlap. | Replace with one `experiment.training_run` linked to `experiment.study`. |
| `ml_run_record_splits`, `split_manifest_registry` | Historical | Split identity and membership are separate and naming is ML-specific. | Replace with `experiment.split` and `experiment.split_member`. |
| `cutoff_feature_snapshots`, `snapshot_record_index` | OULAD lineage | Useful concept but flat naming and separate registry conventions. | Replace with `feature.snapshot` and `feature.snapshot_member`. |
| `prediction_cohorts`, `ml_predictions`, `prediction_snapshots` | Historical | Three representations overlap and contracts differ. | Replace with `evaluation.prediction_set` and optional row-level `evaluation.prediction`; large sets remain Parquet. |
| `ml_run_metrics` | Yes in prior audit | Scope/aggregation semantics are not first-class enough for V5. | Replace with `evaluation.metric` containing explicit scope and aggregation. |
| `ml_evidence_bundles` | V4 closure | Mirrors filesystem evidence and migration history. | Replace with a single `experiment.artifact`; preserve V4 table/migration in the frozen namespace. |
| `ml_recommendations` | Superseded | Overlaps governed recommendation tables. | Drop from V5 design. |
| `recommendation_policies`, feature/action catalog | Prior recommendation | Catalog/feature tables are useful but overly coupled to one phase. | Consolidate rules/catalog in versioned `recommendation.policy.rules`. |
| `recommendation_instances`, revisions, goals, actions | Prior recommendation | Core lifecycle is useful, but many payload-only tables and separate revision entity add joins. | Use `case`, versioned `plan`, normalized `action`, `review`, `follow_up`. |
| `advisor_decisions` | Prior recommendation | Decision is a review concept. | Merge into `recommendation.review`. |
| expert review tables | Templates only | Not required for the operational V5 lifecycle. | Keep expert casebook as an artifact; do not create V5 database tables yet. |
| recommendation outcomes | Not causal evidence | Could imply effectiveness established. | Omit until a real follow-up/effectiveness protocol exists. |

## V5 design decision

V5 is rebuilt as six independent schemas with 27 domain tables and two immutability triggers, versus 29 flat tables and 21 visible triggers. The redesign keeps V4 migrations untouched. It removes dataset/model-specific table names, makes enrollment and snapshot identity explicit, separates metric scope from aggregation, and stores large tensors/predictions outside PostgreSQL with path, row count and SHA-256 lineage.

Nine ordered V5 migrations are placed under `database/v5/migrations`. They are intended for a new disposable/V5 database, not the current `student_predict` database. The migration runner refuses mutation unless `V5_DATABASE_URL` or `POSTGRES_TEST_DSN` is explicitly provided, and reset additionally requires `--confirm-disposable` plus a database name containing `test`, `dev`, `disposable` or `_v5`.

## Integration status

Migration SQL and the Python access layer can be statically validated now. Live migration, permission, backup/restore and rollback tests remain **NOT RUN** until a dedicated V5/disposable DSN and backup path are available. This is an explicit environment limitation, not a PASS.
