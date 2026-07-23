# Database Code Usage

| Schema | Table | Read refs | Write refs | Migration refs | Test refs | Runtime status |
|---|---|---:|---:|---:|---:|---|
| public | advisor_decisions | 0 | 0 | 1 | 0 | MIGRATION_ONLY |
| public | cutoff_feature_snapshots | 0 | 0 | 2 | 1 | MIGRATION_ONLY |
| public | expert_review_cases | 0 | 0 | 1 | 0 | MIGRATION_ONLY |
| public | expert_review_ratings | 0 | 0 | 1 | 0 | MIGRATION_ONLY |
| public | ml_evidence_bundles | 2 | 1 | 1 | 1 | ACTIVE_RUNTIME |
| public | ml_experiment_runs | 8 | 3 | 3 | 2 | ACTIVE_RUNTIME |
| public | ml_predictions | 7 | 3 | 4 | 3 | ACTIVE_RUNTIME |
| public | ml_recommendations | 2 | 2 | 2 | 2 | ACTIVE_RUNTIME |
| public | ml_run_metrics | 2 | 3 | 2 | 1 | ACTIVE_RUNTIME |
| public | ml_run_record_splits | 6 | 3 | 2 | 1 | ACTIVE_RUNTIME |
| public | ml_schema_migrations | 2 | 1 | 1 | 1 | ACTIVE_RUNTIME |
| public | prediction_cohorts | 0 | 0 | 2 | 1 | MIGRATION_ONLY |
| public | prediction_snapshots | 0 | 0 | 2 | 0 | MIGRATION_ONLY |
| public | recommendation_action_catalog | 0 | 0 | 1 | 0 | MIGRATION_ONLY |
| public | recommendation_actions | 0 | 0 | 1 | 0 | MIGRATION_ONLY |
| public | recommendation_feature_registry | 0 | 0 | 1 | 0 | MIGRATION_ONLY |
| public | recommendation_follow_ups | 0 | 0 | 1 | 0 | MIGRATION_ONLY |
| public | recommendation_goals | 0 | 0 | 1 | 0 | MIGRATION_ONLY |
| public | recommendation_instances | 0 | 0 | 1 | 0 | MIGRATION_ONLY |
| public | recommendation_outcomes | 0 | 0 | 1 | 0 | MIGRATION_ONLY |
| public | recommendation_policies | 0 | 0 | 2 | 0 | MIGRATION_ONLY |
| public | recommendation_revisions | 0 | 0 | 2 | 0 | MIGRATION_ONLY |
| public | snapshot_record_index | 0 | 0 | 2 | 1 | MIGRATION_ONLY |
| public | source_dataset_files | 0 | 0 | 2 | 1 | MIGRATION_ONLY |
| public | source_dataset_versions | 5 | 3 | 4 | 3 | ACTIVE_RUNTIME |
| public | source_record_targets | 1 | 3 | 1 | 2 | ACTIVE_RUNTIME |
| public | source_records | 6 | 3 | 4 | 3 | ACTIVE_RUNTIME |
| public | split_manifest_registry | 0 | 0 | 2 | 1 | MIGRATION_ONLY |
| public | study_extension_runs | 0 | 0 | 2 | 0 | MIGRATION_ONLY |

References are file-level, collected from `src/`, `scripts/`, `tests/`, `database/`, `configs/`, `project.py`, and `docker-compose.yml`.
