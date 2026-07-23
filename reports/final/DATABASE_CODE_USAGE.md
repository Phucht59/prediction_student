# Database Code Usage

| Schema | Table | Read refs | Write refs | Migration refs | Test refs | Runtime status |
|---|---|---:|---:|---:|---:|---|
| public | advisor_decisions | 0 | 0 | 3 | 0 | MIGRATION_ONLY |
| public | cutoff_feature_snapshots | 0 | 0 | 4 | 1 | MIGRATION_ONLY |
| public | expert_review_cases | 0 | 0 | 3 | 0 | MIGRATION_ONLY |
| public | expert_review_ratings | 0 | 0 | 3 | 0 | MIGRATION_ONLY |
| public | ml_evidence_bundles | 2 | 1 | 3 | 1 | MIGRATION_ONLY |
| public | ml_experiment_runs | 8 | 3 | 5 | 2 | MIGRATION_ONLY |
| public | ml_predictions | 7 | 3 | 6 | 3 | MIGRATION_ONLY |
| public | ml_recommendations | 2 | 2 | 4 | 2 | MIGRATION_ONLY |
| public | ml_run_metrics | 2 | 3 | 4 | 1 | MIGRATION_ONLY |
| public | ml_run_record_splits | 6 | 3 | 4 | 1 | MIGRATION_ONLY |
| public | ml_schema_migrations | 2 | 1 | 3 | 1 | MIGRATION_ONLY |
| public | prediction_cohorts | 0 | 0 | 4 | 1 | MIGRATION_ONLY |
| public | prediction_snapshots | 0 | 0 | 4 | 0 | MIGRATION_ONLY |
| public | recommendation_action_catalog | 0 | 0 | 3 | 0 | MIGRATION_ONLY |
| public | recommendation_actions | 0 | 0 | 3 | 0 | MIGRATION_ONLY |
| public | recommendation_feature_registry | 0 | 0 | 3 | 0 | MIGRATION_ONLY |
| public | recommendation_follow_ups | 0 | 0 | 3 | 0 | MIGRATION_ONLY |
| public | recommendation_goals | 0 | 0 | 3 | 0 | MIGRATION_ONLY |
| public | recommendation_instances | 0 | 0 | 3 | 0 | MIGRATION_ONLY |
| public | recommendation_outcomes | 0 | 0 | 3 | 0 | MIGRATION_ONLY |
| public | recommendation_policies | 0 | 0 | 4 | 0 | MIGRATION_ONLY |
| public | recommendation_revisions | 0 | 0 | 4 | 0 | MIGRATION_ONLY |
| public | snapshot_record_index | 0 | 0 | 4 | 1 | MIGRATION_ONLY |
| public | source_dataset_files | 0 | 0 | 4 | 1 | MIGRATION_ONLY |
| public | source_dataset_versions | 5 | 3 | 6 | 3 | MIGRATION_ONLY |
| public | source_record_targets | 1 | 3 | 3 | 2 | MIGRATION_ONLY |
| public | source_records | 6 | 3 | 6 | 3 | MIGRATION_ONLY |
| public | split_manifest_registry | 0 | 0 | 4 | 1 | MIGRATION_ONLY |
| public | study_extension_runs | 0 | 0 | 4 | 0 | MIGRATION_ONLY |

Historical training/evidence modules are explicitly classified as non-runtime. The final runtime imports only the version-neutral repositories under `src/database/`.
