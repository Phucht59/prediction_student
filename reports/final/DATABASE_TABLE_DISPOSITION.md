# Database Table Disposition

This plan covers every live legacy table. No destination is unknown.

| Old table | Rows | Used by code | Final evidence | Decision | New destination | Drop allowed |
|---|---:|---|---|---|---|---|
| public.advisor_decisions | 0 | False | False | MERGE | recommendation.review | False |
| public.cutoff_feature_snapshots | 0 | False | False | MIGRATE_TO_ARTIFACT | ml.artifact | False |
| public.expert_review_cases | 0 | False | False | MERGE | recommendation.review | False |
| public.expert_review_ratings | 0 | False | False | MERGE | recommendation.review | False |
| public.ml_evidence_bundles | 0 | True | False | MERGE | ml.artifact | False |
| public.ml_experiment_runs | 0 | True | False | MERGE | ml.run | False |
| public.ml_predictions | 0 | True | False | MIGRATE_TO_ARTIFACT | ml.artifact | False |
| public.ml_recommendations | 0 | True | False | MERGE | recommendation.plan | False |
| public.ml_run_metrics | 0 | True | False | MERGE | ml.metric | False |
| public.ml_run_record_splits | 0 | True | False | MIGRATE_TO_ARTIFACT | ml.artifact | False |
| public.ml_schema_migrations | 0 | True | False | MERGE | system.schema_migration | False |
| public.prediction_cohorts | 0 | False | False | MERGE | catalog.dataset_version | False |
| public.prediction_snapshots | 0 | False | False | MIGRATE_TO_ARTIFACT | ml.artifact | False |
| public.recommendation_action_catalog | 0 | False | False | MERGE | recommendation.policy | False |
| public.recommendation_actions | 0 | False | False | MERGE | recommendation.action | False |
| public.recommendation_feature_registry | 0 | False | False | MERGE | recommendation.policy | False |
| public.recommendation_follow_ups | 0 | False | False | MERGE | recommendation.review | False |
| public.recommendation_goals | 0 | False | False | MERGE | recommendation.plan | False |
| public.recommendation_instances | 0 | False | False | MERGE | recommendation.risk_profile | False |
| public.recommendation_outcomes | 0 | False | False | MERGE | recommendation.review | False |
| public.recommendation_policies | 0 | False | False | MERGE | recommendation.policy | False |
| public.recommendation_revisions | 0 | False | False | MERGE | recommendation.plan | False |
| public.snapshot_record_index | 0 | False | False | MIGRATE_TO_ARTIFACT | ml.artifact | False |
| public.source_dataset_files | 0 | False | False | MERGE | catalog.dataset_version | False |
| public.source_dataset_versions | 0 | True | False | RENAME | catalog.dataset_version | False |
| public.source_record_targets | 0 | True | False | MERGE | catalog.record | False |
| public.source_records | 0 | True | False | RENAME | catalog.record | False |
| public.split_manifest_registry | 0 | False | False | MIGRATE_TO_ARTIFACT | ml.artifact | False |
| public.study_extension_runs | 0 | False | False | MERGE | ml.run | False |

All legacy tables are empty, but `drop_allowed` remains false during audit. The migration CLI may change that outcome only after backup and restore validation, runtime reference removal, dependency checks, and the explicit `--confirm-drop-empty-legacy` flag.
