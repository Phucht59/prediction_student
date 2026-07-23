# Database Table Disposition

Final authorization was recorded after backup/restore, canonical load, reconciliation, permissions, rollback, and full tests passed.

| Old table | Rows | Decision | New destination | Drop allowed |
|---|---:|---|---|---|
| public.advisor_decisions | 0 | DROP_EMPTY_REDUNDANT | recommendation.review | True |
| public.cutoff_feature_snapshots | 0 | DROP_EMPTY_REDUNDANT | ml.artifact | True |
| public.expert_review_cases | 0 | DROP_EMPTY_REDUNDANT | recommendation.review | True |
| public.expert_review_ratings | 0 | DROP_EMPTY_REDUNDANT | recommendation.review | True |
| public.ml_evidence_bundles | 0 | DROP_EMPTY_REDUNDANT | ml.artifact | True |
| public.ml_experiment_runs | 0 | DROP_EMPTY_REDUNDANT | ml.run | True |
| public.ml_predictions | 0 | DROP_EMPTY_REDUNDANT | ml.artifact | True |
| public.ml_recommendations | 0 | DROP_EMPTY_REDUNDANT | recommendation.plan | True |
| public.ml_run_metrics | 0 | DROP_EMPTY_REDUNDANT | ml.metric | True |
| public.ml_run_record_splits | 0 | DROP_EMPTY_REDUNDANT | ml.artifact | True |
| public.ml_schema_migrations | 0 | DROP_EMPTY_REDUNDANT | system.schema_migration | True |
| public.prediction_cohorts | 0 | DROP_EMPTY_REDUNDANT | catalog.dataset_version | True |
| public.prediction_snapshots | 0 | DROP_EMPTY_REDUNDANT | ml.artifact | True |
| public.recommendation_action_catalog | 0 | DROP_EMPTY_REDUNDANT | recommendation.policy | True |
| public.recommendation_actions | 0 | DROP_EMPTY_REDUNDANT | recommendation.action | True |
| public.recommendation_feature_registry | 0 | DROP_EMPTY_REDUNDANT | recommendation.policy | True |
| public.recommendation_follow_ups | 0 | DROP_EMPTY_REDUNDANT | recommendation.review | True |
| public.recommendation_goals | 0 | DROP_EMPTY_REDUNDANT | recommendation.plan | True |
| public.recommendation_instances | 0 | DROP_EMPTY_REDUNDANT | recommendation.risk_profile | True |
| public.recommendation_outcomes | 0 | DROP_EMPTY_REDUNDANT | recommendation.review | True |
| public.recommendation_policies | 0 | DROP_EMPTY_REDUNDANT | recommendation.policy | True |
| public.recommendation_revisions | 0 | DROP_EMPTY_REDUNDANT | recommendation.plan | True |
| public.snapshot_record_index | 0 | DROP_EMPTY_REDUNDANT | ml.artifact | True |
| public.source_dataset_files | 0 | DROP_EMPTY_REDUNDANT | catalog.dataset_version | True |
| public.source_dataset_versions | 0 | DROP_EMPTY_REDUNDANT | catalog.dataset_version | True |
| public.source_record_targets | 0 | DROP_EMPTY_REDUNDANT | catalog.record | True |
| public.source_records | 0 | DROP_EMPTY_REDUNDANT | catalog.record | True |
| public.split_manifest_registry | 0 | DROP_EMPTY_REDUNDANT | ml.artifact | True |
| public.study_extension_runs | 0 | DROP_EMPTY_REDUNDANT | ml.run | True |

All removals require the explicit `--confirm-drop-empty-legacy` flag. No non-empty table is authorized for removal.
