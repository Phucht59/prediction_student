# Database Current State Audit

- Audit timestamp: `2026-07-23T15:14:08.269480+00:00`
- Endpoint: `postgresql://<redacted>@localhost:5432/student_predict`
- Database: `student_predict`
- PostgreSQL: `18.4`
- Base tables: **29** (expected 29: **PASS**)
- Views: **0**
- Triggers: **34**
- Exact total rows: **0**
- Database size: **9654 kB**

## Data-bearing tables

None.

## Empty tables

- `public.advisor_decisions`
- `public.cutoff_feature_snapshots`
- `public.expert_review_cases`
- `public.expert_review_ratings`
- `public.ml_evidence_bundles`
- `public.ml_experiment_runs`
- `public.ml_predictions`
- `public.ml_recommendations`
- `public.ml_run_metrics`
- `public.ml_run_record_splits`
- `public.ml_schema_migrations`
- `public.prediction_cohorts`
- `public.prediction_snapshots`
- `public.recommendation_action_catalog`
- `public.recommendation_actions`
- `public.recommendation_feature_registry`
- `public.recommendation_follow_ups`
- `public.recommendation_goals`
- `public.recommendation_instances`
- `public.recommendation_outcomes`
- `public.recommendation_policies`
- `public.recommendation_revisions`
- `public.snapshot_record_index`
- `public.source_dataset_files`
- `public.source_dataset_versions`
- `public.source_record_targets`
- `public.source_records`
- `public.split_manifest_registry`
- `public.study_extension_runs`

## Finding

All 29 legacy application tables are structurally present but contain zero rows. Canonical final artifacts therefore remain the source of truth for result loading. No table is eligible for removal until backup/restore, protocol lock, code cutover, disposition validation, and the explicit empty-table confirmation gate all pass.
