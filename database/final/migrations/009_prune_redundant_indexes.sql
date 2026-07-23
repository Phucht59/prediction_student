BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '5min';
SELECT pg_advisory_xact_lock(hashtext('final_database_v1'));

-- The corresponding UNIQUE constraints already provide these left-prefix
-- access paths. Keeping both would exceed the locked 20-index budget.
DROP INDEX IF EXISTS catalog.dataset_version_dataset_idx;
DROP INDEX IF EXISTS ml.model_dataset_idx;
DROP INDEX IF EXISTS ml.artifact_run_kind_idx;
DROP INDEX IF EXISTS recommendation.plan_risk_profile_idx;
COMMIT;
