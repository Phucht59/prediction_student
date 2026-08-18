BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '5min';
SELECT pg_advisory_xact_lock(hashtext('final_database_v1'));

-- The locked read-only audit found all 29 legacy tables empty. Abort rather
-- than silently choosing between new legacy rows and canonical artifacts.
DO $$
DECLARE
    table_name TEXT;
    row_count BIGINT;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'advisor_decisions', 'cutoff_feature_snapshots', 'expert_review_cases',
        'expert_review_ratings', 'ml_evidence_bundles', 'ml_experiment_runs',
        'ml_predictions', 'ml_recommendations', 'ml_run_metrics',
        'ml_run_record_splits', 'ml_schema_migrations', 'prediction_cohorts',
        'prediction_snapshots', 'recommendation_action_catalog',
        'recommendation_actions', 'recommendation_feature_registry',
        'recommendation_follow_ups', 'recommendation_goals',
        'recommendation_instances', 'recommendation_outcomes',
        'recommendation_policies', 'recommendation_revisions',
        'snapshot_record_index', 'source_dataset_files',
        'source_dataset_versions', 'source_record_targets', 'source_records',
        'split_manifest_registry', 'study_extension_runs'
    ]
    LOOP
        IF to_regclass(format('public.%I', table_name)) IS NOT NULL THEN
            EXECUTE format('SELECT count(*) FROM public.%I', table_name) INTO row_count;
            IF row_count <> 0 THEN
                RAISE EXCEPTION 'STOP_MIGRATION_CONFLICT: %.% has % rows after locked audit',
                    'public', table_name, row_count;
            END IF;
        END IF;
    END LOOP;
END;
$$;
COMMIT;
