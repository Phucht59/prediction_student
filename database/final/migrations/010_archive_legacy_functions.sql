BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '5min';
SELECT pg_advisory_xact_lock(hashtext('final_database_v1'));

-- The legacy tables were empty and removed after explicit validation. Preserve
-- their now-orphaned trigger functions outside public and outside the runtime
-- search path for forensic rollback documentation.
DO $$
DECLARE
    function_name TEXT;
BEGIN
    FOREACH function_name IN ARRAY ARRAY[
        'reject_append_only_update_delete',
        'reject_governed_recommendation_mutation',
        'reject_source_record_batch_after_run',
        'reject_source_record_insert_after_run',
        'require_materializable_recommendation_parent_run',
        'require_running_run_by_run_id',
        'require_running_run_for_insert_batch',
        'require_running_run_for_recommendation',
        'validate_ml_experiment_run_insert',
        'validate_ml_experiment_run_update'
    ]
    LOOP
        IF to_regprocedure(format('public.%I()', function_name)) IS NOT NULL THEN
            EXECUTE format(
                'ALTER FUNCTION public.%I() SET SCHEMA legacy_202607',
                function_name
            );
        END IF;
    END LOOP;
END;
$$;
COMMIT;
