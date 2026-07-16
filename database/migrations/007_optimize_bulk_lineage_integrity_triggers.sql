BEGIN;

SELECT pg_advisory_xact_lock(hashtext('007_optimize_bulk_lineage_integrity_triggers'));
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '10min';

CREATE OR REPLACE FUNCTION reject_source_record_batch_after_run()
RETURNS TRIGGER AS $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM inserted_source_records inserted
        JOIN ml_experiment_runs run
          ON run.dataset_version_id = inserted.dataset_version_id
    ) THEN
        RAISE EXCEPTION 'Cannot insert source records after a dataset version has been used by a run'
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION require_running_run_for_insert_batch()
RETURNS TRIGGER AS $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM inserted_run_rows inserted
        LEFT JOIN ml_experiment_runs run
          ON run.run_id = inserted.run_id
         AND run.status = 'running'
        WHERE run.run_id IS NULL
    ) THEN
        RAISE EXCEPTION 'Every inserted % row must reference a running experiment run', TG_TABLE_NAME
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_source_records_sealed_dataset ON source_records;
CREATE TRIGGER trg_source_records_sealed_dataset
AFTER INSERT ON source_records
REFERENCING NEW TABLE AS inserted_source_records
FOR EACH STATEMENT EXECUTE FUNCTION reject_source_record_batch_after_run();

DROP TRIGGER IF EXISTS trg_ml_run_record_splits_running_run ON ml_run_record_splits;
CREATE TRIGGER trg_ml_run_record_splits_running_run
AFTER INSERT ON ml_run_record_splits
REFERENCING NEW TABLE AS inserted_run_rows
FOR EACH STATEMENT EXECUTE FUNCTION require_running_run_for_insert_batch();

DROP TRIGGER IF EXISTS trg_ml_predictions_running_run ON ml_predictions;
CREATE TRIGGER trg_ml_predictions_running_run
AFTER INSERT ON ml_predictions
REFERENCING NEW TABLE AS inserted_run_rows
FOR EACH STATEMENT EXECUTE FUNCTION require_running_run_for_insert_batch();

DROP TRIGGER IF EXISTS trg_ml_run_metrics_running_run ON ml_run_metrics;
CREATE TRIGGER trg_ml_run_metrics_running_run
AFTER INSERT ON ml_run_metrics
REFERENCING NEW TABLE AS inserted_run_rows
FOR EACH STATEMENT EXECUTE FUNCTION require_running_run_for_insert_batch();

COMMENT ON FUNCTION reject_source_record_batch_after_run() IS
    'Set-based equivalent of the sealed-dataset insert guard; validates all rows in each INSERT statement.';
COMMENT ON FUNCTION require_running_run_for_insert_batch() IS
    'Set-based running-run guard for bulk split, prediction, and metric registration.';

COMMIT;
