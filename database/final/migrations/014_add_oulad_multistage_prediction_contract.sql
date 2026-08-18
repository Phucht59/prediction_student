BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '5min';
SELECT pg_advisory_xact_lock(hashtext('oulad_unified_multistage_v1'));

ALTER TABLE ml.prediction
    ADD COLUMN IF NOT EXISTS cohort TEXT NOT NULL DEFAULT 'OPERATIONAL_RISK_SET',
    ADD COLUMN IF NOT EXISTS threshold_policy TEXT NOT NULL DEFAULT 'RAW_PROBABILITY',
    ADD COLUMN IF NOT EXISTS cutoff_day INTEGER,
    ADD COLUMN IF NOT EXISTS progress_fraction DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS true_label TEXT;

ALTER TABLE ml.prediction
    DROP CONSTRAINT IF EXISTS prediction_run_id_record_pk_prediction_stage_key;
DROP INDEX IF EXISTS ml.prediction_run_id_record_pk_prediction_stage_key;
DROP INDEX IF EXISTS ml.prediction_natural_key_idx;
CREATE UNIQUE INDEX prediction_natural_key_idx ON ml.prediction(
    run_id, record_pk, prediction_stage, cohort, threshold_policy
);

CREATE INDEX IF NOT EXISTS prediction_stage_cohort_idx
    ON ml.prediction(prediction_stage, cohort, threshold_policy);

COMMIT;
