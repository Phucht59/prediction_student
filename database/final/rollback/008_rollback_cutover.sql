BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '5min';
SELECT pg_advisory_xact_lock(hashtext('final_database_v1'));

-- Schema cutback is deliberately non-destructive. It restores runtime lookup
-- to public while leaving final and legacy evidence untouched for diagnosis.
ALTER ROLE student_predict_reader SET search_path = public;
ALTER ROLE student_predict_writer SET search_path = public;
COMMIT;
