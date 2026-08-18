BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '5min';
SELECT pg_advisory_xact_lock(hashtext('final_database_v1'));

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'student_predict_migrator') THEN
        CREATE ROLE student_predict_migrator NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'student_predict_writer') THEN
        CREATE ROLE student_predict_writer NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'student_predict_reader') THEN
        CREATE ROLE student_predict_reader NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE;
    END IF;
END;
$$;

REVOKE ALL ON SCHEMA system, catalog, ml, recommendation FROM PUBLIC;
GRANT USAGE ON SCHEMA catalog, ml, recommendation TO student_predict_reader, student_predict_writer;
GRANT USAGE ON SCHEMA system, catalog, ml, recommendation TO student_predict_migrator;

GRANT SELECT ON ALL TABLES IN SCHEMA catalog, ml, recommendation TO student_predict_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA catalog, ml, recommendation TO student_predict_writer;
GRANT INSERT ON recommendation.risk_profile, recommendation.plan,
    recommendation.action, recommendation.review TO student_predict_writer;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA recommendation TO student_predict_writer;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA system, catalog, ml, recommendation TO student_predict_migrator;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA system, catalog, ml, recommendation TO student_predict_migrator;

COMMIT;
