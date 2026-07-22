BEGIN;

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
END $$;

REVOKE ALL ON SCHEMA source, education, feature, experiment, evaluation, recommendation FROM PUBLIC;
GRANT USAGE ON SCHEMA source, education, feature, experiment, evaluation, recommendation TO student_predict_reader, student_predict_writer;
GRANT SELECT ON ALL TABLES IN SCHEMA source, education, feature, experiment, evaluation, recommendation TO student_predict_reader;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA source, education, feature, experiment, evaluation, recommendation TO student_predict_writer;
REVOKE UPDATE, DELETE ON source.schema_migration FROM student_predict_writer;
REVOKE ALL ON ALL TABLES IN SCHEMA source, education, feature, experiment, evaluation, recommendation FROM PUBLIC;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA source, education, feature, experiment, evaluation, recommendation TO student_predict_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA source, education, feature, experiment, evaluation, recommendation GRANT SELECT ON TABLES TO student_predict_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA source, education, feature, experiment, evaluation, recommendation GRANT SELECT, INSERT, UPDATE ON TABLES TO student_predict_writer;

COMMIT;
