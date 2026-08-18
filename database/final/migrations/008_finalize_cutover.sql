BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '5min';
SELECT pg_advisory_xact_lock(hashtext('final_database_v1'));

CREATE SCHEMA IF NOT EXISTS legacy_202607;
REVOKE ALL ON SCHEMA legacy_202607 FROM PUBLIC;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;

COMMENT ON SCHEMA system IS 'Final migration control plane.';
COMMENT ON SCHEMA catalog IS 'Canonical datasets, versions, and final cohort records.';
COMMENT ON SCHEMA ml IS 'Canonical model, run, artifact, and metric metadata.';
COMMENT ON SCHEMA recommendation IS 'Risk profiles and final recommendation entities.';
COMMENT ON SCHEMA legacy_202607 IS
    'Read-only legacy structures retained only when non-empty.';
COMMIT;
