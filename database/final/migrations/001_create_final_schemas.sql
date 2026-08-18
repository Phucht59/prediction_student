BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '5min';
SELECT pg_advisory_xact_lock(hashtext('final_database_v1'));

CREATE SCHEMA IF NOT EXISTS system;
CREATE SCHEMA IF NOT EXISTS catalog;
CREATE SCHEMA IF NOT EXISTS ml;
CREATE SCHEMA IF NOT EXISTS recommendation;

CREATE TABLE IF NOT EXISTS system.schema_migration (
    filename TEXT PRIMARY KEY,
    version INTEGER NOT NULL UNIQUE CHECK (version > 0),
    sha256 CHAR(64) NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    applied_by TEXT NOT NULL DEFAULT CURRENT_USER,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

COMMENT ON TABLE system.schema_migration IS
    'Immutable checksummed ledger for final database migrations.';
COMMIT;
