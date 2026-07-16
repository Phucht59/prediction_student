BEGIN;

SELECT pg_advisory_xact_lock(hashtext('006_oulad_v3_fair_evidence_registry'));
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '10min';

CREATE TABLE IF NOT EXISTS ml_schema_migrations (
    migration_id TEXT PRIMARY KEY,
    migration_sha256 CHAR(64) NOT NULL CHECK (migration_sha256 ~ '^[0-9a-f]{64}$'),
    source_commit TEXT NOT NULL,
    applied_by TEXT NOT NULL DEFAULT CURRENT_USER,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_evidence_bundles (
    evidence_bundle_id BIGSERIAL PRIMARY KEY,
    study_id TEXT NOT NULL,
    study_version TEXT NOT NULL,
    run_id TEXT NOT NULL,
    parent_run_id TEXT,
    source_commit CHAR(40) NOT NULL CHECK (source_commit ~ '^[0-9a-f]{40}$'),
    protocol_path TEXT NOT NULL,
    protocol_sha256 CHAR(64) NOT NULL CHECK (protocol_sha256 ~ '^[0-9a-f]{64}$'),
    artifact_root TEXT NOT NULL,
    artifact_manifest_sha256 CHAR(64) NOT NULL CHECK (artifact_manifest_sha256 ~ '^[0-9a-f]{64}$'),
    dataset_version_id BIGINT REFERENCES source_dataset_versions(dataset_version_id) ON DELETE RESTRICT,
    forecast_id TEXT NOT NULL,
    target_contract JSONB NOT NULL CHECK (jsonb_typeof(target_contract) = 'object'),
    split_contract JSONB NOT NULL CHECK (jsonb_typeof(split_contract) = 'object'),
    seed_registry JSONB NOT NULL CHECK (jsonb_typeof(seed_registry) = 'array'),
    candidate_registry JSONB NOT NULL CHECK (jsonb_typeof(candidate_registry) = 'object'),
    benchmark_status TEXT NOT NULL CHECK (benchmark_status IN ('development_only', 'reused_observed', 'not_executed')),
    future_benchmark_status TEXT NOT NULL CHECK (future_benchmark_status IN ('NOT_EXECUTED', 'REUSED_OBSERVED')),
    scientific_verdict TEXT NOT NULL,
    validation_status TEXT NOT NULL CHECK (validation_status IN ('PASS', 'FAIL', 'PENDING')),
    runtime_seconds DOUBLE PRECISION CHECK (runtime_seconds IS NULL OR runtime_seconds >= 0),
    environment JSONB NOT NULL CHECK (jsonb_typeof(environment) = 'object'),
    created_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ NOT NULL CHECK (completed_at >= created_at),
    registered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_ml_evidence_bundle_identity UNIQUE (study_id, study_version, run_id)
);

DROP TRIGGER IF EXISTS trg_ml_evidence_bundles_append_only ON ml_evidence_bundles;
CREATE TRIGGER trg_ml_evidence_bundles_append_only
BEFORE UPDATE OR DELETE ON ml_evidence_bundles
FOR EACH ROW EXECUTE FUNCTION reject_append_only_update_delete();

CREATE INDEX IF NOT EXISTS idx_ml_evidence_bundles_source_commit
    ON ml_evidence_bundles(source_commit);
CREATE INDEX IF NOT EXISTS idx_ml_evidence_bundles_dataset
    ON ml_evidence_bundles(dataset_version_id);
CREATE INDEX IF NOT EXISTS idx_source_records_record_key
    ON source_records(dataset_version_id, (raw_payload ->> 'record_key'))
    WHERE raw_payload ? 'record_key';

COMMENT ON TABLE ml_schema_migrations IS 'Append-only checksummed ledger for repository-owned canonical PostgreSQL migrations.';
COMMENT ON TABLE ml_evidence_bundles IS 'Immutable registry of scientific evidence bundles; prediction rows remain in canonical ml_predictions.';
COMMENT ON COLUMN ml_evidence_bundles.future_benchmark_status IS 'NOT_EXECUTED unless a separately governed reused-observed benchmark is explicitly recorded.';
COMMENT ON INDEX idx_source_records_record_key IS 'Supports exact artifact-to-database OOF record-key reproduction checks.';

REVOKE ALL ON ml_schema_migrations, ml_evidence_bundles FROM student_predict_app;
GRANT SELECT ON ml_schema_migrations, ml_evidence_bundles TO student_predict_app;

REVOKE ALL ON
    source_dataset_files,
    prediction_cohorts,
    cutoff_feature_snapshots,
    snapshot_record_index,
    split_manifest_registry,
    study_extension_runs
FROM student_predict_app;

GRANT SELECT ON
    source_dataset_files,
    prediction_cohorts,
    cutoff_feature_snapshots,
    snapshot_record_index,
    split_manifest_registry,
    study_extension_runs
TO student_predict_app;

COMMIT;
