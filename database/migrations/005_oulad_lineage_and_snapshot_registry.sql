BEGIN;

CREATE TABLE IF NOT EXISTS source_dataset_files (
    source_dataset_file_id BIGSERIAL PRIMARY KEY,
    dataset_version_id BIGINT NOT NULL REFERENCES source_dataset_versions(dataset_version_id),
    logical_name TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    sha256 CHAR(64) NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    row_count BIGINT NOT NULL CHECK (row_count >= 0),
    schema_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (dataset_version_id, logical_name, sha256)
);

CREATE TABLE IF NOT EXISTS prediction_cohorts (
    prediction_cohort_id BIGSERIAL PRIMARY KEY,
    dataset_version_id BIGINT NOT NULL REFERENCES source_dataset_versions(dataset_version_id),
    cohort_id TEXT NOT NULL UNIQUE,
    forecast_id TEXT NOT NULL,
    cutoff_contract JSONB NOT NULL,
    cohort_hash CHAR(64) NOT NULL CHECK (cohort_hash ~ '^[0-9a-f]{64}$'),
    record_count BIGINT NOT NULL CHECK (record_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cutoff_feature_snapshots (
    cutoff_feature_snapshot_id BIGSERIAL PRIMARY KEY,
    prediction_cohort_id BIGINT NOT NULL REFERENCES prediction_cohorts(prediction_cohort_id),
    feature_contract_hash CHAR(64) NOT NULL CHECK (feature_contract_hash ~ '^[0-9a-f]{64}$'),
    target_hash CHAR(64) NOT NULL CHECK (target_hash ~ '^[0-9a-f]{64}$'),
    parquet_relative_path TEXT NOT NULL,
    parquet_sha256 CHAR(64) NOT NULL CHECK (parquet_sha256 ~ '^[0-9a-f]{64}$'),
    sequence_length INTEGER NOT NULL CHECK (sequence_length > 0),
    channel_order JSONB NOT NULL,
    channel_order_hash CHAR(64) NOT NULL CHECK (channel_order_hash ~ '^[0-9a-f]{64}$'),
    row_count BIGINT NOT NULL CHECK (row_count >= 0),
    status TEXT NOT NULL CHECK (status IN ('materialized', 'validated', 'deprecated')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS snapshot_record_index (
    cutoff_feature_snapshot_id BIGINT NOT NULL REFERENCES cutoff_feature_snapshots(cutoff_feature_snapshot_id),
    record_id CHAR(64) NOT NULL CHECK (record_id ~ '^[0-9a-f]{64}$'),
    row_index BIGINT NOT NULL CHECK (row_index >= 0),
    PRIMARY KEY (cutoff_feature_snapshot_id, record_id),
    UNIQUE (cutoff_feature_snapshot_id, row_index)
);

CREATE TABLE IF NOT EXISTS split_manifest_registry (
    split_manifest_registry_id BIGSERIAL PRIMARY KEY,
    prediction_cohort_id BIGINT NOT NULL REFERENCES prediction_cohorts(prediction_cohort_id),
    split_role TEXT NOT NULL CHECK (split_role IN ('historical_development', 'outer_cv', 'inner_cv', 'future_presentation', 'excluded_overlap', 'descriptive_only')),
    manifest_relative_path TEXT NOT NULL,
    manifest_sha256 CHAR(64) NOT NULL CHECK (manifest_sha256 ~ '^[0-9a-f]{64}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (prediction_cohort_id, split_role, manifest_sha256)
);

DO $$
DECLARE
    table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'source_dataset_files',
        'prediction_cohorts',
        'cutoff_feature_snapshots',
        'snapshot_record_index',
        'split_manifest_registry'
    ]
    LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS trg_%I_append_only ON %I', table_name, table_name);
        EXECUTE format(
            'CREATE TRIGGER trg_%I_append_only BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION reject_append_only_update_delete()',
            table_name,
            table_name
        );
    END LOOP;
END $$;

COMMIT;
